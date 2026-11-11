#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

BlockTreeEvictor::BlockTreeEvictor(BlockTree*                     tree,
                                   EvictionPolicy                 device_policy,
                                   EvictionPolicy                 host_policy,
                                   EvictionPolicy                 disk_policy,
                                   const BlockTransferDispatcher* transfer_dispatcher,
                                   BlockTreeTaskPool*             task_pool,
                                   BlockTreeCacheMetricsReporter& metrics_reporter,
                                   std::mutex&                    mutex,
                                   int                            host_timeout_ms,
                                   int                            disk_timeout_ms,
                                   size_t                         max_descriptors_per_batch,
                                   IsTierEnabledFn                is_tier_enabled,
                                   SettledFn                      settled,
                                   size_t                         disk_cache_evict_max_pending_bytes):
    tree_(tree),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    is_tier_enabled_(std::move(is_tier_enabled)),
    settled_(std::move(settled)),
    task_runner_(std::make_unique<EvictionTaskRunner>(tree->groupSets(), transfer_dispatcher)),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    disk_cache_evict_max_pending_bytes_(disk_cache_evict_max_pending_bytes) {
    // GroupSetFactory has already validated that group_set_id equals the vector
    // position. Own one heap per (group resource, tier).
    heaps_.resize(tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        auto& tier_heaps  = heaps_[group_set_id];
        tier_heaps.device = std::make_unique<EvictionHeap>(device_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(host_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(disk_policy);
        tier_heaps.next_device_reclaim_log.resize(tree_->groupSets()[group_set_id]->devicePools().size());
    }
}

BlockTreeEvictor::~BlockTreeEvictor() {
    std::lock_guard<std::mutex> lock(pending_release_mutex_);
    for (const auto& [pool, count] : pending_release_counts_) {
        RTP_LLM_CHECK_WITH_INFO(count == 0,
                                "pending eviction releases remain: pool=%s address=%p count=%zu",
                                pool ? pool->poolName().c_str() : "<null>",
                                static_cast<void*>(pool),
                                count);
    }
}

EvictionHeap* BlockTreeEvictor::heapFor(size_t group_set_id, Tier tier) const {
    const auto& tier_heaps = heaps_[group_set_id];
    switch (tier) {
        case Tier::DEVICE:
            return tier_heaps.device.get();
        case Tier::HOST:
            return tier_heaps.host.get();
        case Tier::DISK:
            return tier_heaps.disk.get();
        default:
            return nullptr;
    }
}

bool BlockTreeEvictor::isEvictable(TreeNode* node, size_t group_set_id, Tier source_tier) const {
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    return !node->is_resident && resource.transfer_state == GroupSetTransferState::IDLE
           && resource.getTopTier() == source_tier
           && (tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL
               || tree_->isLeafAtTier(node, group_set_id, source_tier));
}

void BlockTreeEvictor::suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier) {
    if (source_tier != Tier::NONE) {
        heapFor(group_set_id, source_tier)->erase(node);
    }
}

void BlockTreeEvictor::admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier) {
    if (node->parent == nullptr || node->is_resident || target_tier == Tier::NONE) {
        return;
    }
    const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
    EvictionHeap*      heap      = heapFor(group_set_id, target_tier);
    if (group_set->groupType() == CacheGroupType::FULL && !tree_->isLeafAtTier(node, group_set_id, target_tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, node->group_set_resources[group_set_id].candidate_meta);
}

void BlockTreeEvictor::updateFullCandidate(TreeNode* parent) {
    if (parent == nullptr || parent->parent == nullptr) {
        return;
    }
    for (auto& group_set : tree_->groupSets()) {
        updateFullCandidate(parent, group_set->groupSetId());
    }
}

void BlockTreeEvictor::updateFullCandidate(TreeNode* node, size_t group_set_id) {
    if (node == nullptr || node->parent == nullptr
        || tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL) {
        return;
    }
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier              tier     = resource.getTopTier();
    if (resource.transfer_state != GroupSetTransferState::IDLE) {
        suspendCandidate(node, group_set_id, tier);
        return;
    }
    admitCandidate(node, group_set_id, tier);
}

void BlockTreeEvictor::onLoaded(TreeNode* node, size_t group_set_id) {
    GroupSetResource& resource                    = node->group_set_resources[group_set_id];
    const int64_t     tier_enter_time_us          = currentTimeUs();
    resource.candidate_meta.admission_seq         = ++admission_seq_;
    resource.candidate_meta.tier_enter_time_us    = tier_enter_time_us;
    resource.candidate_meta.last_access_time_us   = tier_enter_time_us;
    resource.candidate_meta.tier_last_hit_time_us = 0;
    admitCandidate(node, group_set_id, Tier::DEVICE);
    updateFullCandidate(node->parent, group_set_id);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInserted(const BlockTreeInsertResult& result) {
    for (TreeNode* node : result.newly_resident_nodes) {
        eraseNodeFromAllHeaps(node);
    }
    for (const auto& candidate : result.replaced_candidates) {
        suspendCandidate(candidate.node, candidate.group_set_id, candidate.source_tier);
    }
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const int64_t     insert_time_us              = currentTimeUs();
            GroupSetResource& resource                    = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq       = nextAccessSeq();
            resource.candidate_meta.admission_seq         = ++admission_seq_;
            resource.candidate_meta.hit_count             = 0;
            resource.candidate_meta.tier_last_hit_time_us = 0;
            resource.candidate_meta.last_access_time_us   = insert_time_us;
            resource.candidate_meta.tier_enter_time_us    = insert_time_us;
            admitCandidate(adopted.first, group_set_id, resource.getTopTier());
            updateFullCandidate(adopted.first->parent, group_set_id);
        }
    }

    for (TreeNode* node : result.inserted_nodes) {
        const uint64_t access         = nextAccessSeq();
        const uint64_t admit          = ++admission_seq_;
        const int64_t  insert_time_us = currentTimeUs();
        for (auto& group_set : tree_->groupSets()) {
            const size_t      group_set_id                = group_set->groupSetId();
            GroupSetResource& resource                    = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq       = access;
            resource.candidate_meta.admission_seq         = admit;
            resource.candidate_meta.hit_count             = 0;
            resource.candidate_meta.tier_last_hit_time_us = 0;
            resource.candidate_meta.last_access_time_us   = insert_time_us;
            resource.candidate_meta.tier_enter_time_us    = insert_time_us;
            admitCandidate(node, group_set_id, resource.getTopTier());
        }
    }

    updateFullCandidate(result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent);
}

uint64_t BlockTreeEvictor::nextAccessSeq() {
    return ++access_seq_;
}

void BlockTreeEvictor::onMatched(TreeNode* node, size_t group_set_id) {
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier              tier     = resource.getTopTier();
    if (tier != Tier::NONE) {
        heapFor(group_set_id, tier)->updateIfPresent(node, resource.candidate_meta);
    }
}

CandidateStats BlockTreeEvictor::candidateStats() const {
    CandidateStats stats;
    for (const auto& tier_heaps : heaps_) {
        stats.device_candidates += tier_heaps.device->size();
        stats.host_candidates += tier_heaps.host->size();
        stats.disk_candidates += tier_heaps.disk->size();
    }
    return stats;
}

size_t BlockTreeEvictor::candidateCount(size_t group_set_id, Tier tier) const {
    const EvictionHeap* heap = heapFor(group_set_id, tier);
    return heap == nullptr ? 0 : heap->size();
}

std::vector<TreeNode*> BlockTreeEvictor::candidateNodes(size_t group_set_id, Tier tier) const {
    const EvictionHeap* heap = heapFor(group_set_id, tier);
    return heap == nullptr ? std::vector<TreeNode*>{} : heap->nodes();
}

VictimSelectionMetrics BlockTreeEvictor::takeVictimSelectionMetrics() const {
    return victim_selection_metrics_.take();
}

size_t BlockTreeEvictor::reclaimDeviceBlocksLocked(size_t group_set_id, size_t member_group_id, size_t num_blocks) {
    const GroupSetPtr&   group_set          = tree_->groupSets()[group_set_id];
    const auto&          device_pool        = group_set->devicePools()[member_group_id];
    const size_t         initial_free       = device_pool->freeBlocksNum();
    size_t               reclaimed          = 0;
    bool                 eviction_triggered = false;
    EvictionBatchContext eviction_batch_context;
    while (reclaimed < num_blocks) {
        if (!dropLocked(group_set_id,
                        Tier::DEVICE,
                        /*notify_settled=*/true,
                        VictimSelectionTrigger::ON_DEMAND,
                        &eviction_batch_context)) {
            break;
        }
        eviction_triggered        = true;
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    if (eviction_triggered) {
        metrics_reporter_->reportEvictionTriggered(Tier::DEVICE, group_set->groupType(), /*force_drop=*/true);
    }
    metrics_reporter_->reportEvictionBlocks(Tier::DEVICE,
                                            group_set->groupType(),
                                            /*force_drop=*/true,
                                            num_blocks,
                                            std::min(reclaimed, num_blocks));
    if (reclaimed == 0 && num_blocks > 0) {
        logZeroDeviceReclaimLocked(
            group_set_id, member_group_id, num_blocks, eviction_batch_context.rejected_candidates.size());
    }
    return reclaimed;
}

void BlockTreeEvictor::logZeroDeviceReclaimLocked(size_t group_set_id,
                                                  size_t member_group_id,
                                                  size_t required_blocks,
                                                  size_t rejected_candidates) {
    auto&      tier_heaps = heaps_[group_set_id];
    auto&      next_log   = tier_heaps.next_device_reclaim_log[member_group_id];
    const auto now        = std::chrono::steady_clock::now();
    if (now < next_log) {
        return;
    }
    next_log = now + std::chrono::seconds(5);

    const auto& group_set  = tree_->groupSets()[group_set_id];
    const auto& pool       = group_set->devicePools()[member_group_id];
    const auto  stats      = pool->referenceStats();
    const auto& heap       = *tier_heaps.device;
    const auto  candidates = heap.size();
    // Frozen state is read from every member pool after the diagnostic rate limit.
    const bool has_unfrozen_candidate = heap.best([&group_set, group_set_id](TreeNode* node) {
        const auto& blocks = node->group_set_resources[group_set_id].device_blocks;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (!isNullBlockIdx(blocks[i]) && group_set->devicePools()[i]->isFrozen(blocks[i])) {
                return false;
            }
        }
        return true;
    }).has_value();
    // A candidate covers at most one block in the requested member pool. The
    // difference is only a lower bound, since candidates may share blocks or hold other references.
    const auto cache_only_outside_heap_min =
        stats.tree.cache_only_blocks - std::min(stats.tree.cache_only_blocks, candidates);
    size_t pending_eviction_releases = 0;
    {
        std::lock_guard<std::mutex> lock(pending_release_mutex_);
        const auto                  pending = pending_release_counts_.find(pool.get());
        if (pending != pending_release_counts_.end()) {
            pending_eviction_releases = pending->second;
        }
    }

    // Reference counts can overlap. Candidate counts are tree nodes, not physical blocks.
    RTP_LLM_LOG_WARNING(
        "BTC on-demand DEVICE reclaim returned zero: pool=%s group_set=%zu group_type=%s required_blocks=%zu "
        "total_blocks=%zu free_blocks=%zu active_blocks=%zu cache_only_blocks=%zu cache_only_outside_heap_min=%zu "
        "device_candidate_nodes=%zu has_unfrozen_candidate=%d rejected_candidate_nodes=%zu pending_eviction_releases=%zu "
        "cache_ref_blocks=%zu load_ref_blocks=%zu eviction_target_ref_blocks=%zu "
        "frozen_ref_blocks=%zu request_ref_blocks=%zu remote_connector_ref_blocks=%zu pd_connector_ref_blocks=%zu",
        pool->poolName().c_str(),
        group_set_id,
        metricCacheGroupTypeName(group_set->groupType()),
        required_blocks,
        stats.tree.total_blocks,
        stats.tree.free_blocks,
        stats.tree.active_blocks,
        stats.tree.cache_only_blocks,
        cache_only_outside_heap_min,
        candidates,
        has_unfrozen_candidate,
        rejected_candidates,
        pending_eviction_releases,
        stats.tree.blocksWithRef(BlockTreeRefType::CACHE),
        stats.tree.blocksWithRef(BlockTreeRefType::LOAD),
        stats.tree.blocksWithRef(BlockTreeRefType::EVICTION),
        stats.tree.blocksWithRef(BlockTreeRefType::FROZEN),
        stats.request_ref_blocks,
        stats.remote_connector_ref_blocks,
        stats.pd_connector_ref_blocks);
}

bool BlockTreeEvictor::dropLocked(size_t                 group_set_id,
                                  Tier                   source_tier,
                                  bool                   notify_settled,
                                  VictimSelectionTrigger trigger,
                                  EvictionBatchContext*  eviction_batch_context) {
    std::optional<TransferDescriptor> eviction_desc =
        chooseVictim(group_set_id, source_tier, /*force_drop=*/true, eviction_batch_context, trigger);
    if (!eviction_desc.has_value()) {
        return false;
    }
    runDropTask(std::move(*eviction_desc),
                notify_settled,
                trigger == VictimSelectionTrigger::WATERMARK ? "watermark" : "force_drop",
                eviction_batch_context);
    return true;
}

bool BlockTreeEvictor::batchDropLocked(size_t  group_set_id,
                                       Tier    source_tier,
                                       size_t  max_victim_count,
                                       size_t& scheduled_count) {
    std::vector<IBlockPool*> source_pools;
    const GroupSetPtr&       group_set = tree_->groupSets()[group_set_id];
    if (source_tier == Tier::DEVICE) {
        for (const DeviceBlockPoolPtr& pool : group_set->devicePools()) {
            source_pools.push_back(pool.get());
        }
    } else if (source_tier == Tier::HOST && group_set->hostPool() != nullptr) {
        source_pools.push_back(group_set->hostPool().get());
    } else if (source_tier == Tier::DISK && group_set->diskPool() != nullptr) {
        source_pools.push_back(group_set->diskPool().get());
    }

    scheduled_count        = 0;
    bool   processed       = false;
    size_t remaining_count = max_victim_count;
    EvictionBatchContext eviction_batch_context;
    while (remaining_count > 0) {
        std::vector<size_t> used_before;
        used_before.reserve(source_pools.size());
        for (IBlockPool* pool : source_pools) {
            used_before.push_back(pool->usedBlocksNum());
        }
        if (!dropLocked(group_set_id, source_tier, false, VictimSelectionTrigger::WATERMARK, &eviction_batch_context)) {
            break;
        }
        processed                  = true;
        size_t physically_released = std::numeric_limits<size_t>::max();
        for (size_t pool_index = 0; pool_index < source_pools.size(); ++pool_index) {
            const size_t used_after = source_pools[pool_index]->usedBlocksNum();
            physically_released =
                std::min(physically_released, used_before[pool_index] - std::min(used_before[pool_index], used_after));
        }
        scheduled_count += std::min(physically_released, remaining_count);
        const size_t progress = std::min(std::max<size_t>(1, physically_released), remaining_count);
        remaining_count -= progress;
    }
    if (processed) {
        settled_(false);
    }
    return processed;
}

bool BlockTreeEvictor::submitEvictionTask(std::vector<TransferDescriptor>     descriptors,
                                          std::vector<EvictionTimingSnapshot> timings) {
    const TransferDescriptor& descriptor = descriptors.front();
    const auto                timeout    = std::chrono::milliseconds(
        descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK ? disk_timeout_ms_ :
                                                                                       host_timeout_ms_);
    auto task_ptr =
        std::make_shared<EvictionTransferTask>(TransferTask(std::move(descriptors), timeout), std::move(timings));
    for (size_t i = 0; i < task_ptr->descriptors().size(); ++i) {
        const auto& desc = task_ptr->descriptors()[i];
        heaps_[desc.group_set_id].pending_timings.emplace(desc.node, task_ptr->timings[i]);
    }
    updatePendingRelease(task_ptr->descriptors(), true);
    const int64_t queue_begin = currentTimeUs();
    auto          on_timeout  = [this, task_ptr, queue_begin]() {
        metrics_reporter_->reportQueueWaitMetric(false,
                                                 "business",
                                                 cacheTransferOperationName(CacheTransferOperation::EVICT),
                                                 Tier::NONE,
                                                 Tier::NONE,
                                                 currentTimeUs() - queue_begin);
        RTP_LLM_LOG_WARNING("eviction expired in business queue, source=%s target=%s descriptors=%zu",
                            tierName(task_ptr->descriptors().front().source_tier),
                            tierName(task_ptr->descriptors().front().target_tier),
                            task_ptr->descriptors().size());
        scheduleEvictionSettlement(task_ptr, false);
    };
    const auto deadline  = task_ptr->transfer_task.deadline();
    const bool submitted = task_pool_->submit(
        BlockTreeTaskClass::BACKGROUND,
        [this, task_ptr, queue_begin]() {
            metrics_reporter_->reportQueueWaitMetric(false,
                                                     "business",
                                                     cacheTransferOperationName(CacheTransferOperation::EVICT),
                                                     Tier::NONE,
                                                     Tier::NONE,
                                                     currentTimeUs() - queue_begin);
            runEvictionTask(task_ptr);
        },
        deadline,
        std::move(on_timeout));
    if (!submitted) {
        const auto& first  = task_ptr->descriptors().front();
        const auto  queues = task_pool_->queueSizes();
        RTP_LLM_INTERVAL_LOG(5,
                             WARN,
                             "eviction submission rejected: group_set=%zu source=%s target=%s blocks=%zu "
                             "queue_load=%zu queue_background=%zu queue_completion=%zu",
                             first.group_set_id,
                             tierName(first.source_tier),
                             tierName(first.target_tier),
                             task_ptr->descriptors().size(),
                             queues.load,
                             queues.background,
                             queues.completion);
        for (const auto& desc : task_ptr->descriptors()) {
            heaps_[desc.group_set_id].pending_timings.erase(desc.node);
        }
        updatePendingRelease(task_ptr->descriptors(), false);
        rollbackTransferLocked(task_ptr->descriptors());
        metrics_reporter_->reportEvictionFinished(*task_ptr, tree_->groupSets(), false);
        return false;
    }
    return true;
}

BlockTreeEvictor::BatchEvictResult
BlockTreeEvictor::batchEvictLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count) {
    if (max_victim_count == 0) {
        return {};
    }
    const Tier target_tier = watermarkTargetTier(source_tier);
    if (target_tier == Tier::NONE) {
        size_t     scheduled_count = 0;
        const bool direct_progress = batchDropLocked(group_set_id, source_tier, max_victim_count, scheduled_count);
        return {/*direct_progress=*/direct_progress,
                /*async_submitted=*/false,
                /*scheduled_count=*/scheduled_count};
    }

    std::vector<TransferDescriptor>     descriptors;
    std::vector<EvictionTimingSnapshot> timings;
    EvictionBatchContext                eviction_batch_context;
    for (size_t victim_count = 0; victim_count < max_victim_count; ++victim_count) {
        auto eviction_desc = chooseVictim(group_set_id,
                                          source_tier,
                                          /*force_drop=*/false,
                                          &eviction_batch_context,
                                          VictimSelectionTrigger::WATERMARK);
        if (!eviction_desc.has_value()) {
            break;
        }
        timings.emplace_back(eviction_desc->node->group_set_resources[eviction_desc->group_set_id].candidate_meta);
        reserveSource({*eviction_desc});
        descriptors.push_back(std::move(*eviction_desc));
    }
    if (descriptors.empty()) {
        return {/*direct_progress=*/false,
                /*async_submitted=*/false,
                /*scheduled_count=*/0};
    }

    const auto& group_set     = tree_->groupSets()[group_set_id];
    auto        target_blocks = group_set->allocateBlocks(descriptors.size(), target_tier, BlockTreeRefType::EVICTION);
    if (!target_blocks.has_value()) {
        const IBlockPool* pool = target_tier == Tier::HOST ?
                                     static_cast<const IBlockPool*>(group_set->hostPool().get()) :
                                     static_cast<const IBlockPool*>(group_set->diskPool().get());
        RTP_LLM_INTERVAL_LOG(5,
                             WARN,
                             "eviction target allocation failed: group_set=%zu source=%s target=%s blocks=%zu "
                             "target_pool=%s",
                             group_set_id,
                             tierName(source_tier),
                             tierName(target_tier),
                             descriptors.size(),
                             pool ? pool->debugString().c_str() : "<null>");
        rollbackTransferLocked(descriptors);
        metrics_reporter_->reportEvictionFinished(
            EvictionTransferTask(TransferTask(std::move(descriptors), std::chrono::milliseconds(0)),
                                 std::move(timings)),
            tree_->groupSets(),
            false);
        return {/*direct_progress=*/false,
                /*async_submitted=*/false,
                /*scheduled_count=*/0};
    }
    for (size_t desc_index = 0; desc_index < descriptors.size(); ++desc_index) {
        descriptors[desc_index].target_blocks = {(*target_blocks)[desc_index]};
    }
    const size_t async_scheduled_count = descriptors.size();
    const bool   submitted             = submitEvictionTask(std::move(descriptors), std::move(timings));
    return {/*direct_progress=*/false,
            /*async_submitted=*/submitted,
            /*scheduled_count=*/submitted ? async_scheduled_count : 0};
}

void BlockTreeEvictor::runEvictionTask(std::shared_ptr<const EvictionTransferTask> task) noexcept {
    try {
        task_runner_->runTransfer(task, *metrics_reporter_, [this, task](ErrorInfo error) noexcept {
            completeEvictionTask(task, std::move(error));
        });
        return;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
    completeEvictionTask(std::move(task), ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "eviction task runner exception"));
}

void BlockTreeEvictor::completeEvictionTask(std::shared_ptr<const EvictionTransferTask> task,
                                            ErrorInfo                                   error) noexcept {
    scheduleEvictionSettlement(std::move(task), error.ok());
}

void BlockTreeEvictor::scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task,
                                                  bool                                        success) noexcept {
    auto observation = CompletionStageContext::forTransfer(*metrics_reporter_, task->transfer_task, "evict", success);
    auto timers      = CompletionStageTimers::create(observation);
    const int64_t queue_begin = currentTimeUs();
    auto settle = [this, task = std::move(task), success, observation, timers, queue_begin](bool queued) noexcept {
        timers.finishQueue(success);
        block_tree_cache_detail::ScopeRollback finish([&] { timers.finish(success); });
        CompletionStageScope                   execute(observation, "completion_execute", "business_completion");
        if (queued) {
            metrics_reporter_->reportQueueWaitMetric(
                true, "business", "evict", Tier::NONE, Tier::NONE, currentTimeUs() - queue_begin);
        }
        bool              any_not_detached = false;
        std::vector<bool> detached;
        detached.reserve(task->descriptors().size());
        {
            std::unique_lock<std::mutex> lock(*mutex_, std::defer_lock);
            {
                CompletionStageScope wait(
                    observation, "completion_lock_wait", "business_completion", AsyncProfileKind::Wait);
                lock.lock();
            }
            CompletionStageScope tree(observation, "completion_settle", "business_completion");
            for (const TransferDescriptor& desc : task->descriptors()) {
                heaps_[desc.group_set_id].pending_timings.erase(desc.node);
                const bool is_detached = desc.node->group_set_resources[desc.group_set_id].transfer_detached;
                detached.push_back(is_detached);
                any_not_detached = any_not_detached || !is_detached;
            }

            if (success) {
                {
                    CompletionStageScope update(observation, "completion_tree_update", "business_completion");
                    completeEvict(task->descriptors());
                }
                {
                    CompletionStageScope cleanup(observation, "completion_cleanup", "business_completion");
                    settleEviction(task->descriptors());
                }
            } else {
                CompletionStageScope rollback(observation, "completion_rollback", "business_completion");
                rollbackTransferLocked(task->descriptors());
            }
            updatePendingRelease(task->descriptors(), false);
            const auto& descriptor = task->descriptors().front();

            {
                CompletionStageScope watermark(observation, "completion_watermark", "business_completion");
                const bool is_host_disk = descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK;
                auto&      round =
                    heaps_[descriptor.group_set_id].watermark_rounds[static_cast<size_t>(descriptor.source_tier)];
                // Keep the round active while settled_ rechecks watermarks.
                settled_(success && any_not_detached);
                if (is_host_disk) {
                    if (!success) {
                        round.watermark.reset();
                    }
                    scheduleWatermarkEvictionsLocked(Tier::HOST);
                }
                if (!is_host_disk || !round.watermark.has_value()) {
                    finishWatermarkRoundLocked(descriptor.group_set_id, descriptor.source_tier);
                }
            }
        }
        if (any_not_detached) {
            CompletionStageScope metrics(observation, "completion_metrics", "business_completion");
            metrics_reporter_->reportEvictionFinished(*task, tree_->groupSets(), success, detached);
        }
    };

    bool submitted = false;
    try {
        submitted = task_pool_->submitCompletion([settle] { settle(true); });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement with unknown exception");
    }
    if (!submitted) {
        RTP_LLM_LOG_WARNING("eviction completion queue is closed; settling inline");
        settle(false);
    }
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(size_t                 group_set_id,
                                                                 Tier                   tier,
                                                                 bool                   force_drop,
                                                                 EvictionBatchContext*  eviction_batch_context,
                                                                 VictimSelectionTrigger trigger) {
    EvictionHeap* heap = heapFor(group_set_id, tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    // Instrumentation adds no allocation, atomics, extra locks, or reporting in the selection loop.
    auto* counters =
        metrics_reporter_->enabled() ?
            &victim_selection_metrics_.at(trigger, tier, tree_->groupSets()[group_set_id]->groupType(), force_drop) :
            nullptr;
    const auto begin    = counters ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool       selected = false;
    block_tree_cache_detail::ScopeRollback record([&] {
        if (counters != nullptr) {
            counters->observed = true;
            ++counters->calls;
            counters->selected += selected;
            counters->latency_sum_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        }
    });
    EvictionBatchContext                   local_eviction_batch_context;
    if (eviction_batch_context == nullptr) {
        eviction_batch_context = &local_eviction_batch_context;
    }
    struct SelectionContext {
        VictimSelectionCounters* counters;
        EvictionBatchContext*    eviction_batch_context;

        bool skip(VictimSkipReason reason) const {
            if (counters != nullptr) {
                ++counters->skipped[static_cast<size_t>(reason)];
            }
            return false;
        }
    } context{counters, eviction_batch_context};
    const auto can_evict_uncached = [this, group_set_id, tier, force_drop, &context](TreeNode* node) {
        if (force_drop && tier == Tier::DEVICE && isEvictable(node, group_set_id, tier)) {
            const TransferDescriptor desc(node,
                                          group_set_id,
                                          /*path_index=*/0,
                                          tier,
                                          Tier::NONE,
                                          node->group_set_resources[group_set_id].getBlocks(tier));
            if (!canReclaimDeviceDrop(desc, context.eviction_batch_context)) {
                return context.skip(VictimSkipReason::NOT_RECLAIMABLE);
            }
        }
        if (!force_drop) {
            const GroupSetPtr&      group_set = tree_->groupSets()[group_set_id];
            const GroupSetResource& resource  = node->group_set_resources[group_set_id];
            if (tier == Tier::DEVICE) {
                for (size_t i = 0; i < resource.device_blocks.size(); ++i) {
                    if (!isNullBlockIdx(resource.device_blocks[i])) {
                        if (group_set->devicePools()[i]->hasRequestRef(resource.device_blocks[i])) {
                            return context.skip(VictimSkipReason::REQUEST_REF);
                        }
                        if (group_set->devicePools()[i]->isFrozen(resource.device_blocks[i])) {
                            return context.skip(VictimSkipReason::FROZEN);
                        }
                    }
                }
            } else if ((tier == Tier::HOST && !isNullBlockIdx(resource.host_block)
                        && group_set->hostPool()->isFrozen(resource.host_block))
                       || (tier == Tier::DISK && !isNullBlockIdx(resource.disk_block)
                           && group_set->diskPool()->isFrozen(resource.disk_block))) {
                return context.skip(VictimSkipReason::FROZEN);
            }
        }
        if (force_drop || tier != Tier::DEVICE
            || tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL) {
            return true;
        }
        const bool children_idle =
            std::none_of(node->children.begin(), node->children.end(), [group_set_id](const auto& child_entry) {
                const GroupSetTransferState state =
                    child_entry.second->group_set_resources[group_set_id].transfer_state;
                return state == GroupSetTransferState::LOAD_PENDING || state == GroupSetTransferState::LOADING;
            });
        return children_idle || context.skip(VictimSkipReason::CHILD_LOADING);
    };
    // Keep the std::function passed to best() at its original two-pointer capture size.
    const auto can_evict = [&context, &can_evict_uncached](TreeNode* node) {
        if (context.counters != nullptr) {
            ++context.counters->scanned_candidates;
        }
        if (context.eviction_batch_context->rejected_candidates.count(node) != 0) {
            return context.skip(VictimSkipReason::CACHED_REJECTION);
        }
        if (!can_evict_uncached(node)) {
            // Request releases may happen concurrently. Retry those candidates in
            // the next pass instead of repeatedly locking their pools in this one.
            context.eviction_batch_context->rejected_candidates.insert(node);
            return false;
        }
        return true;
    };
    std::optional<EvictionEntry> entry;
    while ((entry = heap->best(can_evict)).has_value()) {
        if (isEvictable(entry->node, group_set_id, tier)) {
            break;
        }
        context.skip(VictimSkipReason::INVALID_CANDIDATE);
        const GroupSetResource& resource = entry->node->group_set_resources[group_set_id];
        RTP_LLM_LOG_ERROR("invalid eviction candidate: node=%p key=%ld group_set_id=%zu source_tier=%s "
                          "top_tier=%s transfer_state=%d",
                          static_cast<void*>(entry->node),
                          entry->node->cache_key,
                          group_set_id,
                          tierName(tier),
                          tierName(resource.getTopTier()),
                          static_cast<int>(resource.transfer_state));
        heap->erase(entry->node);
    }
    if (!entry.has_value()) {
        return std::nullopt;
    }

    GroupSetResource& resource    = entry->node->group_set_resources[group_set_id];
    Tier              target_tier = Tier::NONE;
    if (!force_drop) {
        if (tier == Tier::DEVICE && is_tier_enabled_(Tier::HOST)) {
            target_tier = Tier::HOST;
        } else if (tier == Tier::HOST && is_tier_enabled_(Tier::DISK)) {
            target_tier = Tier::DISK;
        }
    }
    TransferDescriptor victim(entry->node,
                              group_set_id,
                              /*path_index=*/0,
                              tier,
                              target_tier,
                              resource.getBlocks(tier));
    selected = true;
    return victim;
}

Tier BlockTreeEvictor::watermarkTargetTier(Tier source_tier) const {
    if (source_tier == Tier::DEVICE && is_tier_enabled_(Tier::HOST)) {
        return Tier::HOST;
    }
    if (source_tier == Tier::HOST && is_tier_enabled_(Tier::DISK)) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

size_t
BlockTreeEvictor::watermarkLogicalBatchLimit(const GroupSet& group_set, Tier source_tier, Tier target_tier) const {
    if (target_tier == Tier::NONE) {
        return std::numeric_limits<size_t>::max();
    }
    if (source_tier != Tier::HOST || target_tier != Tier::DISK || !group_set.diskPool()) {
        return max_descriptors_per_batch_;
    }
    size_t available_bytes = disk_cache_evict_max_pending_bytes_;
    bool   has_pending     = false;
    for (const auto& pending_group : tree_->groupSets()) {
        const size_t count =
            heaps_[pending_group->groupSetId()].watermark_rounds[static_cast<size_t>(Tier::HOST)].pending_count;
        if (count == 0) {
            continue;
        }
        has_pending        = true;
        const size_t bytes = pending_group->diskPool()->strideBytes();
        if (count > available_bytes / bytes) {
            return 0;
        }
        available_bytes -= count * bytes;
    }
    size_t count = available_bytes / group_set.diskPool()->strideBytes();
    if (count == 0 && !has_pending) {
        count = 1;
    }
    return std::min(count, max_descriptors_per_batch_);
}

void BlockTreeEvictor::scheduleWatermarkEvictionsLocked(Tier                         tier,
                                                        std::optional<TierWatermark> tier_watermark,
                                                        std::optional<TierWatermark> full_watermark) {
    const bool   is_host_disk       = tier == Tier::HOST && watermarkTargetTier(tier) == Tier::DISK;
    const auto&  group_sets         = tree_->groupSets();
    const size_t first_group        = is_host_disk ? next_disk_evict_group_ : 0;
    bool         waiting_for_budget = false;
    for (size_t i = 0; i < group_sets.size(); ++i) {
        const size_t group_index = (first_group + i) % group_sets.size();
        const auto&  group_set   = group_sets[group_index];
        auto&        round       = heaps_[group_set->groupSetId()].watermark_rounds[static_cast<size_t>(tier)];
        if (tier_watermark.has_value()
            && (!is_host_disk
                || (!round.watermark.has_value() && round.required_count == 0 && round.pending_count == 0))) {
            round.watermark = group_set->groupType() == CacheGroupType::FULL && full_watermark.has_value() ?
                                  full_watermark :
                                  tier_watermark;
        }
        if (!round.watermark.has_value()) {
            continue;
        }
        const auto& watermark      = *round.watermark;
        size_t      required_count = computeWatermarkEvictCount(*group_set, tier, watermark);
        if (round.required_count == 0) {
            round.required_count = required_count;
        }
        bool eviction_reported = is_host_disk && round.scheduled_count != 0;
        while (required_count > 0 && !waiting_for_budget) {
            const Tier   target_tier = watermarkTargetTier(tier);
            const size_t batch_count =
                std::min(required_count, watermarkLogicalBatchLimit(*group_set, tier, target_tier));
            if (batch_count == 0) {
                waiting_for_budget = true;
                // Let pending writes drain so smaller groups cannot starve this block.
                next_disk_evict_group_ = group_index;
                break;
            }
            const BatchEvictResult result = batchEvictLocked(group_set->groupSetId(), tier, batch_count);
            round.scheduled_count += result.scheduled_count;
            if (result.madeProgress() && !eviction_reported) {
                metrics_reporter_->reportEvictionTriggered(tier, group_set->groupType(), /*force_drop=*/false);
                eviction_reported = true;
            }
            if (!result.madeProgress()) {
                if (!is_host_disk || round.pending_count == 0) {
                    round.watermark.reset();
                }
                break;
            }
            if (is_host_disk && result.async_submitted) {
                next_disk_evict_group_ = (group_index + 1) % group_sets.size();
            }
            if (result.async_submitted) {
                break;
            }

            const size_t remaining_count = computeWatermarkEvictCount(*group_set, tier, watermark);
            if (remaining_count >= required_count) {
                break;
            }
            required_count = remaining_count;
        }
        if (!is_host_disk || !round.watermark.has_value() || required_count == 0) {
            finishWatermarkRoundLocked(group_set->groupSetId(), tier);
        }
    }
}

void BlockTreeEvictor::finishWatermarkRoundLocked(size_t group_set_id, Tier source_tier) {
    auto& round = heaps_[group_set_id].watermark_rounds[static_cast<size_t>(source_tier)];
    if (round.pending_count != 0) {
        return;
    }
    if (round.required_count != 0) {
        metrics_reporter_->reportEvictionBlocks(source_tier,
                                                tree_->groupSets()[group_set_id]->groupType(),
                                                /*force_drop=*/false,
                                                round.required_count,
                                                round.scheduled_count);
    }
    round = {};
}

void BlockTreeEvictor::updatePendingRelease(const std::vector<TransferDescriptor>& descs, bool reserve) {
    std::unordered_map<IBlockPool*, size_t> deltas;
    deltas.reserve(descs.size());
    const auto add_delta = [&deltas](IBlockPool* pool, size_t delta) {
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "pending eviction release references a null pool");
        deltas[pool] += delta;
    };
    for (const TransferDescriptor& desc : descs) {
        const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
        if (desc.source_tier == Tier::DEVICE) {
            const auto& pools = group_set->devicePools();
            RTP_LLM_CHECK_WITH_INFO(desc.source_blocks.size() == pools.size(),
                                    "pending eviction release device block count mismatch: group_set=%zu blocks=%zu "
                                    "pools=%zu",
                                    desc.group_set_id,
                                    desc.source_blocks.size(),
                                    pools.size());
            for (const DeviceBlockPoolPtr& pool : pools) {
                add_delta(pool.get(), 1);
            }
        } else if (desc.source_tier == Tier::HOST) {
            add_delta(group_set->hostPool().get(), desc.source_blocks.size());
        }
    }

    std::lock_guard<std::mutex> lock(pending_release_mutex_);
    if (reserve) {
        for (const auto& [pool, delta] : deltas) {
            pending_release_counts_[pool] += delta;
        }
        for (const TransferDescriptor& desc : descs) {
            ++heaps_[desc.group_set_id].watermark_rounds[static_cast<size_t>(desc.source_tier)].pending_count;
        }
        return;
    }
    for (const auto& [pool, delta] : deltas) {
        const auto it = pending_release_counts_.find(pool);
        RTP_LLM_CHECK_WITH_INFO(it != pending_release_counts_.end() && it->second >= delta,
                                "invalid pending eviction release settlement: "
                                "pool=%s address=%p required=%zu pending=%zu",
                                pool->poolName().c_str(),
                                static_cast<void*>(pool),
                                delta,
                                it == pending_release_counts_.end() ? 0 : it->second);
    }
    for (const auto& [pool, delta] : deltas) {
        auto it = pending_release_counts_.find(pool);
        it->second -= delta;
    }
    for (const TransferDescriptor& desc : descs) {
        --heaps_[desc.group_set_id].watermark_rounds[static_cast<size_t>(desc.source_tier)].pending_count;
    }
}

// ---- Migration pipeline (begin -> copy -> finish) ----
void BlockTreeEvictor::collectFullPrune(const TransferDescriptor&                  eviction_desc,
                                        EvictionDropTask&                          task,
                                        std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const {
    if (task.hasFullPrune() || tree_->groupSets()[eviction_desc.group_set_id]->groupType() != CacheGroupType::FULL) {
        return;
    }

    // Reuse the output vector as the worklist. Reversing a parent-before-child
    // traversal gives the bottom-up order required by topology cleanup.
    task.full_prune_nodes_bottom_up.push_back(eviction_desc.node);
    for (size_t i = 0; i < task.full_prune_nodes_bottom_up.size(); ++i) {
        TreeNode* node = task.full_prune_nodes_bottom_up[i];
        for (const auto& [_, child] : node->children) {
            task.full_prune_nodes_bottom_up.push_back(child);
        }
        if (node == eviction_desc.node) {
            continue;
        }

        // Once a FULL prefix is removed, every descendant resource is unreachable
        // through matching. Prune every idle resource and detach in-flight transfers.
        for (const GroupSetPtr& group_set : tree_->groupSets()) {
            const size_t            group_set_id = group_set->groupSetId();
            const GroupSetResource& resource     = node->group_set_resources[group_set_id];
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                detached_resources.emplace_back(node, group_set_id);
                continue;
            }
            const Tier source_tier = resource.getTopTier();
            if (source_tier == Tier::NONE) {
                continue;
            }
            task.dependent_prune_descs.emplace_back(node,
                                                    group_set_id,
                                                    /*path_index=*/0,
                                                    source_tier,
                                                    Tier::NONE,
                                                    resource.getBlocks(source_tier));
        }
    }
    std::reverse(task.full_prune_nodes_bottom_up.begin(), task.full_prune_nodes_bottom_up.end());
}

void BlockTreeEvictor::runDropTask(TransferDescriptor    eviction_desc,
                                   bool                  notify_settled,
                                   const char*           trigger_type,
                                   EvictionBatchContext* eviction_batch_context) {
    const EvictionDropTask task = createDropTask(std::move(eviction_desc), trigger_type, eviction_batch_context);
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        completeDrop(dependent_desc);
    }
    completeDrop(task.primary_desc);
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        completeDrop(cascade_desc);
    }
    if (task.hasFullPrune()) {
        TreeNode* const boundary_node = task.primary_desc.node->parent;
        for (TreeNode* node : task.full_prune_nodes_bottom_up) {
            eraseNodeFromAllHeaps(node);
            tree_->removeNode(node);
        }
        TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(boundary_node);
        updateFullCandidate(survivor);
    } else {
        settleSingleEviction(task.primary_desc.node);
    }
    if (notify_settled) {
        settled_(false);
    }
    metrics_reporter_->reportEvictionFinished(task, tree_->groupSets(), trigger_type);
}

void BlockTreeEvictor::completeDrop(const TransferDescriptor& desc) {
    const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
    auto&              resource  = desc.node->group_set_resources[desc.group_set_id];
    RTP_LLM_CHECK_WITH_INFO(resource.transfer_state == GroupSetTransferState::DEMOTING,
                            "drop completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s",
                            desc.group_set_id,
                            desc.node->cache_key,
                            static_cast<int>(resource.transfer_state),
                            tierName(desc.source_tier));

    const MultiNodeResource source_holder{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    resource.evictFromTier(desc.source_tier);
    resource.transfer_state = GroupSetTransferState::IDLE;
    tree_->refreshPublishedState(desc.node);
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "drop settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            desc.group_set_id,
                            desc.node->cache_key);
}

void BlockTreeEvictor::completeEvict(const std::vector<TransferDescriptor>& descs) {
    const TransferDescriptor&       first     = descs.front();
    const GroupSetPtr&              group_set = tree_->groupSets()[first.group_set_id];
    MultiNodeResource               source_holder{first.group_set_id, first.source_tier, {}};
    MultiNodeResource               target_holder{first.group_set_id, first.target_tier, {}};
    MultiNodeResource               active_target_holder{first.group_set_id, first.target_tier, {}};
    std::vector<TransferDescriptor> detached_descs;
    source_holder.node_blocks.reserve(descs.size());
    target_holder.node_blocks.reserve(descs.size());
    active_target_holder.node_blocks.reserve(descs.size());
    detached_descs.reserve(descs.size());

    for (const TransferDescriptor& desc : descs) {
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        if (resource.transfer_detached) {
            detached_descs.push_back(desc);
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(
            resource.transfer_state == GroupSetTransferState::DEMOTING,
            "eviction completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s target=%s",
            desc.group_set_id,
            desc.node->cache_key,
            static_cast<int>(resource.transfer_state),
            tierName(desc.source_tier),
            tierName(desc.target_tier));
        active_target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        source_holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
    }

    if (!active_target_holder.node_blocks.empty()) {
        group_set->referenceBlocks(active_target_holder, BlockTreeRefType::CACHE);
    }
    group_set->unreferenceBlocks(target_holder, BlockTreeRefType::EVICTION);
    if (!source_holder.node_blocks.empty()) {
        group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    }

    const int64_t tier_enter_time_us = currentTimeUs();
    for (const TransferDescriptor& desc : descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (resource.transfer_detached) {
            continue;
        }
        resource.evictFromTier(desc.source_tier);
        resource.transfer_state       = GroupSetTransferState::IDLE;
        resource.setBlocks(desc.target_tier, desc.target_blocks);
        tree_->refreshPublishedState(desc.node);
        RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                                desc.group_set_id,
                                desc.node->cache_key);

        resource.candidate_meta.admission_seq         = ++admission_seq_;
        resource.candidate_meta.tier_enter_time_us    = tier_enter_time_us;
        resource.candidate_meta.last_access_time_us   = tier_enter_time_us;
        resource.candidate_meta.tier_last_hit_time_us = 0;
        admitCandidate(desc.node, desc.group_set_id, desc.target_tier);
    }
    discardDetachedTransfer(detached_descs);
}

void BlockTreeEvictor::rollbackTransferLocked(const std::vector<TransferDescriptor>& descs) {
    releaseTargetBlocks(descs);
    const std::vector<TransferDescriptor> detached_descs = restoreSource(descs);
    settleEviction(detached_descs);
}

// Reserve the source: remove it from its source heap and mark the in-flight state so
// no other selector can pick it. The caller must first verify that the resource
// is IDLE; assigning DEMOTING is not safe for an already reserved resource.
void BlockTreeEvictor::reserveSource(const std::vector<TransferDescriptor>& eviction_descs) {
    for (const TransferDescriptor& desc : eviction_descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
        resource.transfer_state       = GroupSetTransferState::DEMOTING;
        tree_->refreshPublishedState(desc.node);
    }
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
std::vector<TransferDescriptor> BlockTreeEvictor::restoreSource(const std::vector<TransferDescriptor>& eviction_descs) {
    std::vector<TransferDescriptor> detached_descs;
    detached_descs.reserve(eviction_descs.size());
    for (const TransferDescriptor& desc : eviction_descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (resource.transfer_detached) {
            detached_descs.push_back(desc);
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
            RTP_LLM_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", desc.group_set_id, desc.node->cache_key);
            continue;
        }
        resource.transfer_state       = GroupSetTransferState::IDLE;
        tree_->refreshPublishedState(desc.node);
        RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                                desc.group_set_id,
                                desc.node->cache_key);

        admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
    }
    discardDetachedTransfer(detached_descs);
    return detached_descs;
}

void BlockTreeEvictor::discardDetachedTransfer(const std::vector<TransferDescriptor>& transfer_descs) {
    if (transfer_descs.empty()) {
        return;
    }

    for (const TransferDescriptor& desc : transfer_descs) {
        GroupSetResource&  resource  = desc.node->group_set_resources[desc.group_set_id];
        const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
        const MultiNodeResource source_holder{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
        resource.evictFromTier(desc.source_tier);
        resource.transfer_state       = GroupSetTransferState::IDLE;
        resource.transfer_detached    = false;
        tree_->refreshPublishedState(desc.node);
    }
}

void BlockTreeEvictor::invalidateSources(const std::vector<InvalidationSource>& sources) {
    for (const auto& source : sources) {
        const auto path = tree_->findNode(source.cache_keys);
        // findNode returns the surviving prefix when an earlier drop removed this source.
        if (path.size() != source.cache_keys.size()) {
            continue;
        }
        TreeNode*               node     = path.back();
        const GroupSetResource& resource = node->group_set_resources[source.group_set_id];
        if (!resource.hasTier(source.source_tier)) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(resource.transfer_state == GroupSetTransferState::IDLE,
                                "invalidated source must be settled: group_set=%zu node_key=%ld",
                                source.group_set_id,
                                node->cache_key);
        runDropTask(TransferDescriptor(node,
                                       source.group_set_id,
                                       /*path_index=*/0,
                                       source.source_tier,
                                       Tier::NONE,
                                       resource.getBlocks(source.source_tier)),
                    /*notify_settled=*/false);
    }
}

void BlockTreeEvictor::releaseTargetBlocks(const std::vector<TransferDescriptor>& descs) {
    if (descs.empty()) {
        return;
    }

    BlockIdList target_blocks;
    target_blocks.reserve(descs.size());
    for (const TransferDescriptor& desc : descs) {
        target_blocks.insert(target_blocks.end(), desc.target_blocks.begin(), desc.target_blocks.end());
    }

    if (!target_blocks.empty()) {
        const TransferDescriptor& first = descs.front();
        tree_->groupSets()[first.group_set_id]->releaseBlocks(
            first.target_tier, target_blocks, BlockTreeRefType::EVICTION);
    }
}

void BlockTreeEvictor::settleEviction(const std::vector<TransferDescriptor>& descs) {
    std::unordered_set<TreeNode*> pending_nodes;
    std::unordered_set<TreeNode*> refresh_nodes;
    pending_nodes.reserve(descs.size());
    refresh_nodes.reserve(descs.size());
    for (const TransferDescriptor& desc : descs) {
        if (desc.node->parent == nullptr) {
            pending_nodes.insert(desc.node);
        } else {
            refresh_nodes.insert(desc.node->parent);
        }
    }

    tree_->removePendingNodes(pending_nodes);
    for (TreeNode* node : refresh_nodes) {
        updateFullCandidate(node);
    }
}

void BlockTreeEvictor::settleSingleEviction(TreeNode* node) {
    if (tree_->isRemovable(node)) {
        RTP_LLM_LOG_DEBUG("deleting empty node key=%ld", node->cache_key);
        eraseNodeFromAllHeaps(node);
        TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(node);
        updateFullCandidate(survivor);
    } else {
        updateFullCandidate(node->parent);
    }
}

void BlockTreeEvictor::eraseNodeFromAllHeaps(TreeNode* node) {
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            heapFor(group_set->groupSetId(), tier)->erase(node);
        }
    }
}

bool BlockTreeEvictor::canReclaimDeviceDrop(const TransferDescriptor& desc,
                                            EvictionBatchContext*     eviction_batch_context) const {
    const auto can_reclaim = [this](const TransferDescriptor& candidate) {
        if (candidate.source_tier != Tier::DEVICE) {
            return true;
        }
        const auto& pools = tree_->groupSets()[candidate.group_set_id]->devicePools();
        for (size_t i = 0; i < candidate.source_blocks.size(); ++i) {
            if (!isNullBlockIdx(candidate.source_blocks[i])
                && !pools[i]->canReclaimCacheBlock(candidate.source_blocks[i])) {
                return false;
            }
        }
        return true;
    };
    if (!can_reclaim(desc)) {
        return false;
    }
    if (tree_->groupSets()[desc.group_set_id]->groupType() == CacheGroupType::FULL) {
        // FULL usually shares request ownership with its related resources. Rarely,
        // an upward cascade can reach a request-held ancestor whose SWA was dropped.
        // Check only the primary resource to avoid planning the closure twice; any
        // held related block stays physically alive through its request reference.
        return true;
    }
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    const EvictionDropTask                    task = planDropTask(desc, detached_resources, eviction_batch_context);
    // FULL pruning and cascades must not discard cache entries backed by held
    // device blocks either. Planning does not reserve or detach any resource.
    return std::all_of(task.cascade_descs.begin(), task.cascade_descs.end(), can_reclaim)
           && std::all_of(task.dependent_prune_descs.begin(), task.dependent_prune_descs.end(), can_reclaim)
           && std::none_of(detached_resources.begin(), detached_resources.end(), [](const auto& entry) {
                  return entry.first->group_set_resources[entry.second].hasTier(Tier::DEVICE);
              });
}

EvictionDropTask BlockTreeEvictor::planDropTask(TransferDescriptor                         eviction_desc,
                                                std::vector<std::pair<TreeNode*, size_t>>& detached_resources,
                                                EvictionBatchContext* eviction_batch_context) const {
    EvictionDropTask task;
    task.primary_timing =
        EvictionTimingSnapshot(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    task.primary_desc = std::move(eviction_desc);
    const TransferDescriptor&                 primary_desc = task.primary_desc;
    const Tier                                target_tier  = primary_desc.target_tier;
    collectFullPrune(primary_desc, task, detached_resources);
    TreeNode* const                           primary_node     = primary_desc.node;
    const size_t                              primary_group_id = primary_desc.group_set_id;
    const Tier                                source_tier      = primary_desc.source_tier;
    const auto&                               group_sets       = tree_->groupSets();
    std::vector<std::pair<TreeNode*, size_t>> busy_root_cascades;

    const auto append_desc = [&](TreeNode* node, size_t group_set_id) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(
            node, group_set_id, /*path_index=*/0, source_tier, target_tier, resource.getBlocks(source_tier));
    };

    const bool primary_is_leaf        = tree_->isLeafAtTier(primary_node, primary_group_id, source_tier);
    const CacheGroupType primary_type = group_sets[primary_group_id]->groupType();
    for (const GroupSetPtr& group_set : group_sets) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set_id == primary_group_id) {
            continue;
        }

        bool should_cascade = primary_is_leaf;
        if (!should_cascade) {
            switch (primary_type) {
                case CacheGroupType::FULL:
                    should_cascade = group_set->groupType() == CacheGroupType::SWA
                                     || group_set->groupType() == CacheGroupType::LINEAR;
                    break;
                case CacheGroupType::SWA:
                    should_cascade = group_set->groupType() == CacheGroupType::LINEAR;
                    break;
                case CacheGroupType::LINEAR:
                    break;
            }
        }
        if (!should_cascade) {
            continue;
        }
        const GroupSetResource& resource = primary_node->group_set_resources[group_set_id];
        if (resource.transfer_state != GroupSetTransferState::IDLE) {
            busy_root_cascades.emplace_back(primary_node, group_set_id);
            continue;
        }
        if (!isEvictable(primary_node, group_set_id, source_tier)) {
            continue;
        }

        append_desc(primary_node, group_set_id);
        collectFullPrune(task.cascade_descs.back(), task, detached_resources);
    }

    if (task.hasFullPrune()) {
        detached_resources.insert(detached_resources.end(), busy_root_cascades.begin(), busy_root_cascades.end());
        // Keep the later closure-root cleanup, using one serving tier per resource.
        // Primary and cascade descriptors already own their selected root resources.
        TreeNode* const closure_root = task.full_prune_nodes_bottom_up.back();
        for (const GroupSetPtr& group_set : group_sets) {
            const size_t group_set_id = group_set->groupSetId();
            if ((task.primary_desc.node == closure_root && task.primary_desc.group_set_id == group_set_id)
                || std::any_of(task.cascade_descs.begin(), task.cascade_descs.end(), [&](const auto& desc) {
                       return desc.node == closure_root && desc.group_set_id == group_set_id;
                   })) {
                continue;
            }
            const GroupSetResource& resource = closure_root->group_set_resources[group_set_id];
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                const auto entry = std::make_pair(closure_root, group_set_id);
                if (std::find(detached_resources.begin(), detached_resources.end(), entry)
                    == detached_resources.end()) {
                    detached_resources.push_back(entry);
                }
            } else if (!resource.is_empty()) {
                const Tier tier = resource.getTopTier();
                task.dependent_prune_descs.emplace_back(
                    closure_root, group_set_id, /*path_index=*/0, tier, Tier::NONE, resource.getBlocks(tier));
            }
        }
    }

    if (detached_resources.empty()) {
        selectUpwardCascades(task, eviction_batch_context);
    }

    return task;
}

EvictionDropTask BlockTreeEvictor::createDropTask(TransferDescriptor    eviction_desc,
                                                  const char*           trigger_type,
                                                  EvictionBatchContext* eviction_batch_context) {
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    EvictionDropTask task = planDropTask(std::move(eviction_desc), detached_resources, eviction_batch_context);
    const auto&                             group_sets = tree_->groupSets();

    task.dependent_prune_timings.resize(task.dependent_prune_descs.size());
    for (size_t i = 0; i < task.dependent_prune_descs.size(); ++i) {
        const TransferDescriptor& desc = task.dependent_prune_descs[i];
        task.dependent_prune_timings[i] =
            EvictionTimingSnapshot(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }
    task.cascade_timings.reserve(task.cascade_descs.size());
    for (const TransferDescriptor& desc : task.cascade_descs) {
        task.cascade_timings.emplace_back(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }

    if (eviction_batch_context != nullptr) {
        auto& endpoint_matchable = eviction_batch_context->endpoint_matchable;
        // Removing a subtree preserves every surviving prefix. Interior resource
        // changes can invalidate descendant windows, so discard the whole snapshot.
        if (task.hasFullPrune() || task.primary_desc.node->children.empty()) {
            endpoint_matchable.erase(task.primary_desc.node);
            for (TreeNode* node : task.full_prune_nodes_bottom_up) {
                endpoint_matchable.erase(node);
            }
            for (const auto& desc : task.cascade_descs) {
                endpoint_matchable.erase(desc.node);
            }
        } else if (!endpoint_matchable.empty()) {
            endpoint_matchable.clear();
        }
    }

    reserveSource({task.primary_desc});
    reserveSource(task.cascade_descs);
    reserveSource(task.dependent_prune_descs);
    for (const auto& [node, group_set_id] : detached_resources) {
        GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_detached) {
            continue;
        }
        const Tier               tier = resource.getTopTier();
        const TransferDescriptor desc{node, group_set_id, 0, tier, Tier::NONE, resource.getBlocks(tier)};
        const auto               timing = resource.transfer_state == GroupSetTransferState::DEMOTING ?
                                              heaps_[group_set_id].pending_timings.at(node) :
                                              EvictionTimingSnapshot(resource.candidate_meta);
        metrics_reporter_->reportEvictionDescriptor(desc, timing, group_sets, trigger_type, true);
        resource.transfer_detached = true;
        tree_->refreshPublishedState(node);
    }
    return task;
}

void BlockTreeEvictor::selectUpwardCascades(EvictionDropTask&     task,
                                            EvictionBatchContext* eviction_batch_context) const {
    const TransferDescriptor& primary_desc = task.primary_desc;
    TreeNode* const           primary_node = primary_desc.node;
    const Tier                target_tier  = primary_desc.target_tier;
    const auto&               group_sets   = tree_->groupSets();

    std::vector<bool> selected(group_sets.size(), false);
    selected[primary_desc.group_set_id] = true;
    for (const TransferDescriptor& desc : task.cascade_descs) {
        if (desc.node == primary_node) {
            selected[desc.group_set_id] = true;
        }
    }
    for (const TransferDescriptor& desc : task.dependent_prune_descs) {
        if (desc.node == primary_node) {
            selected[desc.group_set_id] = true;
        }
    }
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        const GroupSetResource& resource = primary_node->group_set_resources[group_set_id];
        if (!resource.is_empty() && (resource.getTopTier() != primary_desc.source_tier || !selected[group_set_id])) {
            return;
        }
    }
    if (!task.hasFullPrune() && !primary_node->children.empty()) {
        return;
    }

    const auto append_desc = [&](TreeNode* node, size_t group_set_id, Tier source_tier) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(
            node, group_set_id, /*path_index=*/0, source_tier, target_tier, resource.getBlocks(source_tier));
    };

    std::vector<TreeNode*> ancestors;
    std::vector<bool> endpoint_matchable;
    if (eviction_batch_context == nullptr
        || eviction_batch_context->endpoint_matchable.count(primary_node->parent) == 0) {
        for (TreeNode* node = primary_node->parent; node != tree_->root(); node = node->parent) {
            ancestors.push_back(node);
        }
        std::reverse(ancestors.begin(), ancestors.end());

        std::vector<std::unique_ptr<MatchValidator>> match_validators;
        match_validators.reserve(group_sets.size());
        for (const GroupSetPtr& group_set : group_sets) {
            match_validators.push_back(group_set->createMatchValidator());
        }
        endpoint_matchable.reserve(ancestors.size());
        for (TreeNode* node : ancestors) {
            bool all_groups_valid = true;
            for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
                if (!match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                    all_groups_valid = false;
                }
            }
            endpoint_matchable.push_back(all_groups_valid);
            if (eviction_batch_context != nullptr) {
                eviction_batch_context->endpoint_matchable[node] = all_groups_valid;
            }
        }
    }

    size_t path_index = ancestors.size();
    for (TreeNode* parent = primary_node->parent; parent != tree_->root(); parent = parent->parent) {
        const bool matchable = eviction_batch_context ? eviction_batch_context->endpoint_matchable.at(parent) :
                                                        endpoint_matchable[--path_index];
        if (parent->is_resident || matchable) {
            break;
        }

        if (parent->children.size() != 1) {
            break;
        }

        const size_t desc_begin = task.cascade_descs.size();
        bool         releasable = true;
        for (const GroupSetPtr& group_set : group_sets) {
            const size_t            group_set_id = group_set->groupSetId();
            const GroupSetResource& resource     = parent->group_set_resources[group_set_id];
            if (resource.is_empty()) {
                continue;
            }
            const Tier source_tier = resource.getTopTier();
            if (resource.transfer_state != GroupSetTransferState::IDLE) {
                releasable = false;
                break;
            }
            append_desc(parent, group_set_id, source_tier);
        }
        if (!releasable) {
            task.cascade_descs.resize(desc_begin);
            break;
        }
    }
}

size_t
BlockTreeEvictor::computeWatermarkEvictCount(const GroupSet& group_set, Tier tier, const TierWatermark& watermark) {
    if (tier != Tier::DEVICE && tier != Tier::HOST && tier != Tier::DISK) {
        return 0;
    }

    std::vector<IBlockPool*> pools;
    if (tier == Tier::DEVICE) {
        for (const DeviceBlockPoolPtr& pool : group_set.devicePools()) {
            pools.push_back(pool.get());
        }
    } else if (tier == Tier::HOST && group_set.hostPool() != nullptr) {
        pools.push_back(group_set.hostPool().get());
    } else if (tier == Tier::DISK && group_set.diskPool() != nullptr) {
        pools.push_back(group_set.diskPool().get());
    }

    bool   has_pending    = false;
    bool   high_reached   = false;
    size_t required_count = 0;
    for (IBlockPool* pool : pools) {
        size_t pending_count = 0;
        {
            std::lock_guard<std::mutex> lock(pending_release_mutex_);
            const auto                  pending = pending_release_counts_.find(pool);
            if (pending != pending_release_counts_.end()) {
                pending_count = pending->second;
            }
        }
        const size_t used  = pool->usedBlocksNum();
        const size_t total = pool->totalBlocksNum();
        if (pending_count > 0) {
            RTP_LLM_CHECK_WITH_INFO(pending_count <= used,
                                    "pending eviction releases exceed used blocks: "
                                    "pool=%s address=%p pending=%zu used=%zu total=%zu low_ratio=%f high_ratio=%f",
                                    pool->poolName().c_str(),
                                    static_cast<void*>(pool),
                                    pending_count,
                                    used,
                                    total,
                                    watermark.low_ratio,
                                    watermark.high_ratio);
        }
        const size_t effective_used = used - pending_count;
        const size_t high_used_blocks =
            static_cast<size_t>(std::ceil(static_cast<double>(total) * watermark.high_ratio));
        const size_t low_used_blocks =
            static_cast<size_t>(std::floor(static_cast<double>(total) * watermark.low_ratio));
        has_pending  = has_pending || pending_count != 0;
        high_reached = high_reached || effective_used >= high_used_blocks;
        required_count =
            std::max(required_count, effective_used > low_used_blocks ? effective_used - low_used_blocks : 0);
    }

    bool& triggered = heaps_[group_set.groupSetId()].watermark_triggered[static_cast<size_t>(tier)];
    if (required_count == 0 && !has_pending) {
        triggered = false;
    } else if (high_reached) {
        triggered = true;
    }
    return triggered ? required_count : 0;
}

}  // namespace rtp_llm
