#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

BlockTreeEvictor::BlockTreeEvictor(BlockTree*                     tree,
                                   const BlockTreeCacheConfig&    config,
                                   const BlockTransferDispatcher* transfer_dispatcher,
                                   BlockTreeTaskPool*             task_pool,
                                   BlockTreeCacheMetricsReporter& metrics_reporter,
                                   std::mutex&                    mutex,
                                   std::function<void()>          schedule_capacity_check):
    tree_(tree),
    config_(config),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    schedule_capacity_check_(std::move(schedule_capacity_check)),
    task_runner_(
        std::make_unique<EvictionTaskRunner>(tree->groupSets(), transfer_dispatcher, task_pool, metrics_reporter)),
    watermark_strategies_{
        WatermarkEvictionStrategy::create(Tier::DEVICE, config, tree->groupSets().size(), metrics_reporter),
        WatermarkEvictionStrategy::create(Tier::HOST, config, tree->groupSets().size(), metrics_reporter),
        WatermarkEvictionStrategy::create(Tier::DISK, config, tree->groupSets().size(), metrics_reporter)},
    on_demand_strategy_(metrics_reporter) {
    heaps_.resize(tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        auto& tier_heaps  = heaps_[group_set_id];
        tier_heaps.device = std::make_unique<EvictionHeap>(config.device_eviction_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(config.host_eviction_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(config.disk_eviction_policy);
        tier_heaps.next_device_evict_log.resize(tree_->groupSets()[group_set_id]->devicePools().size());
    }
}

BlockTreeEvictor::~BlockTreeEvictor() {
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
    if (node->parent == nullptr || node->is_resident || target_tier == Tier::NONE
        || node->group_set_resources[group_set_id].transfer_state != GroupSetTransferState::IDLE) {
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
    updatePendingReleaseBlocks(result.revoked_evictions, /*reserve=*/false, /*revoked=*/true);
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

EvictionVictimMetrics BlockTreeEvictor::getAndResetEvictionVictimMetrics() const {
    return eviction_victim_metrics_.getAndReset();
}

void BlockTreeEvictor::logZeroDeviceEvictLocked(size_t group_set_id,
                                                size_t member_group_id,
                                                size_t required_blocks,
                                                size_t rejected_candidates) {
    auto&      tier_heaps = heaps_[group_set_id];
    auto&      next_log   = tier_heaps.next_device_evict_log[member_group_id];
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
    const bool has_unfrozen_candidate =
        heap.best(
                [&group_set, group_set_id](TreeNode* node) {
                    const auto& blocks = node->group_set_resources[group_set_id].device_blocks;
                    for (size_t i = 0; i < blocks.size(); ++i) {
                        if (!isNullBlockIdx(blocks[i]) && group_set->devicePools()[i]->isFrozen(blocks[i])) {
                            return false;
                        }
                    }
                    return true;
                },
                nullptr)
            .has_value();
    // A candidate covers at most one block in the requested member pool. The
    // difference is only a lower bound, since candidates may share blocks or hold other references.
    const auto cache_only_outside_heap_min =
        stats.tree.cache_only_blocks - std::min(stats.tree.cache_only_blocks, candidates);
    const auto   pending                   = pending_release_counts_.find(pool.get());
    const size_t pending_eviction_releases = pending == pending_release_counts_.end() ? 0 : pending->second;

    // Reference counts can overlap. Candidate counts are tree nodes, not physical blocks.
    RTP_LLM_LOG_WARNING(
        "BTC on-demand DEVICE evict freed no blocks: pool=%s group_set=%zu group_type=%s required_blocks=%zu "
        "total_blocks=%zu free_blocks=%zu active_blocks=%zu cache_only_blocks=%zu cache_only_outside_heap_min=%zu "
        "device_candidate_nodes=%zu has_unfrozen_candidate=%d rejected_candidate_nodes=%zu pending_eviction_releases=%zu "
        "cache_ref_blocks=%zu load_ref_blocks=%zu eviction_ref_blocks=%zu "
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
        stats.tree.blocksWithRef(BlockTreeRefType::EVICT),
        stats.tree.blocksWithRef(BlockTreeRefType::FROZEN),
        stats.request_ref_blocks,
        stats.remote_connector_ref_blocks,
        stats.pd_connector_ref_blocks);
}

EvictionBatchResult BlockTreeEvictor::batchDropLocked(size_t                group_set_id,
                                                      Tier                  source_tier,
                                                      size_t                required_block_count,
                                                      std::optional<size_t> initial_free,
                                                      std::optional<size_t> member_group_id) {
    const auto& group_set          = *tree_->groupSets()[group_set_id];
    auto*       watermark_strategy = initial_free ? nullptr : watermarkStrategy(source_tier);
    const bool  device_request     = initial_free && source_tier == Tier::DEVICE;
    const auto* pool =
        device_request ? group_set.devicePools()[member_group_id.value()].get() : group_set.hostOrDiskPool(source_tier);
    const EvictionStrategy& strategy =
        watermark_strategy ? static_cast<const EvictionStrategy&>(*watermark_strategy) : on_demand_strategy_;
    const size_t         batch_limit = strategy.evictionBatchSize(group_set, required_block_count);
    EvictionBatchContext context;
    context.watermark_drop = watermark_strategy != nullptr;
    context.drop_pool      = initial_free && !device_request ? pool : nullptr;
    EvictionBatchResult result{required_block_count};
    auto                scope = DropScope::CASCADE;
    for (size_t drops = 0; result.freed < batch_limit && (device_request || drops < batch_limit);) {
        auto victim = chooseVictim(group_set_id, source_tier, /*force_drop=*/true, &context, scope);
        if (!victim) {
            const auto group_type = group_set.groupType();
            if (device_request && scope == DropScope::CASCADE
                && (group_type == CacheGroupType::SWA || group_type == CacheGroupType::LINEAR)) {
                scope   = DropScope::PRIMARY_ONLY;
                context = {};
                continue;
            }
            break;
        }
        runDropTask(std::move(*victim), &context, scope);
        ++drops;
        result.dropped = true;
        if (watermark_strategy) {
            const size_t remaining_count =
                watermark_strategy->computeWatermarkEvictCount(group_set, pending_release_counts_);
            result.freed =
                std::max(result.freed, required_block_count - std::min(required_block_count, remaining_count));
        } else {
            const size_t current_free = pool->freeBlocksNum();
            result.freed              = current_free - std::min(current_free, *initial_free);
        }
    }
    if (watermark_strategy) {
        result.needs_next_batch = watermark_strategy->finishBatchDrop(group_set, result);
    } else {
        on_demand_strategy_.finishBatchDrop(source_tier, group_set, result);
        if (device_request && result.freed == 0 && result.required != 0) {
            logZeroDeviceEvictLocked(
                group_set_id, *member_group_id, result.required, context.rejected_candidates.size());
        }
    }
    return result;
}

bool BlockTreeEvictor::submitEvictionTask(std::vector<TransferDescriptor>    descriptors,
                                          std::vector<EvictionTimingMetrics> timings) {
    const auto timeout =
        std::chrono::milliseconds(descriptors.front().source_tier == Tier::HOST ? config_.host_to_disk_timeout_ms :
                                                                                  config_.device_host_timeout_ms);
    auto task_ptr =
        std::make_shared<EvictionTransferTask>(TransferTask(std::move(descriptors), timeout), std::move(timings));
    for (size_t i = 0; i < task_ptr->descriptors().size(); ++i) {
        const auto& desc = task_ptr->descriptors()[i];
        heaps_[desc.group_set_id].pending_timings.emplace(std::make_pair(desc.node, desc.generation),
                                                          task_ptr->timings[i]);
    }
    updatePendingRelease(task_ptr->descriptors(), true);
    if (!task_runner_->submit(task_ptr,
                              [this, task_ptr](bool success) { scheduleEvictionSettlement(task_ptr, success); })) {
        for (const auto& desc : task_ptr->descriptors()) {
            heaps_[desc.group_set_id].pending_timings.erase({desc.node, desc.generation});
        }
        updatePendingRelease(task_ptr->descriptors(), false);
        rollbackTransferLocked(task_ptr->descriptors());
        metrics_reporter_->reportEvictionFinished(*task_ptr, tree_->groupSets(), false, {}, 0);
        return false;
    }
    return true;
}

size_t BlockTreeEvictor::batchEvictLocked(size_t group_set_id, Tier source_tier, size_t max_victim_count) {
    const auto* watermark_strategy = watermarkStrategy(source_tier);
    if (!watermark_strategy || watermark_strategy->targetTier() == Tier::NONE) {
        return 0;
    }
    const Tier                      target_tier = watermark_strategy->targetTier();
    EvictionBatchContext            context;
    std::vector<TransferDescriptor> descriptors;
    while (descriptors.size() < max_victim_count) {
        auto victim = chooseVictim(group_set_id, source_tier, false, &context);
        if (!victim) {
            break;
        }
        if (descriptors.empty()) {
            descriptors.reserve(max_victim_count);
        }
        context.previous_candidate = victim->node;
        descriptors.push_back(std::move(*victim));
    }
    if (descriptors.empty()) {
        return 0;
    }
    const size_t count         = descriptors.size();
    const auto&  group_set     = tree_->groupSets()[group_set_id];
    auto         target_blocks = group_set->allocateBlocks(count, target_tier, BlockTreeRefType::EVICT);
    if (!target_blocks.has_value()) {
        const IBlockPool* pool = group_set->hostOrDiskPool(target_tier);
        RTP_LLM_INTERVAL_LOG(5,
                             WARN,
                             "eviction target allocation failed: group_set=%zu source=%s target=%s blocks=%zu "
                             "target_pool=%s",
                             group_set_id,
                             tierName(source_tier),
                             tierName(target_tier),
                             count,
                             pool ? pool->debugString().c_str() : "<null>");
        metrics_reporter_->reportEvictionFinished(
            EvictionTransferTask(TransferTask(std::move(descriptors), std::chrono::milliseconds(0))),
            tree_->groupSets(),
            false,
            {},
            0);
        return 0;
    }
    std::vector<EvictionTimingMetrics> timings;
    timings.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto& desc         = descriptors[i];
        auto& resource     = desc.node->group_set_resources[desc.group_set_id];
        desc.target_blocks = {(*target_blocks)[i]};
        timings.emplace_back(resource.candidate_meta);
        suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
        resource.eviction_access_seq = resource.candidate_meta.last_access_seq;
        resource.transfer_state      = GroupSetTransferState::EVICTING;
        tree_->refreshPublishedState(desc.node);
    }
    return submitEvictionTask(std::move(descriptors), std::move(timings)) ? count : 0;
}

void BlockTreeEvictor::scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task, bool success) {
    auto stage_context = CacheStageContext::forTransfer(*metrics_reporter_, task->transfer_task, "evict");
    auto timers        = CompletionStageTimers::create(stage_context);
    auto settle        = [this, task = std::move(task), success, stage_context, timers]() noexcept {
        timers.finishQueue(success);
        block_tree_cache_detail::ScopeRollback finish([&] { timers.finish(success); });
        auto execute = TransferStageTimer::scope(stage_context, "completion_execute", "business_completion", success);
        size_t            cancelled_blocks           = 0;
        bool              has_valid_eviction;
        std::vector<bool> skip_eviction_metrics;
        skip_eviction_metrics.reserve(task->descriptors().size());
        const auto&        descriptor = task->descriptors().front();
        TransferStageTimer watermark_timer;
        {
            CacheLockGuard lock(*mutex_, "watermark", stage_context);
            has_valid_eviction =
                settleEvictLocked(task->descriptors(), success, cancelled_blocks, skip_eviction_metrics, stage_context);
            watermark_timer = TransferStageTimer(stage_context, "completion_watermark", "business_completion");
            if (!success) {
                watermarkStrategy(descriptor.source_tier)->stopWatermark(descriptor.group_set_id);
            }
        }
        if (success && has_valid_eviction) {
            schedule_capacity_check_();
        } else {
            scheduleWatermarkEvictions(descriptor.source_tier);
        }
        watermark_timer.finish(success);
        if (has_valid_eviction) {
            auto metrics =
                TransferStageTimer::scope(stage_context, "completion_metrics", "business_completion", success);
            metrics_reporter_->reportEvictionFinished(
                *task, tree_->groupSets(), success, skip_eviction_metrics, cancelled_blocks);
        }
    };

    if (!task_pool_->submitCompletion(settle)) {
        RTP_LLM_LOG_WARNING("eviction completion queue is closed; settling inline");
        settle();
    }
}

bool BlockTreeEvictor::settleEvictLocked(DescriptorView           descs,
                                         bool                     success,
                                         size_t&                  cancelled_blocks,
                                         std::vector<bool>&       skip_eviction_metrics,
                                         const CacheStageContext& stage_context) {
    bool has_valid_eviction = false;
    {
        auto cleanup = TransferStageTimer::scope(stage_context, "completion_cleanup", "business_completion", success);
        updatePendingRelease(descs, false);
        for (const TransferDescriptor& desc : descs) {
            heaps_[desc.group_set_id].pending_timings.erase({desc.node, desc.generation});
            const auto& resource          = desc.node->group_set_resources[desc.group_set_id];
            const bool  is_valid_eviction = desc.isCurrentOwner() && !resource.transfer_detached;
            const bool  cancelled_by_access =
                is_valid_eviction && success && resource.candidate_meta.last_access_seq != resource.eviction_access_seq;
            cancelled_blocks += cancelled_by_access;
            skip_eviction_metrics.push_back(!is_valid_eviction || cancelled_by_access);
            has_valid_eviction |= is_valid_eviction;
        }
    }

    auto settlement = TransferStageTimer::scope(
        stage_context, success ? "completion_tree_update" : "completion_rollback", "business_completion", success);
    if (success) {
        commitTransferLocked(descs);
    } else {
        rollbackTransferLocked(descs);
    }
    return has_valid_eviction;
}

std::optional<EvictionVictimSkipReason>
BlockTreeEvictor::watermarkEvictionSkipReason(TreeNode* node, size_t group_set_id, Tier tier) const {
    const auto& group    = tree_->groupSets()[group_set_id];
    const auto& resource = node->group_set_resources[group_set_id];
    if (tier == Tier::DEVICE) {
        for (size_t i = 0; i < resource.device_blocks.size(); ++i) {
            const auto block = resource.device_blocks[i];
            if (isNullBlockIdx(block)) {
                continue;
            }
            if (group->devicePools()[i]->hasRequestRef(block)) {
                return EvictionVictimSkipReason::REQUEST_REF;
            }
            if (group->devicePools()[i]->isFrozen(block)) {
                return EvictionVictimSkipReason::FROZEN;
            }
        }
        if (group->groupType() == CacheGroupType::FULL
            && std::any_of(node->children.begin(), node->children.end(), [group_set_id](const auto& child) {
                   const auto state = child.second->group_set_resources[group_set_id].transfer_state;
                   return state == GroupSetTransferState::LOAD_PENDING || state == GroupSetTransferState::LOADING;
               })) {
            return EvictionVictimSkipReason::CHILD_LOADING;
        }
    } else if ((tier == Tier::HOST && !isNullBlockIdx(resource.host_block)
                && group->hostPool()->isFrozen(resource.host_block))
               || (tier == Tier::DISK && !isNullBlockIdx(resource.disk_block)
                   && group->diskPool()->isFrozen(resource.disk_block))) {
        return EvictionVictimSkipReason::FROZEN;
    }
    return std::nullopt;
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(
    size_t group_set_id, Tier tier, bool force_drop, EvictionBatchContext* eviction_batch_context, DropScope scope) {
    EvictionHeap* heap = heapFor(group_set_id, tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    EvictionBatchContext local_eviction_batch_context;
    if (eviction_batch_context == nullptr) {
        eviction_batch_context = &local_eviction_batch_context;
    }
    // Instrumentation adds no allocation, atomics, extra locks, or reporting in the selection loop.
    auto* counters = metrics_reporter_->enabled() ?
                         &eviction_victim_metrics_.at((force_drop && !eviction_batch_context->watermark_drop) ?
                                                          EvictionVictimTrigger::FORCE_DROP :
                                                          EvictionVictimTrigger::WATERMARK,
                                                      tier,
                                                      tree_->groupSets()[group_set_id]->groupType(),
                                                      force_drop) :
                         nullptr;
    if (counters != nullptr) {
        counters->candidates_sum += heap->size();
    }
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
    struct SelectionContext {
        EvictionVictimCounters* counters;
        EvictionBatchContext*   eviction_batch_context;

        bool skip(EvictionVictimSkipReason reason) const {
            if (counters != nullptr) {
                ++counters->skipped[static_cast<size_t>(reason)];
            }
            return false;
        }
    } context{counters, eviction_batch_context};
    const auto can_evict_uncached = [this, group_set_id, tier, force_drop, scope, &context](TreeNode* node) {
        if (force_drop && tier == Tier::DEVICE && isEvictable(node, group_set_id, tier)) {
            const TransferDescriptor desc(node,
                                          group_set_id,
                                          /*path_index=*/0,
                                          tier,
                                          Tier::NONE,
                                          node->group_set_resources[group_set_id].getBlocks(tier),
                                          /*initial_owner=*/true);
            if (!canDropDeviceBlocks(desc, context.eviction_batch_context, scope)) {
                return context.skip(EvictionVictimSkipReason::NOT_RECLAIMABLE);
            }
        }
        if (context.eviction_batch_context->drop_pool != nullptr && isEvictable(node, group_set_id, tier)) {
            const auto& resource = node->group_set_resources[group_set_id];
            const auto  block    = tier == Tier::HOST ? resource.host_block : resource.disk_block;
            if (!context.eviction_batch_context->drop_pool->canReclaimCacheBlock(block)) {
                return context.skip(EvictionVictimSkipReason::NOT_RECLAIMABLE);
            }
        }
        if (!force_drop) {
            if (const auto reason = watermarkEvictionSkipReason(node, group_set_id, tier)) {
                return context.skip(*reason);
            }
        }
        return true;
    };
    // Keep the std::function passed to best() at its original two-pointer capture size.
    const auto can_evict = [&context, &can_evict_uncached](TreeNode* node) {
        if (context.counters != nullptr) {
            ++context.counters->scanned_candidates;
        }
        if (context.eviction_batch_context->rejected_candidates.count(node) != 0) {
            return context.skip(EvictionVictimSkipReason::CACHED_REJECTION);
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
    while ((entry = heap->best(can_evict, eviction_batch_context->previous_candidate)).has_value()) {
        if (isEvictable(entry->node, group_set_id, tier)) {
            break;
        }
        context.skip(EvictionVictimSkipReason::INVALID_CANDIDATE);
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

    GroupSetResource&  resource           = entry->node->group_set_resources[group_set_id];
    const auto*        watermark_strategy = force_drop ? nullptr : watermarkStrategy(tier);
    TransferDescriptor victim(entry->node,
                              group_set_id,
                              /*path_index=*/0,
                              tier,
                              watermark_strategy ? watermark_strategy->targetTier() : Tier::NONE,
                              resource.getBlocks(tier),
                              /*initial_owner=*/true);
    selected = true;
    return victim;
}

WatermarkEvictionStrategy* BlockTreeEvictor::watermarkStrategy(Tier tier) const {
    return watermark_strategies_[static_cast<size_t>(tier)].get();
}

bool BlockTreeEvictor::capacityCheckEnabled() const {
    return config_.usesWatermark(Tier::DEVICE) || config_.usesWatermark(Tier::HOST)
           || config_.usesWatermark(Tier::DISK);
}

void BlockTreeEvictor::checkCapacity() {
    bool needs_next_batch = false;
    for (Tier tier : {Tier::DISK, Tier::HOST, Tier::DEVICE}) {
        if (!config_.usesWatermark(tier)) {
            continue;
        }
        TierWatermark watermark;
        {
            std::lock_guard<std::mutex> lock(*mutex_);
            watermark = config_.watermarkForTier(tier);
        }
        needs_next_batch |= scheduleWatermarkEvictions(tier, watermark);
    }
    if (needs_next_batch) {
        schedule_capacity_check_();
    }
}

bool BlockTreeEvictor::scheduleWatermarkEvictions(Tier tier, std::optional<TierWatermark> watermark) {
    if (tier >= Tier::REMOTE || !watermarkStrategy(tier)) {
        return false;
    }
    auto&                       watermark_strategy = *watermarkStrategy(tier);
    std::lock_guard<std::mutex> schedule_lock(watermark_mutex_);
    const size_t                group_set_count    = tree_->groupSets().size();
    const size_t                first_group_set_id = watermark_strategy.nextGroupSetId();
    bool                        waiting_for_budget = false;
    bool                        needs_next_batch   = false;
    const auto                  stage_context =
        CacheStageContext::forLock(metrics_reporter_, "watermark", tier, watermark_strategy.targetTier());
    for (size_t i = 0; i < group_set_count; ++i) {
        CacheLockGuard lock(*mutex_, "watermark", stage_context);
        needs_next_batch |= scheduleWatermarkGroupLocked(
            (first_group_set_id + i) % group_set_count, tier, watermark, waiting_for_budget);
    }
    return needs_next_batch;
}

bool BlockTreeEvictor::scheduleWatermarkGroupLocked(size_t                       group_set_id,
                                                    Tier                         tier,
                                                    std::optional<TierWatermark> watermark,
                                                    bool&                        waiting_for_budget) {
    const auto& group_set          = *tree_->groupSets()[group_set_id];
    auto&       watermark_strategy = *watermarkStrategy(tier);
    if (!watermark_strategy.startWatermark(group_set_id, watermark)) {
        watermark_strategy.finishTransferRound(group_set);
        return false;
    }
    const Tier   target_tier = watermark_strategy.targetTier();
    const size_t required    = watermark_strategy.computeWatermarkEvictCount(group_set, pending_release_counts_);
    if (target_tier == Tier::NONE) {
        return batchDropLocked(
                   group_set_id, tier, required, /*initial_free=*/std::nullopt, /*member_group_id=*/std::nullopt)
            .needs_next_batch;
    }
    size_t batch_limit = watermark_strategy.transferBatchSize(group_set, required, waiting_for_budget);
    if (batch_limit == 0) {
        return false;
    }
    if (const auto* target_pool = group_set.hostOrDiskPool(target_tier);
        target_pool && !watermarkStrategy(target_tier)) {
        const size_t initial_free         = target_pool->freeBlocksNum();
        const size_t required_block_count = batch_limit - std::min(batch_limit, initial_free);
        const auto   drop_result =
            batchDropLocked(group_set_id, target_tier, required_block_count, initial_free, std::nullopt);
        batch_limit =
            watermark_strategy.transferBatchSizeAfterDrop(group_set, batch_limit, drop_result, waiting_for_budget);
        if (drop_result.dropped && !waiting_for_budget) {
            batch_limit = std::min(batch_limit,
                                   watermark_strategy.computeWatermarkEvictCount(group_set, pending_release_counts_));
        }
    }
    if (!waiting_for_budget) {
        watermark_strategy.finishTransferBatchScheduling(group_set, batchEvictLocked(group_set_id, tier, batch_limit));
    }
    return false;
}

void BlockTreeEvictor::updatePendingRelease(DescriptorView descs, bool reserve) {
    updatePendingReleaseBlocks(descs, reserve, /*revoked=*/false);
    for (const auto& desc : descs) {
        watermarkStrategy(desc.source_tier)->updatePending(*tree_->groupSets()[desc.group_set_id], reserve);
    }
}

void BlockTreeEvictor::updatePendingReleaseBlocks(DescriptorView descs, bool reserve, bool revoked) {
    if (descs.empty()) {
        return;
    }
    std::unordered_map<IBlockPool*, size_t> deltas;
    deltas.reserve(descs.size());
    const auto add_delta = [&deltas](IBlockPool* pool, size_t delta) {
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "pending eviction release references a null pool");
        deltas[pool] += delta;
    };
    for (const TransferDescriptor& desc : descs) {
        if (!reserve) {
            if (revoked) {
                if (heaps_[desc.group_set_id].pending_timings.count({desc.node, desc.generation}) == 0) {
                    continue;
                }
            } else if (!desc.isCurrentOwner()) {
                continue;
            }
        }
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

    if (reserve) {
        for (const auto& [pool, delta] : deltas) {
            pending_release_counts_[pool] += delta;
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
                                                    resource.getBlocks(source_tier),
                                                    /*initial_owner=*/true);
        }
    }
    std::reverse(task.full_prune_nodes_bottom_up.begin(), task.full_prune_nodes_bottom_up.end());
}

void BlockTreeEvictor::runDropTask(TransferDescriptor    eviction_desc,
                                   EvictionBatchContext* eviction_batch_context,
                                   DropScope             scope) {
    const EvictionDropTask task = createDropTask(std::move(eviction_desc), eviction_batch_context, scope);
    for (const auto& desc : task.dependent_prune_descs) {
        commitDropLocked(desc);
    }
    if (!task.primary_desc.node->group_set_resources[task.primary_desc.group_set_id].transfer_detached) {
        commitDropLocked(task.primary_desc);
    }
    for (const auto& desc : task.cascade_descs) {
        commitDropLocked(desc);
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
    metrics_reporter_->reportEvictionFinished(
        task,
        tree_->groupSets(),
        eviction_batch_context && eviction_batch_context->watermark_drop ? "watermark" : "force_drop");
}

void BlockTreeEvictor::commitDropLocked(const TransferDescriptor& desc) {
    const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
    auto&              resource  = desc.node->group_set_resources[desc.group_set_id];
    RTP_LLM_CHECK_WITH_INFO(resource.transfer_state == GroupSetTransferState::IDLE,
                            "drop completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s",
                            desc.group_set_id,
                            desc.node->cache_key,
                            static_cast<int>(resource.transfer_state),
                            tierName(desc.source_tier));

    suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
    const MultiNodeResource source_holder{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    resource.evictFromTier(desc.source_tier);
    tree_->refreshPublishedState(desc.node);
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "drop settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            desc.group_set_id,
                            desc.node->cache_key);
}

void BlockTreeEvictor::commitTransferLocked(DescriptorView descs) {
    const TransferDescriptor&       first     = descs.front();
    const GroupSetPtr&              group_set = tree_->groupSets()[first.group_set_id];
    MultiNodeResource               source_holder{first.group_set_id, first.source_tier, {}};
    MultiNodeResource               target_holder{first.group_set_id, first.target_tier, {}};
    MultiNodeResource               active_target_holder{first.group_set_id, first.target_tier, {}};
    std::vector<TransferDescriptor> restore_descs;
    source_holder.node_blocks.reserve(descs.size());
    target_holder.node_blocks.reserve(descs.size());
    active_target_holder.node_blocks.reserve(descs.size());

    const int64_t tier_enter_time_us = currentTimeUs();
    for (const TransferDescriptor& desc : descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        if (!desc.isCurrentOwner()) {
            continue;
        }
        if (resource.transfer_detached) {
            restore_descs.push_back(desc);
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(
            resource.transfer_state == GroupSetTransferState::EVICTING,
            "eviction completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s target=%s",
            desc.group_set_id,
            desc.node->cache_key,
            static_cast<int>(resource.transfer_state),
            tierName(desc.source_tier),
            tierName(desc.target_tier));
        if (resource.candidate_meta.last_access_seq != resource.eviction_access_seq) {
            restore_descs.push_back(desc);
            continue;
        }
        active_target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        source_holder.node_blocks.emplace_back(desc.node, desc.source_blocks);

        resource.evictFromTier(desc.source_tier);
        resource.transfer_state = GroupSetTransferState::IDLE;
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

    if (!active_target_holder.node_blocks.empty()) {
        group_set->referenceBlocks(active_target_holder, BlockTreeRefType::CACHE);
    }
    group_set->unreferenceBlocks(target_holder, BlockTreeRefType::EVICT);
    if (!source_holder.node_blocks.empty()) {
        group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
    }

    restoreSource(restore_descs);
    cleanupTransferNodesLocked(descs);
}

void BlockTreeEvictor::rollbackTransferLocked(DescriptorView descs) {
    unreferenceTargetBlocks(descs);
    restoreSource(descs);
    cleanupTransferNodesLocked(descs);
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
void BlockTreeEvictor::restoreSource(DescriptorView eviction_descs) {
    std::vector<TransferDescriptor> detached_descs;
    for (const TransferDescriptor& desc : eviction_descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (!desc.isCurrentOwner()) {
            continue;
        }
        if (resource.transfer_detached) {
            detached_descs.push_back(desc);
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::EVICTING) {
            RTP_LLM_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", desc.group_set_id, desc.node->cache_key);
            continue;
        }
        resource.transfer_state = GroupSetTransferState::IDLE;
        tree_->refreshPublishedState(desc.node);
        RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                                desc.group_set_id,
                                desc.node->cache_key);

        admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
    }
    discardDetachedTransfer(detached_descs);
}

void BlockTreeEvictor::discardDetachedTransfer(DescriptorView transfer_descs) {
    std::map<std::pair<size_t, Tier>, MultiNodeResource> source_holders;
    for (const TransferDescriptor& desc : transfer_descs) {
        GroupSetResource& resource     = desc.node->group_set_resources[desc.group_set_id];
        const Tier        serving_tier = resource.getTopTier();
        auto&             holder       = source_holders
                           .try_emplace(std::make_pair(desc.group_set_id, serving_tier),
                                        MultiNodeResource{desc.group_set_id, serving_tier, {}})
                           .first->second;
        holder.node_blocks.emplace_back(desc.node, resource.getBlocks(serving_tier));
        resource.evictFromTier(serving_tier);
        resource.transfer_state    = GroupSetTransferState::IDLE;
        resource.transfer_detached = false;
        tree_->refreshPublishedState(desc.node);
    }
    for (const auto& [key, holder] : source_holders) {
        tree_->groupSets()[holder.group_set_id]->unreferenceBlocks(holder, BlockTreeRefType::CACHE);
    }
    for (const auto& desc : transfer_descs) {
        updateFullCandidate(desc.node->parent, desc.group_set_id);
    }
}

void BlockTreeEvictor::invalidateSources(const std::vector<InvalidationSource>& sources) {
    for (const auto& source : sources) {
        const auto path = tree_->findNode(source.cache_keys);
        // findNode returns the surviving prefix when an earlier drop removed this source.
        if (path.size() != source.cache_keys.size()) {
            continue;
        }
        TreeNode*         node     = path.back();
        GroupSetResource& resource = node->group_set_resources[source.group_set_id];
        if (!resource.hasTier(source.source_tier)) {
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::IDLE
            && tree_->groupSets()[source.group_set_id]->groupType() != CacheGroupType::FULL) {
            resource.transfer_detached = true;
            tree_->refreshPublishedState(node);
            continue;
        }
        runDropTask(TransferDescriptor(node,
                                       source.group_set_id,
                                       /*path_index=*/0,
                                       source.source_tier,
                                       Tier::NONE,
                                       resource.getBlocks(source.source_tier),
                                       /*initial_owner=*/true));
    }
}

void BlockTreeEvictor::unreferenceTargetBlocks(DescriptorView descs) {
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
        tree_->groupSets()[first.group_set_id]->unreferenceBlocks(
            first.target_tier, target_blocks, BlockTreeRefType::EVICT);
    }
}

void BlockTreeEvictor::cleanupTransferNodesLocked(DescriptorView descs) {
    std::map<std::pair<size_t, Tier>, MultiNodeResource> source_holders;
    std::unordered_map<TreeNode*, size_t>                pending_nodes;
    std::unordered_set<TreeNode*>                        refresh_nodes;
    pending_nodes.reserve(descs.size());
    refresh_nodes.reserve(descs.size());
    for (const TransferDescriptor& desc : descs) {
        if (!desc.isCurrentOwner()) {
            auto& holder = source_holders
                               .try_emplace(std::make_pair(desc.group_set_id, desc.source_tier),
                                            MultiNodeResource{desc.group_set_id, desc.source_tier, {}})
                               .first->second;
            holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
            ++pending_nodes[desc.node];
        }
        if (desc.node->parent == nullptr) {
            pending_nodes.try_emplace(desc.node, 0);
        } else {
            refresh_nodes.insert(desc.node->parent);
        }
    }

    for (const auto& [key, holder] : source_holders) {
        tree_->groupSets()[holder.group_set_id]->unreferenceBlocks(holder, BlockTreeRefType::EVICT);
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

bool BlockTreeEvictor::canDropDeviceBlocks(const TransferDescriptor& desc,
                                           EvictionBatchContext*     eviction_batch_context,
                                           DropScope                 scope) const {
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
        return scope == DropScope::CASCADE;
    }
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    const EvictionDropTask task = planDropTask(desc, detached_resources, eviction_batch_context, DropScope::CASCADE);
    if (scope == DropScope::PRIMARY_ONLY) {
        // A request-held FULL sibling must actually be in the original cascade.
        for (const auto& sibling : task.cascade_descs) {
            const auto& group_set = tree_->groupSets()[sibling.group_set_id];
            if (sibling.node != desc.node || group_set->groupType() != CacheGroupType::FULL) {
                continue;
            }
            for (size_t i = 0; i < sibling.source_blocks.size(); ++i) {
                const auto block = sibling.source_blocks[i];
                if (!isNullBlockIdx(block) && group_set->devicePools()[i]->hasRequestRef(block)) {
                    return true;
                }
            }
        }
        return false;
    }
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
                                                EvictionBatchContext*                      eviction_batch_context,
                                                DropScope                                  scope) const {
    EvictionDropTask task;
    task.primary_timing =
        EvictionTimingMetrics(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    task.primary_desc = std::move(eviction_desc);
    if (scope == DropScope::PRIMARY_ONLY) {
        return task;
    }
    const TransferDescriptor& primary_desc = task.primary_desc;
    const Tier                target_tier  = primary_desc.target_tier;
    collectFullPrune(primary_desc, task, detached_resources);
    TreeNode* const                           primary_node     = primary_desc.node;
    const size_t                              primary_group_id = primary_desc.group_set_id;
    const Tier                                source_tier      = primary_desc.source_tier;
    const auto&                               group_sets       = tree_->groupSets();
    if (primary_node->group_set_resources[primary_group_id].transfer_state != GroupSetTransferState::IDLE) {
        detached_resources.emplace_back(primary_node, primary_group_id);
    }
    std::vector<std::pair<TreeNode*, size_t>> busy_root_cascades;

    const auto append_desc = [&](TreeNode* node, size_t group_set_id) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(node,
                                        group_set_id,
                                        /*path_index=*/0,
                                        source_tier,
                                        target_tier,
                                        resource.getBlocks(source_tier),
                                        /*initial_owner=*/true);
    };

    const bool           primary_is_leaf = tree_->isLeafAtTier(primary_node, primary_group_id, source_tier);
    const CacheGroupType primary_type    = group_sets[primary_group_id]->groupType();
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
                task.dependent_prune_descs.emplace_back(closure_root,
                                                        group_set_id,
                                                        /*path_index=*/0,
                                                        tier,
                                                        Tier::NONE,
                                                        resource.getBlocks(tier),
                                                        /*initial_owner=*/true);
            }
        }
    }

    if (detached_resources.empty()) {
        selectUpwardCascades(task, eviction_batch_context);
    }

    return task;
}

EvictionDropTask BlockTreeEvictor::createDropTask(TransferDescriptor    eviction_desc,
                                                  EvictionBatchContext* eviction_batch_context,
                                                  DropScope             scope) {
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    EvictionDropTask task = planDropTask(std::move(eviction_desc), detached_resources, eviction_batch_context, scope);
    const auto&      group_sets = tree_->groupSets();

    task.dependent_prune_timings.resize(task.dependent_prune_descs.size());
    for (size_t i = 0; i < task.dependent_prune_descs.size(); ++i) {
        const TransferDescriptor& desc = task.dependent_prune_descs[i];
        task.dependent_prune_timings[i] =
            EvictionTimingMetrics(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
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

    for (const auto& [node, group_set_id] : detached_resources) {
        GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_detached) {
            continue;
        }
        const Tier               tier = resource.getTopTier();
        const TransferDescriptor desc{
            node, group_set_id, 0, tier, Tier::NONE, resource.getBlocks(tier), /*initial_owner=*/true};
        const auto timing = resource.transfer_state == GroupSetTransferState::EVICTING ?
                                heaps_[group_set_id].pending_timings.at({node, resource.generation}) :
                                EvictionTimingMetrics(resource.candidate_meta);
        if (node == task.primary_desc.node && group_set_id == task.primary_desc.group_set_id) {
            task.primary_timing = timing;
        } else {
            metrics_reporter_->reportEvictionDescriptor(
                desc,
                timing,
                group_sets,
                eviction_batch_context && eviction_batch_context->watermark_drop ? "watermark" : "force_drop",
                true);
        }
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
        task.cascade_descs.emplace_back(node,
                                        group_set_id,
                                        /*path_index=*/0,
                                        source_tier,
                                        target_tier,
                                        resource.getBlocks(source_tier),
                                        /*initial_owner=*/true);
    };

    std::vector<TreeNode*> ancestors;
    std::vector<bool>      endpoint_matchable;
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

}  // namespace rtp_llm
