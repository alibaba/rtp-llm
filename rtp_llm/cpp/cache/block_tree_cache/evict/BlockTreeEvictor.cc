#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

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
                                   int                            memory_timeout_ms,
                                   int                            disk_timeout_ms,
                                   size_t                         max_device_host_batch,
                                   size_t                         max_non_device_host_batch,
                                   IsTierEnabledFn                is_tier_enabled,
                                   SettledFn                      settled):
    tree_(tree),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    is_tier_enabled_(std::move(is_tier_enabled)),
    settled_(std::move(settled)),
    task_runner_(std::make_unique<EvictionTaskRunner>(
        tree->groupSets(), transfer_dispatcher, memory_timeout_ms, disk_timeout_ms)),
    disk_timeout_ms_(disk_timeout_ms),
    max_device_host_batch_(max_device_host_batch),
    max_non_device_host_batch_(max_non_device_host_batch) {
    // GroupSetFactory has already validated that group_set_id equals the vector
    // position. Own one heap per (group resource, tier).
    heaps_.resize(tree_->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        auto& tier_heaps  = heaps_[group_set_id];
        tier_heaps.device = std::make_unique<EvictionHeap>(device_policy);
        tier_heaps.host   = std::make_unique<EvictionHeap>(host_policy);
        tier_heaps.disk   = std::make_unique<EvictionHeap>(disk_policy);
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
    return resource.transfer_state == GroupSetTransferState::IDLE && resource.getTopTier() == source_tier
           && (tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL
               || tree_->isLeafAtTier(node, group_set_id, source_tier));
}

void BlockTreeEvictor::suspendCandidate(TreeNode* node, size_t group_set_id, Tier source_tier) {
    if (source_tier != Tier::NONE) {
        heapFor(group_set_id, source_tier)->erase(node);
    }
}

void BlockTreeEvictor::admitCandidate(TreeNode* node, size_t group_set_id, Tier target_tier) {
    if (target_tier == Tier::NONE) {
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
    GroupSetResource& resource                  = node->group_set_resources[group_set_id];
    const int64_t     tier_enter_time_us        = currentTimeUs();
    resource.candidate_meta.admission_seq       = ++admission_seq_;
    resource.candidate_meta.tier_enter_time_us  = tier_enter_time_us;
    resource.candidate_meta.last_access_time_us = tier_enter_time_us;
    admitCandidate(node, group_set_id, Tier::DEVICE);
    updateFullCandidate(node->parent, group_set_id);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInserted(const BlockTreeInsertResult& result) {
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const int64_t     insert_time_us            = currentTimeUs();
            GroupSetResource& resource                  = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq     = ++access_seq_;
            resource.candidate_meta.admission_seq       = ++admission_seq_;
            resource.candidate_meta.hit_count           = 0;
            resource.candidate_meta.insert_time_us      = insert_time_us;
            resource.candidate_meta.last_access_time_us = insert_time_us;
            resource.candidate_meta.tier_enter_time_us  = insert_time_us;
            admitCandidate(adopted.first, group_set_id, resource.getTopTier());
            updateFullCandidate(adopted.first->parent, group_set_id);
        }
    }

    for (TreeNode* node : result.inserted_nodes) {
        const uint64_t access         = ++access_seq_;
        const uint64_t admit          = ++admission_seq_;
        const int64_t  insert_time_us = currentTimeUs();
        for (auto& group_set : tree_->groupSets()) {
            const size_t      group_set_id              = group_set->groupSetId();
            GroupSetResource& resource                  = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq     = access;
            resource.candidate_meta.admission_seq       = admit;
            resource.candidate_meta.hit_count           = 0;
            resource.candidate_meta.insert_time_us      = insert_time_us;
            resource.candidate_meta.last_access_time_us = insert_time_us;
            resource.candidate_meta.tier_enter_time_us  = insert_time_us;
            admitCandidate(node, group_set_id, resource.getTopTier());
        }
    }

    updateFullCandidate(result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent);
}

void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t access         = ++access_seq_;
    const int64_t  access_time_us = currentTimeUs();
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        const size_t group_set_id = group_set->groupSetId();
        const size_t reuse_count  = std::min(group_set->computeReuseBlockCount(path.size()), path.size());
        for (size_t path_index = path.size() - reuse_count; path_index < path.size(); ++path_index) {
            TreeNode*         node     = path[path_index];
            GroupSetResource& resource = node->group_set_resources[group_set_id];
            const Tier        top      = resource.getTopTier();
            if (top == Tier::NONE) {
                continue;
            }
            resource.candidate_meta.last_access_seq     = access;
            resource.candidate_meta.last_access_time_us = access_time_us;
            ++resource.candidate_meta.hit_count;
            heapFor(group_set_id, top)->updateIfPresent(node, resource.candidate_meta);
        }
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

bool BlockTreeEvictor::dropLocked(size_t group_set_id, Tier source_tier, bool notify_settled) {
    std::optional<TransferDescriptor> eviction_desc = chooseVictim(group_set_id, source_tier, /*force_drop=*/true);
    if (!eviction_desc.has_value()) {
        return false;
    }
    runDropTask(std::move(*eviction_desc), notify_settled);
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
    while (remaining_count > 0) {
        std::vector<size_t> used_before;
        used_before.reserve(source_pools.size());
        for (IBlockPool* pool : source_pools) {
            used_before.push_back(pool->usedBlocksNum());
        }
        if (!dropLocked(group_set_id, source_tier, false)) {
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
        settled_(true, false);
    }
    return processed;
}

bool BlockTreeEvictor::submitEvictionTask(EvictionTransferTask task) {
    auto task_ptr = std::make_shared<EvictionTransferTask>(std::move(task));
    updatePendingRelease(task_ptr->descs, true);
    auto on_timeout = [this, task_ptr]() {
        RTP_LLM_LOG_WARNING("eviction expired in business queue, source=%s target=%s descriptors=%zu",
                            tierName(task_ptr->descs.front().source_tier),
                            tierName(task_ptr->descs.front().target_tier),
                            task_ptr->descs.size());
        scheduleEvictionSettlement(task_ptr, false);
        metrics_reporter_->reportTransferTaskQueueWait(CacheTransferOperation::EVICT,
                                                       currentTimeUs() - task_ptr->enqueue_time_us);
    };
    task_ptr->enqueue_time_us = currentTimeUs();
    const bool submitted      = task_pool_->submit([this, task_ptr]() { runEvictionTask(task_ptr); },
                                                   BlockTreeTaskPool::kDefaultQueueWaitTimeout,
                                                   std::move(on_timeout));
    if (!submitted) {
        updatePendingRelease(task_ptr->descs, false);
        rollbackTransferLocked(task_ptr->descs);
        return false;
    }
    return true;
}

bool BlockTreeEvictor::batchEvictLocked(size_t  group_set_id,
                                        Tier    source_tier,
                                        size_t  max_victim_count,
                                        size_t& scheduled_count) {
    scheduled_count = 0;
    if (max_victim_count == 0) {
        return false;
    }
    const Tier target_tier = watermarkTargetTier(source_tier);
    if (target_tier == Tier::NONE) {
        return batchDropLocked(group_set_id, source_tier, max_victim_count, scheduled_count);
    }

    const GroupSetPtr&   group_set = tree_->groupSets()[group_set_id];
    EvictionTransferTask batch;
    for (size_t victim_count = 0; victim_count < max_victim_count; ++victim_count) {
        auto eviction_desc = chooseVictim(group_set_id, source_tier, /*force_drop=*/false);
        if (!eviction_desc.has_value()) {
            break;
        }
        batch.timings.emplace_back(
            eviction_desc->node->group_set_resources[eviction_desc->group_set_id].candidate_meta);
        reserveSource({*eviction_desc});
        batch.descs.push_back(std::move(*eviction_desc));
    }
    if (batch.descs.empty()) {
        return false;
    }

    auto target_blocks = group_set->allocateBlocks(batch.descs.size(), target_tier, BlockTreeRefType::EVICTION);
    if (!target_blocks.has_value()) {
        rollbackTransferLocked(batch.descs);
        return false;
    }
    for (size_t desc_index = 0; desc_index < batch.descs.size(); ++desc_index) {
        batch.descs[desc_index].target_blocks = {(*target_blocks)[desc_index]};
    }
    scheduled_count = batch.descs.size();
    if (!submitEvictionTask(std::move(batch))) {
        scheduled_count = 0;
        return false;
    }
    return true;
}

void BlockTreeEvictor::runEvictionTask(std::shared_ptr<const EvictionTransferTask> task) noexcept {
    try {
        metrics_reporter_->reportTransferTaskQueueWait(CacheTransferOperation::EVICT,
                                                       currentTimeUs() - task->enqueue_time_us);
        task_runner_->runTransfer(
            task, *metrics_reporter_, [this, task](bool success) { scheduleEvictionSettlement(task, success); });
        return;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
    scheduleEvictionSettlement(std::move(task), false);
}

void BlockTreeEvictor::scheduleEvictionSettlement(std::shared_ptr<const EvictionTransferTask> task,
                                                  bool                                        success) noexcept {
    auto settle = [this, task = std::move(task), success]() noexcept {
        bool                 any_detached     = false;
        bool                 any_not_detached = false;
        EvictionTransferTask settled_task;
        {
            std::lock_guard<std::mutex> lock(*mutex_);
            std::vector<bool>           detached;
            detached.reserve(task->descs.size());
            for (const TransferDescriptor& desc : task->descs) {
                const bool is_detached = desc.node->group_set_resources[desc.group_set_id].transfer_detached;
                detached.push_back(is_detached);
                any_detached     = any_detached || is_detached;
                any_not_detached = any_not_detached || !is_detached;
            }

            if (success) {
                completeEvict(task->descs);
                settleEviction(task->descs);
            } else {
                rollbackTransferLocked(task->descs);
            }
            updatePendingRelease(task->descs, false);

            for (size_t desc_index = 0; desc_index < task->descs.size(); ++desc_index) {
                const TransferDescriptor& desc = task->descs[desc_index];
                if (success || detached[desc_index]) {
                    TransferDescriptor settled_desc = desc;
                    if (detached[desc_index]) {
                        settled_desc.target_tier = Tier::NONE;
                    }
                    settled_task.descs.push_back(std::move(settled_desc));
                    settled_task.timings.push_back(task->timings[desc_index]);
                }
            }
            settled_(success || any_detached, success && any_not_detached);
        }
        if (!settled_task.descs.empty()) {
            metrics_reporter_->reportEvictionFinished(settled_task, tree_->groupSets());
        }
    };

    bool submitted = false;
    try {
        submitted = task_pool_->submitCompletion(settle);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement with unknown exception");
    }
    if (!submitted) {
        RTP_LLM_LOG_WARNING("eviction completion queue is closed; dropping settlement during shutdown");
    }
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(size_t group_set_id, Tier tier, bool force_drop) {
    EvictionHeap* heap = heapFor(group_set_id, tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    const auto can_evict = [this, group_set_id, tier, force_drop](TreeNode* node) {
        if (force_drop || tier != Tier::DEVICE
            || tree_->groupSets()[group_set_id]->groupType() != CacheGroupType::FULL) {
            return true;
        }
        return std::none_of(node->children.begin(), node->children.end(), [group_set_id](const auto& child_entry) {
            const GroupSetTransferState state = child_entry.second->group_set_resources[group_set_id].transfer_state;
            return state == GroupSetTransferState::LOAD_PENDING || state == GroupSetTransferState::LOADING;
        });
    };
    std::optional<EvictionEntry> entry;
    while ((entry = heap->best(can_evict)).has_value()) {
        if (isEvictable(entry->node, group_set_id, tier)) {
            break;
        }
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
        } else if ((tier == Tier::DEVICE || tier == Tier::HOST) && is_tier_enabled_(Tier::DISK)) {
            target_tier = Tier::DISK;
        }
    }
    return TransferDescriptor(entry->node,
                              group_set_id,
                              /*path_index=*/0,
                              tier,
                              target_tier,
                              resource.getBlocks(tier));
}

Tier BlockTreeEvictor::watermarkTargetTier(Tier source_tier) const {
    if (source_tier == Tier::DEVICE && is_tier_enabled_(Tier::HOST)) {
        return Tier::HOST;
    }
    if ((source_tier == Tier::DEVICE || source_tier == Tier::HOST) && is_tier_enabled_(Tier::DISK)) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

size_t BlockTreeEvictor::watermarkLogicalBatchLimit(Tier source_tier, Tier target_tier) const {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return max_device_host_batch_;
    }
    if (source_tier == Tier::DEVICE && target_tier == Tier::DISK) {
        return 1;
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return max_non_device_host_batch_;
    }
    return std::numeric_limits<size_t>::max();
}

void BlockTreeEvictor::scheduleWatermarkEvictionsLocked(Tier tier, const TierWatermark& watermark) {
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        const size_t required_count = computeWatermarkEvictCount(*group_set, tier, watermark);
        if (required_count == 0) {
            continue;
        }
        const Tier   target_tier     = watermarkTargetTier(tier);
        const size_t batch_count     = std::min(required_count, watermarkLogicalBatchLimit(tier, target_tier));
        size_t       scheduled_count = 0;
        if (batchEvictLocked(group_set->groupSetId(), tier, batch_count, scheduled_count)) {
            metrics_reporter_->reportEvictionTriggered(tier, group_set->groupType(), /*force_drop=*/false);
        }
        metrics_reporter_->reportEvictionBlocks(tier,
                                                group_set->groupType(),
                                                /*force_drop=*/false,
                                                required_count,
                                                scheduled_count);
    }
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
                                                    resource.getBlocks(source_tier));
        }
    }
    std::reverse(task.full_prune_nodes_bottom_up.begin(), task.full_prune_nodes_bottom_up.end());
}

void BlockTreeEvictor::runDropTask(TransferDescriptor eviction_desc, bool notify_settled) {
    const EvictionDropTask task = createDropTask(std::move(eviction_desc));
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
            if (tree_->isRemovable(node)) {
                eraseNodeFromAllHeaps(node);
                tree_->removeNode(node);
            } else {
                updateFullCandidate(node);
            }
        }
        TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(boundary_node);
        updateFullCandidate(survivor);
    } else {
        settleSingleEviction(task.primary_desc.node);
    }
    if (notify_settled) {
        settled_(true, false);
    }
    metrics_reporter_->reportEvictionFinished(task, tree_->groupSets());
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
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "drop settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            desc.group_set_id,
                            desc.node->cache_key);
}

void BlockTreeEvictor::completeEvict(const std::vector<TransferDescriptor>& descs) {
    const TransferDescriptor& first     = descs.front();
    const GroupSetPtr&        group_set = tree_->groupSets()[first.group_set_id];
    MultiNodeResource         source_holder{first.group_set_id, first.source_tier, {}};
    MultiNodeResource         target_holder{first.group_set_id, first.target_tier, {}};
    MultiNodeResource         active_target_holder{first.group_set_id, first.target_tier, {}};
    source_holder.node_blocks.reserve(descs.size());
    target_holder.node_blocks.reserve(descs.size());
    active_target_holder.node_blocks.reserve(descs.size());

    for (const TransferDescriptor& desc : descs) {
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (!resource.transfer_detached) {
            RTP_LLM_CHECK_WITH_INFO(
                resource.transfer_state == GroupSetTransferState::DEMOTING,
                "eviction completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s target=%s",
                desc.group_set_id,
                desc.node->cache_key,
                static_cast<int>(resource.transfer_state),
                tierName(desc.source_tier),
                tierName(desc.target_tier));
            active_target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
        }
        source_holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
        target_holder.node_blocks.emplace_back(desc.node, desc.target_blocks);
    }

    if (!active_target_holder.node_blocks.empty()) {
        group_set->referenceBlocks(active_target_holder, BlockTreeRefType::CACHE);
    }
    group_set->unreferenceBlocks(target_holder, BlockTreeRefType::EVICTION);
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);

    const int64_t tier_enter_time_us = currentTimeUs();
    for (const TransferDescriptor& desc : descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        const bool        detached = resource.transfer_detached;
        resource.evictFromTier(desc.source_tier);
        resource.transfer_state = GroupSetTransferState::IDLE;
        if (detached) {
            resource.transfer_detached = false;
        } else {
            resource.setBlocks(desc.target_tier, desc.target_blocks);
            RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                                    "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                                    desc.group_set_id,
                                    desc.node->cache_key);

            resource.candidate_meta.admission_seq      = ++admission_seq_;
            resource.candidate_meta.tier_enter_time_us = tier_enter_time_us;
            admitCandidate(desc.node, desc.group_set_id, desc.target_tier);
        }
    }
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
        desc.node->group_set_resources[desc.group_set_id].transfer_state = GroupSetTransferState::DEMOTING;
        suspendCandidate(desc.node, desc.group_set_id, desc.source_tier);
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
        resource.transfer_state = GroupSetTransferState::IDLE;
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

    const TransferDescriptor& first = transfer_descs.front();
    MultiNodeResource         source_holder{first.group_set_id, first.source_tier, {}};
    source_holder.node_blocks.reserve(transfer_descs.size());
    for (const TransferDescriptor& desc : transfer_descs) {
        RTP_LLM_CHECK_WITH_INFO(desc.group_set_id == first.group_set_id && desc.source_tier == first.source_tier,
                                "detached eviction batch must share group set and source tier");
        source_holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
    }
    tree_->groupSets()[first.group_set_id]->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);

    for (const TransferDescriptor& desc : transfer_descs) {
        GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        resource.evictFromTier(desc.source_tier);
        resource.transfer_state    = GroupSetTransferState::IDLE;
        resource.transfer_detached = false;
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
        pending_nodes.insert(desc.node);
    }

    while (!pending_nodes.empty()) {
        TreeNode* node = *pending_nodes.begin();
        pending_nodes.erase(node);
        if (!tree_->isRemovable(node)) {
            refresh_nodes.insert(node->parent);
            continue;
        }

        TreeNode* current = node;
        do {
            TreeNode* parent = current->parent;
            pending_nodes.erase(current);
            refresh_nodes.erase(current);
            RTP_LLM_LOG_DEBUG("deleting empty node key=%ld", current->cache_key);
            eraseNodeFromAllHeaps(current);
            tree_->removeNode(current);
            current = parent;
        } while (tree_->isRemovable(current));
        refresh_nodes.insert(current);
    }

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

EvictionDropTask BlockTreeEvictor::createDropTask(TransferDescriptor eviction_desc) {
    EvictionDropTask task;
    task.primary_timing =
        EvictionTimingSnapshot(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    task.primary_desc = std::move(eviction_desc);
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    const TransferDescriptor&                 primary_desc = task.primary_desc;
    const Tier                                target_tier  = primary_desc.target_tier;
    collectFullPrune(primary_desc, task, detached_resources);
    TreeNode* const primary_node     = primary_desc.node;
    const size_t    primary_group_id = primary_desc.group_set_id;
    const Tier      source_tier      = primary_desc.source_tier;
    const auto&     group_sets       = tree_->groupSets();

    const auto append_desc = [&](TreeNode* node, size_t group_set_id) {
        const GroupSetResource& resource = node->group_set_resources[group_set_id];
        task.cascade_descs.emplace_back(
            node, group_set_id, /*path_index=*/0, source_tier, target_tier, resource.getBlocks(source_tier));
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
        if (!isEvictable(primary_node, group_set_id, source_tier)) {
            continue;
        }

        append_desc(primary_node, group_set_id);
        collectFullPrune(task.cascade_descs.back(), task, detached_resources);
    }

    // FullPrune detaches every in-flight resource in its closure, including the
    // closure root skipped by collectFullPrune().
    if (task.hasFullPrune()) {
        for (const GroupSetPtr& group_set : group_sets) {
            const size_t group_set_id = group_set->groupSetId();
            if (primary_node->group_set_resources[group_set_id].transfer_state != GroupSetTransferState::IDLE) {
                detached_resources.emplace_back(primary_node, group_set_id);
            }
        }
    }

    if (detached_resources.empty()) {
        selectUpwardCascades(task);
    }

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

    reserveSource({task.primary_desc});
    reserveSource(task.cascade_descs);
    reserveSource(task.dependent_prune_descs);
    for (const auto& [node, group_set_id] : detached_resources) {
        node->group_set_resources[group_set_id].transfer_detached = true;
    }
    return task;
}

void BlockTreeEvictor::selectUpwardCascades(EvictionDropTask& task) {
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
    for (TreeNode* node = primary_node->parent; node != tree_->root(); node = node->parent) {
        ancestors.push_back(node);
    }
    std::reverse(ancestors.begin(), ancestors.end());

    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(group_sets.size());
    for (const GroupSetPtr& group_set : group_sets) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    std::vector<bool> endpoint_matchable;
    endpoint_matchable.reserve(ancestors.size());
    for (TreeNode* node : ancestors) {
        bool all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
            if (!match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                all_groups_valid = false;
            }
        }
        endpoint_matchable.push_back(all_groups_valid);
    }

    for (size_t path_index = ancestors.size(); path_index > 0; --path_index) {
        TreeNode* parent = ancestors[path_index - 1];
        if (endpoint_matchable[path_index - 1]) {
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

size_t BlockTreeEvictor::computePoolWatermarkRequired(IBlockPool*          pool,
                                                      size_t               pending_count,
                                                      const TierWatermark& watermark,
                                                      bool&                high_reached) const {
    const size_t used = pool->usedBlocksNum();
    if (pending_count > 0) {
        RTP_LLM_CHECK_WITH_INFO(pending_count <= used,
                                "pending eviction releases exceed used blocks: "
                                "pool=%s address=%p pending=%zu used=%zu total=%zu low_ratio=%f high_ratio=%f",
                                pool->poolName().c_str(),
                                static_cast<void*>(pool),
                                pending_count,
                                used,
                                pool->totalBlocksNum(),
                                watermark.low_ratio,
                                watermark.high_ratio);
    }
    const size_t effective_used = used - pending_count;
    const size_t high_used_blocks =
        static_cast<size_t>(std::ceil(static_cast<double>(pool->totalBlocksNum()) * watermark.high_ratio));
    const size_t low_used_blocks =
        static_cast<size_t>(std::floor(static_cast<double>(pool->totalBlocksNum()) * watermark.low_ratio));
    high_reached = high_reached || effective_used >= high_used_blocks;
    return effective_used > low_used_blocks ? effective_used - low_used_blocks : 0;
}

size_t
BlockTreeEvictor::computeWatermarkEvictCount(const GroupSet& group_set, Tier tier, const TierWatermark& watermark) {
    RTP_LLM_CHECK_WITH_INFO(watermark.low_ratio > 0.0 && watermark.low_ratio < watermark.high_ratio
                                && watermark.high_ratio <= 1.0,
                            "watermark ratios must satisfy 0 < low < high <= 1: "
                            "group_set=%zu tier=%s low_ratio=%f high_ratio=%f",
                            group_set.groupSetId(),
                            tierName(tier),
                            watermark.low_ratio,
                            watermark.high_ratio);
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

    std::vector<size_t> pending_counts(pools.size(), 0);
    {
        std::lock_guard<std::mutex> lock(pending_release_mutex_);
        for (size_t pool_index = 0; pool_index < pools.size(); ++pool_index) {
            const auto pending = pending_release_counts_.find(pools[pool_index]);
            if (pending != pending_release_counts_.end()) {
                pending_counts[pool_index] = pending->second;
            }
        }
    }

    bool&  triggered      = heaps_[group_set.groupSetId()].watermark_triggered[static_cast<size_t>(tier)];
    bool   high_reached   = false;
    size_t required_count = 0;
    for (size_t pool_index = 0; pool_index < pools.size(); ++pool_index) {
        required_count = std::max(
            required_count,
            computePoolWatermarkRequired(pools[pool_index], pending_counts[pool_index], watermark, high_reached));
    }

    if (required_count == 0) {
        triggered = false;
    } else if (high_reached) {
        triggered = true;
    }
    return triggered ? required_count : 0;
}

}  // namespace rtp_llm
