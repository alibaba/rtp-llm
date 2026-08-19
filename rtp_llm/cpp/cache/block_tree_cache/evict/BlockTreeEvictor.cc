#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

#include <algorithm>
#include <exception>
#include <string>
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
                                   IsTierEnabledFn                is_tier_enabled,
                                   SettledFn                      settled):
    tree_(tree),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    is_tier_enabled_(std::move(is_tier_enabled)),
    settled_(std::move(settled)),
    task_runner_(std::make_unique<EvictionTaskRunner>(
        tree->groupSets(), transfer_dispatcher, memory_timeout_ms, disk_timeout_ms)) {
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

bool BlockTreeEvictor::isEvictable(const GroupSet& group_set, const TreeNode* node, Tier tier) const {
    const size_t            group_set_id = group_set.groupSetId();
    const GroupSetResource& resource     = node->group_set_resources[group_set_id];
    if (resource.transfer_state != GroupSetTransferState::IDLE) {
        return false;
    }
    switch (tier) {
        case Tier::DEVICE:
            if (!resource.hasCompleteDeviceValue()) {
                return false;
            }
            for (size_t i = 0; i < resource.device_blocks.size(); ++i) {
                if (!group_set.devicePools()[i]->isAllocated(resource.device_blocks[i])) {
                    return false;
                }
            }
            break;
        case Tier::HOST:
            if (!resource.hasTier(Tier::HOST) || !group_set.hostPool()->isAllocated(resource.host_block)) {
                return false;
            }
            break;
        case Tier::DISK:
            if (!resource.hasTier(Tier::DISK) || !group_set.diskPool()->isAllocated(resource.disk_slot)) {
                return false;
            }
            break;
        default:
            return false;
    }

    return group_set.groupType() != CacheGroupType::FULL || tree_->isLeafAtTier(node, group_set_id, tier);
}

// ---- Candidate eligibility gate (design section 4.3) ----
void BlockTreeEvictor::refreshCandidate(GroupSet& group_set, TreeNode* node, Tier tier) {
    if (tier == Tier::NONE) {
        return;
    }
    const size_t  group_set_id = group_set.groupSetId();
    EvictionHeap* heap         = heapFor(group_set_id, tier);
    auto&         resource     = node->group_set_resources[group_set_id];
    if (!isEvictable(group_set, node, tier)) {
        heap->erase(node);
        return;
    }
    heap->upsert(node, resource.candidate_meta);
}

void BlockTreeEvictor::refreshCandidate(TreeNode* node, size_t group_set_id) {
    const GroupSetPtr& group_set      = tree_->groupSets()[group_set_id];
    GroupSetResource&  resource       = node->group_set_resources[group_set_id];
    bool               found_top_tier = false;
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        EvictionHeap* heap = heapFor(group_set_id, tier);
        if (!found_top_tier && resource.hasTier(tier)) {
            found_top_tier = true;
            if (isEvictable(*group_set, node, tier)) {
                heap->upsert(node, resource.candidate_meta);
                continue;
            }
        }
        heap->erase(node);
    }
}

void BlockTreeEvictor::onLoaded(TreeNode* node, size_t group_set_id) {
    GroupSetResource& resource = node->group_set_resources[group_set_id];
    const int64_t tier_enter_time_us            = currentTimeUs();
    resource.candidate_meta.admission_seq      = ++admission_seq_;
    resource.candidate_meta.tier_enter_time_us  = tier_enter_time_us;
    resource.candidate_meta.last_access_time_us = tier_enter_time_us;
    refreshCandidate(node, group_set_id);
}

// ---- Semantic events ----
void BlockTreeEvictor::onInserted(const BlockTreeInsertResult& result) {
    // An existing empty GroupSetResource may be repopulated independently from the
    // node topology. Existing fills precede the newly created suffix in tree
    // traversal order, so admit them first and preserve that ordering in the
    // eviction policy clocks.
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
            const int64_t      insert_time_us           = currentTimeUs();
            GroupSetResource& resource                 = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq    = ++access_seq_;
            resource.candidate_meta.admission_seq      = ++admission_seq_;
            resource.candidate_meta.hit_count          = 0;
            resource.candidate_meta.insert_time_us      = insert_time_us;
            resource.candidate_meta.last_access_time_us = insert_time_us;
            resource.candidate_meta.tier_enter_time_us  = insert_time_us;
            refreshCandidate(*group_set, adopted.first, resource.getTopTier());

            TreeNode* parent = adopted.first->parent;
            if (parent->parent != nullptr) {
                refreshCandidate(*group_set, parent, parent->group_set_resources[group_set_id].getTopTier());
            }
        }
    }

    // Every newly inserted node is offered to every group resource. FULL's topology
    // predicate filters interior nodes; SWA/LINEAR admit every ready node.
    for (TreeNode* node : result.inserted_nodes) {
        const uint64_t access             = ++access_seq_;
        const uint64_t admit              = ++admission_seq_;
        const int64_t  insert_time_us = currentTimeUs();
        for (auto& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            GroupSetResource& resource                 = node->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq    = access;
            resource.candidate_meta.admission_seq      = admit;
            resource.candidate_meta.hit_count          = 0;
            resource.candidate_meta.insert_time_us      = insert_time_us;
            resource.candidate_meta.last_access_time_us = insert_time_us;
            resource.candidate_meta.tier_enter_time_us  = insert_time_us;
            refreshCandidate(*group_set, node, resource.getTopTier());
        }
    }

    // inserted_nodes contains only newly created nodes. If a new suffix is
    // attached below an existing FULL leaf, its direct parent is not in that
    // list and must be refreshed once. Higher ancestors keep the same direct
    // children, and root never participates in eviction.
    TreeNode* existing_parent =
        result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent;
    if (existing_parent != nullptr && existing_parent->parent != nullptr) {
        for (auto& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            refreshCandidate(
                *group_set, existing_parent, existing_parent->group_set_resources[group_set_id].getTopTier());
        }
    }
}

void BlockTreeEvictor::onMatched(const std::vector<TreeNode*>& path) {
    const uint64_t access = ++access_seq_;
    const int64_t  access_time_us = currentTimeUs();
    for (TreeNode* node : path) {
        for (const GroupSetPtr& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            GroupSetResource& resource                  = node->group_set_resources[group_set_id];
            const Tier top      = resource.getTopTier();
            if (top == Tier::NONE) {
                continue;
            }
            resource.candidate_meta.last_access_seq = access;
            resource.candidate_meta.last_access_time_us = access_time_us;
            ++resource.candidate_meta.hit_count;
            EvictionHeap* heap = heapFor(group_set_id, top);
            if (heap->contains(node)) {
                heap->upsert(node, resource.candidate_meta);
            }
        }
    }
}

void BlockTreeEvictor::onTopologyChanged(TreeNode* parent) {
    for (auto& group_set : tree_->groupSets()) {
        refreshCandidate(*group_set, parent, parent->group_set_resources[group_set->groupSetId()].getTopTier());
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

// ---- Eviction selection ----
bool BlockTreeEvictor::evictLocked(size_t group_set_id, Tier source_tier, bool force_drop) {
    auto eviction_desc = chooseVictim(group_set_id, source_tier, force_drop);
    if (!eviction_desc.has_value()) {
        return false;
    }

    auto task = prepareEvictionLocked(*eviction_desc);
    if (!task.has_value() || !task->needsCopy()) {
        return task.has_value();
    }

    auto task_ptr = std::make_shared<EvictionTask>(std::move(*task));
    updatePendingReleases(*task_ptr, true);
    const bool submitted = task_pool_->submit([this, task_ptr]() { runEvictionTask(*task_ptr); });
    if (!submitted) {
        updatePendingReleases(*task_ptr, false);
        abortEvictionLocked(*task_ptr);
        return false;
    }
    return true;
}

void BlockTreeEvictor::runEvictionTask(const EvictionTask& task) {
    EvictionTaskResult task_result;
    task_result.primary_success = false;
    task_result.cascade_success.assign(task.cascade_descs.size(), false);
    try {
        task_result = task_runner_->runTransfer(task, *metrics_reporter_);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
    {
        std::lock_guard<std::mutex> lock(*mutex_);
        settleEvictionLocked(task, task_result);
        updatePendingReleases(task, false);
        settled_(task_result.primary_success,
                 task_result.primary_success
                     && std::all_of(task_result.cascade_success.begin(),
                                    task_result.cascade_success.end(),
                                    [](bool success) { return success; }));
    }
    metrics_reporter_->reportEvictionFinished(task, task_result, tree_->groupSets());
}

std::optional<TransferDescriptor> BlockTreeEvictor::chooseVictim(size_t group_set_id, Tier tier, bool force_drop) {
    GroupSet&     group_set = *tree_->groupSets()[group_set_id];
    EvictionHeap* heap      = heapFor(group_set_id, tier);
    if (heap == nullptr) {
        return std::nullopt;
    }

    while (true) {
        auto entry = heap->best();
        if (!entry.has_value()) {
            return std::nullopt;
        }
        GroupSetResource& resource = entry->node->group_set_resources[group_set_id];
        if (!isEvictable(group_set, entry->node, tier)) {
            heap->erase(entry->node);
            continue;
        }
        Tier target_tier = Tier::NONE;
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
}

void BlockTreeEvictor::scheduleWatermarkEvictionsLocked(Tier tier, double watermark_ratio) {
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        size_t excess = computeGroupSetExcess(*group_set, tier, watermark_ratio);
        while (excess > 0) {
            if (!evictLocked(group_set->groupSetId(), tier, /*force_drop=*/false)) {
                break;
            }
            excess = std::min(excess - 1, computeGroupSetExcess(*group_set, tier, watermark_ratio));
        }
    }
}

void BlockTreeEvictor::updatePendingReleases(const EvictionTask& task, bool reserve) {
    std::lock_guard<std::mutex> lock(pending_release_mutex_);
    auto update_block = [this, reserve](IBlockPool* pool, BlockIdxType block) {
        if (reserve) {
            ++pending_release_counts_[pool];
            return;
        }
        const auto it = pending_release_counts_.find(pool);
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr && it != pending_release_counts_.end() && it->second > 0,
                                "invalid pending eviction release settlement: pool=%s address=%p block=%d pending=%zu",
                                pool ? pool->poolName().c_str() : "<null>",
                                static_cast<void*>(pool),
                                block,
                                it == pending_release_counts_.end() ? 0 : it->second);
        --it->second;
    };
    auto update_desc = [this, &update_block](const TransferDescriptor& desc) {
        const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
        if (desc.source_tier == Tier::DEVICE) {
            const auto& pools = group_set->devicePools();
            for (size_t i = 0; i < pools.size(); ++i) {
                update_block(pools[i].get(), desc.source_blocks[i]);
            }
        } else if (desc.source_tier == Tier::HOST) {
            update_block(group_set->hostPool().get(), desc.source_blocks.front());
        }
    };

    update_desc(task.primary_desc);
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        update_desc(cascade_desc);
    }
}

// ---- Migration pipeline (begin -> copy -> finish) ----
bool BlockTreeEvictor::collectFullPrune(const TransferDescriptor&                  eviction_desc,
                                        EvictionTask&                              task,
                                        std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const {
    if (task.hasFullPrune() || eviction_desc.target_tier != Tier::NONE
        || tree_->groupSets()[eviction_desc.group_set_id]->groupType() != CacheGroupType::FULL) {
        return false;
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
    return true;
}

std::optional<EvictionTask> BlockTreeEvictor::prepareEvictionLocked(TransferDescriptor eviction_desc) {
    EvictionTask task;
    task.primary_timing =
        EvictionTimingSnapshot(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    size_t                                    full_prune_group_set_id = eviction_desc.group_set_id;
    collectFullPrune(eviction_desc, task, detached_resources);
    std::vector<size_t> preexisting_root_transfer_ids;
    if (eviction_desc.target_tier == Tier::NONE) {
        preexisting_root_transfer_ids.reserve(tree_->groupSets().size());
        for (const GroupSetPtr& group_set : tree_->groupSets()) {
            const size_t group_set_id = group_set->groupSetId();
            if (eviction_desc.node->group_set_resources[group_set_id].transfer_state
                != GroupSetTransferState::IDLE) {
                preexisting_root_transfer_ids.push_back(group_set_id);
            }
        }
    }
    task.primary_desc = std::move(eviction_desc);

    std::vector<std::pair<size_t, bool>> cascades =
        selectCascades(task.primary_desc.node, task.primary_desc.group_set_id, task.primary_desc.source_tier);
    task.cascade_descs.reserve(cascades.size());
    for (auto& [cascade_group_set_id, should_cascade] : cascades) {
        if (!should_cascade) {
            continue;
        }
        const GroupSetResource& cascade_resource = task.primary_desc.node->group_set_resources[cascade_group_set_id];
        TransferDescriptor      cascade_desc(task.primary_desc.node,
                                        cascade_group_set_id,
                                        /*path_index=*/0,
                                        task.primary_desc.source_tier,
                                        task.primary_desc.target_tier,
                                        cascade_resource.getBlocks(task.primary_desc.source_tier));
        if (collectFullPrune(cascade_desc, task, detached_resources)) {
            full_prune_group_set_id = cascade_desc.group_set_id;
        }
        task.cascade_descs.push_back(std::move(cascade_desc));
    }

    task.cascade_timings.resize(task.cascade_descs.size());
    task.dependent_prune_timings.resize(task.dependent_prune_descs.size());
    if (!allocateTargets(task, cascades)) {
        return std::nullopt;
    }

    activateTaskLocked(task, detached_resources, preexisting_root_transfer_ids, cascades, full_prune_group_set_id);
    if (!task.needsCopy()) {
        EvictionTaskResult task_result;
        task_result.primary_success = true;
        task_result.cascade_success.assign(task.cascade_descs.size(), true);
        settleEvictionLocked(task, task_result);
        settled_(true, false);
        metrics_reporter_->reportEvictionFinished(task, task_result, tree_->groupSets());
    }
    return task;
}

void BlockTreeEvictor::settleEvictionLocked(const EvictionTask& task, const EvictionTaskResult& task_result) {
    if (!task_result.primary_success) {
        abortEvictionLocked(task);
        return;
    }

    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        completeEvict(dependent_desc);
    }
    completeEvict(task.primary_desc);
    for (size_t i = 0; i < task.cascade_descs.size(); ++i) {
        if (!task_result.cascade_success[i]) {
            releaseTargetBlocks(task.cascade_descs[i]);
            restoreSource(task.cascade_descs[i]);
            continue;
        }
        completeEvict(task.cascade_descs[i]);
    }

    if (task.hasFullPrune()) {
        settleFullPrune(task);
    } else {
        settleSingleEviction(task.primary_desc.node);
    }
}

// Move source blocks out of the resource, install target blocks (if demoting),
// clear the transfer state, and re-admit the node at its new tier. Settlement
// releases only the cache hold; any external holder controls the eventual free.
void BlockTreeEvictor::completeEvict(const TransferDescriptor& desc) {
    const GroupSetPtr& group_set = tree_->groupSets()[desc.group_set_id];
    auto&              resource  = desc.node->group_set_resources[desc.group_set_id];
    if (resource.transfer_detached) {
        releaseTargetBlocks(desc);
        discardDetachedTransfer(desc);
        return;
    }
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING("state mismatch, group_set=%zu node_key=%ld", desc.group_set_id, desc.node->cache_key);
        releaseTargetBlocks(desc);
        return;
    }

    if (desc.target_tier != Tier::NONE) {
        MultiNodeResource target_holder{
            desc.group_set_id, desc.target_tier, {{desc.node, desc.target_blocks}}};
        resource.setBlocks(desc.target_tier, desc.target_blocks);
        group_set->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
        group_set->unreferenceBlocks(target_holder, BlockRefType::EVICTION);
    }

    // DEMOTING is the operation's ownership token. Release its saved source
    // cache hold before clearing the corresponding resource tier. The target is
    // installed while the state is still non-IDLE, then IDLE is published last.
    const MultiNodeResource source_holder{
        desc.group_set_id,
        desc.source_tier,
        {{desc.node, desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockRefType::BLOCK_CACHE);
    resource.evictFromTier(desc.source_tier);
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "eviction settlement produced invalid steady state: group_set_id=%zu node_key=%ld",
                            desc.group_set_id,
                            desc.node->cache_key);

    if (desc.target_tier != Tier::NONE) {
        // Section 7.5: keep last_access_seq / hit_count, refresh the admission clock.
        resource.candidate_meta.admission_seq      = ++admission_seq_;
        resource.candidate_meta.tier_enter_time_us = currentTimeUs();
        refreshCandidate(*group_set, desc.node, desc.target_tier);
    }
}

void BlockTreeEvictor::abortEvictionLocked(const EvictionTask& task) {
    bool detached = task.primary_desc.node->group_set_resources[task.primary_desc.group_set_id].transfer_detached;
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        detached = detached
                   || cascade_desc.node->group_set_resources[cascade_desc.group_set_id].transfer_detached;
    }

    releaseTargetBlocks(task.primary_desc);
    restoreSource(task.primary_desc);
    for (const auto& cascade_desc : task.cascade_descs) {
        releaseTargetBlocks(cascade_desc);
        restoreSource(cascade_desc);
    }
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        restoreSource(dependent_desc);
    }
    if (detached) {
        settleSingleEviction(task.primary_desc.node);
    }
}

bool BlockTreeEvictor::allocateTargets(EvictionTask& task, std::vector<std::pair<size_t, bool>>& cascades) {
    auto prepare_target = [](TransferDescriptor& desc) {
        if (desc.target_tier != Tier::NONE) {
            desc.target_blocks.resize(1, NULL_BLOCK_IDX);
        }
    };
    prepare_target(task.primary_desc);
    for (TransferDescriptor& cascade_desc : task.cascade_descs) {
        prepare_target(cascade_desc);
    }

    auto allocate_target = [this](TransferDescriptor& desc) {
        if (desc.target_tier == Tier::NONE) {
            return true;
        }
        BlockIdxType target =
            tree_->groupSets()[desc.group_set_id]->allocateSingleBlock(desc.target_tier, BlockRefType::EVICTION);
        if (isNullBlockIdx(target)) {
            return false;
        }
        desc.target_blocks.front() = target;
        return true;
    };

    if (!allocate_target(task.primary_desc)) {
        return false;
    }
    for (size_t i = 0; i < task.dependent_prune_descs.size(); ++i) {
        const TransferDescriptor& desc = task.dependent_prune_descs[i];
        task.dependent_prune_timings[i] =
            EvictionTimingSnapshot(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }

    size_t retained_cascades  = 0;
    size_t cascade_desc_index = 0;
    for (auto& [_, should_cascade] : cascades) {
        if (!should_cascade) {
            continue;
        }
        TransferDescriptor& cascade_desc = task.cascade_descs[cascade_desc_index];
        if (!allocate_target(cascade_desc)) {
            should_cascade = false;
            ++cascade_desc_index;
            continue;
        }
        if (retained_cascades != cascade_desc_index) {
            task.cascade_descs[retained_cascades] = std::move(cascade_desc);
        }
        const TransferDescriptor& desc = task.cascade_descs[retained_cascades];
        task.cascade_timings[retained_cascades] =
            EvictionTimingSnapshot(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
        ++retained_cascades;
        ++cascade_desc_index;
    }
    task.cascade_descs.resize(retained_cascades);
    task.cascade_timings.resize(retained_cascades);
    return true;
}

void BlockTreeEvictor::activateTaskLocked(const EvictionTask&                              task,
                                          const std::vector<std::pair<TreeNode*, size_t>>& detached_resources,
                                          const std::vector<size_t>&                  preexisting_root_transfer_ids,
                                          const std::vector<std::pair<size_t, bool>>& cascades,
                                          size_t                                      full_prune_group_set_id) {
    reserveSource(task.primary_desc);
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        reserveSource(cascade_desc);
    }
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        reserveSource(dependent_desc);
    }
    for (const auto& [group_set_id, should_cascade] : cascades) {
        if (!should_cascade) {
            refreshCandidate(task.primary_desc.node, group_set_id);
        }
    }
    for (const auto& [node, group_set_id] : detached_resources) {
        node->group_set_resources[group_set_id].transfer_detached = true;
    }
    if (task.hasFullPrune()) {
        for (size_t group_set_id : preexisting_root_transfer_ids) {
            task.primary_desc.node->group_set_resources[group_set_id].transfer_detached = true;
        }
        RTP_LLM_LOG_WARNING("event=block_tree_full_prune root_key=%ld trigger_group_set_id=%zu source_tier=%s "
                            "closure_nodes=%zu dependent_resources=%zu detached_resources=%zu",
                            task.primary_desc.node->cache_key,
                            full_prune_group_set_id,
                            tierName(task.primary_desc.source_tier),
                            task.full_prune_nodes_bottom_up.size(),
                            task.dependent_prune_descs.size(),
                            detached_resources.size());
    }
}

// Reserve the source: exclude it from all heaps and mark the in-flight state so
// no other selector can pick it. The caller must first verify that the resource
// is IDLE; assigning DEMOTING is not safe for an already reserved resource.
void BlockTreeEvictor::reserveSource(const TransferDescriptor& eviction_desc) {
    eviction_desc.node->group_set_resources[eviction_desc.group_set_id].transfer_state =
        GroupSetTransferState::DEMOTING;
    heapFor(eviction_desc.group_set_id, eviction_desc.source_tier)->erase(eviction_desc.node);
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
void BlockTreeEvictor::restoreSource(const TransferDescriptor& eviction_desc) {
    auto              group_set_id = eviction_desc.group_set_id;
    GroupSetResource& resource = eviction_desc.node->group_set_resources[group_set_id];
    if (resource.transfer_detached) {
        discardDetachedTransfer(eviction_desc);
        return;
    }
    if (resource.transfer_state != GroupSetTransferState::DEMOTING) {
        RTP_LLM_LOG_WARNING(
            "state mismatch, group_set=%zu node_key=%ld", eviction_desc.group_set_id, eviction_desc.node->cache_key);
        return;
    }
    resource.transfer_state = GroupSetTransferState::IDLE;
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE)
                                || resource.hasCompleteDeviceValue(),
                            "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                            eviction_desc.group_set_id,
                            eviction_desc.node->cache_key);
    refreshCandidate(*tree_->groupSets()[group_set_id], eviction_desc.node, eviction_desc.source_tier);
}

void BlockTreeEvictor::discardDetachedTransfer(const TransferDescriptor& transfer_desc) {
    GroupSetResource& resource = transfer_desc.node->group_set_resources[transfer_desc.group_set_id];
    const GroupSetPtr& group_set = tree_->groupSets()[transfer_desc.group_set_id];
    const MultiNodeResource source_holder{
        transfer_desc.group_set_id,
        transfer_desc.source_tier,
        {{transfer_desc.node, transfer_desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockRefType::BLOCK_CACHE);
    resource.evictFromTier(transfer_desc.source_tier);
    resource.transfer_state    = GroupSetTransferState::IDLE;
    resource.transfer_detached = false;
}

void BlockTreeEvictor::releaseTargetBlocks(const TransferDescriptor& eviction_desc) {
    if (eviction_desc.target_blocks.empty()) {
        return;
    }
    auto& group_set = tree_->groupSets()[eviction_desc.group_set_id];
    for (auto block : eviction_desc.target_blocks) {
        group_set->releaseSingleBlock(eviction_desc.target_tier, block, BlockRefType::EVICTION);
    }
}

void BlockTreeEvictor::settleSingleEviction(TreeNode* node) {
    if (tree_->isRemovable(node)) {
        RTP_LLM_LOG_DEBUG("deleting empty node key=%ld", node->cache_key);
        eraseNodeFromAllHeaps(node);
        TreeNode* surviving_ancestor = tree_->removeNodeAndEmptyAncestors(node);
        if (surviving_ancestor != tree_->root()) {
            onTopologyChanged(surviving_ancestor);
        }
    } else if (node->parent && node->parent != tree_->root()) {
        onTopologyChanged(node->parent);
    }
}

void BlockTreeEvictor::eraseNodeFromAllHeaps(TreeNode* node) {
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            heapFor(group_set->groupSetId(), tier)->erase(node);
        }
    }
}

void BlockTreeEvictor::settleFullPrune(const EvictionTask& task) {
    TreeNode* const boundary_node = task.primary_desc.node->parent;
    for (TreeNode* node : task.full_prune_nodes_bottom_up) {
        if (tree_->isRemovable(node)) {
            eraseNodeFromAllHeaps(node);
            tree_->removeNode(node);
        } else {
            onTopologyChanged(node);
        }
    }
    TreeNode* survivor = tree_->removeNodeAndEmptyAncestors(boundary_node);
    if (survivor != tree_->root()) {
        onTopologyChanged(survivor);
    }
}

std::vector<std::pair<size_t, bool>>
BlockTreeEvictor::selectCascades(const TreeNode* node, size_t source_group_set_id, Tier tier) const {
    std::vector<std::pair<size_t, bool>> result;
    if (tree_->isLeafAtTier(node, source_group_set_id, tier)) {
        for (const auto& group_set : tree_->groupSets()) {
            if (group_set->groupSetId() != source_group_set_id) {
                result.emplace_back(group_set->groupSetId(), isEvictable(*group_set, node, tier));
            }
        }
        return result;
    }

    // Forward cascading is intrinsic to group-resource priority: when a high-priority
    // resource is evicted, lower-priority resources on the same node follow.
    CacheGroupType source_type = tree_->groupSets()[source_group_set_id]->groupType();
    for (const auto& group_set : tree_->groupSets()) {
        bool below = false;
        switch (source_type) {
            case CacheGroupType::FULL:
                below =
                    (group_set->groupType() == CacheGroupType::SWA || group_set->groupType() == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::SWA:
                below = (group_set->groupType() == CacheGroupType::LINEAR);
                break;
            case CacheGroupType::LINEAR:
                below = false;
                break;
        }
        if (below) {
            result.emplace_back(group_set->groupSetId(), isEvictable(*group_set, node, tier));
        }
    }
    return result;
}

size_t BlockTreeEvictor::poolWatermarkExcess(IBlockPool* pool, double ratio) const {
    size_t used          = pool->usedBlocksNum();
    size_t pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(pending_release_mutex_);
        const auto                  pending = pending_release_counts_.find(pool);
        if (pending != pending_release_counts_.end()) {
            pending_count = pending->second;
        }
    }
    if (pending_count > 0) {
        RTP_LLM_CHECK_WITH_INFO(pending_count <= used,
                                "pending eviction releases exceed used blocks: "
                                "pool=%s address=%p pending=%zu used=%zu total=%zu ratio=%f",
                                pool->poolName().c_str(),
                                static_cast<void*>(pool),
                                pending_count,
                                used,
                                pool->totalBlocksNum(),
                                ratio);
        used -= pending_count;
    }
    const size_t threshold = static_cast<size_t>(pool->totalBlocksNum() * ratio);
    return used > threshold ? used - threshold : 0;
}

size_t BlockTreeEvictor::computeGroupSetExcess(const GroupSet& group_set, Tier tier, double ratio) const {
    RTP_LLM_CHECK_WITH_INFO(ratio > 0.0,
                            "watermark ratio must be positive: group_set=%zu tier=%s ratio=%f",
                            group_set.groupSetId(),
                            tierName(tier),
                            ratio);
    if (tier == Tier::DEVICE) {
        size_t excess = 0;
        for (const DeviceBlockPoolPtr& pool : group_set.devicePools()) {
            excess = std::max(excess, poolWatermarkExcess(pool.get(), ratio));
        }
        return excess;
    }
    if (tier == Tier::HOST) {
        const auto pool = group_set.hostPool();
        return pool ? poolWatermarkExcess(pool.get(), ratio) : 0;
    }
    if (tier == Tier::DISK) {
        const auto pool = group_set.diskPool();
        return pool ? poolWatermarkExcess(pool.get(), ratio) : 0;
    }
    return 0;
}

}  // namespace rtp_llm
