#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

#include <algorithm>
#include <exception>
#include <string>
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

void BlockTreeEvictor::updateFullCandidateForTopology(TreeNode* node, size_t group_set_id) {
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
}

void BlockTreeEvictor::onTierChanged(TreeNode* node, size_t group_set_id) {
    TreeNode* parent = node->parent;
    if (parent != tree_->root() && tree_->groupSets()[group_set_id]->groupType() == CacheGroupType::FULL) {
        updateFullCandidateForTopology(parent, group_set_id);
    }
}

// ---- Semantic events ----
void BlockTreeEvictor::onInserted(const BlockTreeInsertResult& result) {
    // An existing empty GroupSetResource may be repopulated independently from the
    // node topology. Existing fills precede the newly created suffix in tree
    // traversal order, so admit them first and preserve that ordering in the
    // eviction policy clocks.
    for (const auto& adopted : result.adopted_nodes) {
        for (size_t group_set_id : adopted.second) {
            const GroupSetPtr& group_set                = tree_->groupSets()[group_set_id];
            const int64_t      insert_time_us           = currentTimeUs();
            GroupSetResource&  resource                 = adopted.first->group_set_resources[group_set_id];
            resource.candidate_meta.last_access_seq     = ++access_seq_;
            resource.candidate_meta.admission_seq       = ++admission_seq_;
            resource.candidate_meta.hit_count           = 0;
            resource.candidate_meta.insert_time_us      = insert_time_us;
            resource.candidate_meta.last_access_time_us = insert_time_us;
            resource.candidate_meta.tier_enter_time_us  = insert_time_us;
            admitCandidate(adopted.first, group_set_id, resource.getTopTier());

            TreeNode* parent = adopted.first->parent;
            if (parent->parent != nullptr && group_set->groupType() == CacheGroupType::FULL) {
                updateFullCandidateForTopology(parent, group_set_id);
            }
        }
    }

    // Every newly inserted node is offered to every group resource. FULL's topology
    // predicate filters interior nodes; SWA/LINEAR admit every ready node.
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

    // inserted_nodes contains only newly created nodes. If a new suffix is
    // attached below an existing FULL leaf, its direct parent is not in that
    // list and must be refreshed once. Higher ancestors keep the same direct
    // children, and root never participates in eviction.
    TreeNode* existing_parent = result.inserted_nodes.empty() ? nullptr : result.inserted_nodes.front()->parent;
    if (existing_parent != nullptr && existing_parent->parent != nullptr) {
        for (auto& group_set : tree_->groupSets()) {
            if (group_set->groupType() == CacheGroupType::FULL) {
                updateFullCandidateForTopology(existing_parent, group_set->groupSetId());
            }
        }
    }
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

void BlockTreeEvictor::onTopologyChanged(TreeNode* parent) {
    for (auto& group_set : tree_->groupSets()) {
        if (group_set->groupType() == CacheGroupType::FULL) {
            updateFullCandidateForTopology(parent, group_set->groupSetId());
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

// ---- Eviction selection ----
bool BlockTreeEvictor::evictLocked(size_t group_set_id, Tier source_tier, bool force_drop) {
    std::optional<TransferDescriptor> eviction_desc = chooseVictim(group_set_id, source_tier, force_drop);
    if (!eviction_desc.has_value()) {
        return false;
    }

    auto task = prepareEvictionLocked(*eviction_desc);
    if (!task.has_value()) {
        return false;
    }
    if (!task->needsCopy()) {
        metrics_reporter_->reportEvictionTriggered(
            source_tier, tree_->groupSets()[group_set_id]->groupType(), force_drop);
        return true;
    }

    auto task_ptr = std::make_shared<EvictionTask>(std::move(*task));
    if (!task_pool_->acquireBusinessCredit()) {
        abortEvictionLocked(*task_ptr);
        return false;
    }
    task_ptr->business_credit_acquired = true;
    updatePendingReleases(*task_ptr, true);
    const bool submitted = task_pool_->submit([this, task_ptr]() { runEvictionTask(task_ptr); });
    if (!submitted) {
        task_pool_->releaseBusinessCredit();
        updatePendingReleases(*task_ptr, false);
        abortEvictionLocked(*task_ptr);
        return false;
    }
    metrics_reporter_->reportEvictionTriggered(source_tier, tree_->groupSets()[group_set_id]->groupType(), force_drop);
    return true;
}

void BlockTreeEvictor::runEvictionTask(std::shared_ptr<const EvictionTask> task) noexcept {
    if (task == nullptr) {
        return;
    }
    EvictionTaskResult task_result;
    task_result.primary_success = false;
    task_result.cascade_success.assign(task->cascade_descs.size(), false);
    try {
        task_runner_->runTransfer(task, *metrics_reporter_, [this, task](EvictionTaskResult completed) {
            scheduleEvictionSettlement(task, std::move(completed));
        });
        return;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
    scheduleEvictionSettlement(std::move(task), std::move(task_result));
}

void BlockTreeEvictor::scheduleEvictionSettlement(std::shared_ptr<const EvictionTask> task,
                                                  EvictionTaskResult                  task_result) noexcept {
    auto settle = [this, task = std::move(task), task_result = std::move(task_result)]() noexcept {
        block_tree_cache_detail::ScopeRollback credit_guard([this, &task]() {
            if (task->business_credit_acquired) {
                task_pool_->releaseBusinessCredit();
            }
        });
        {
            std::lock_guard<std::mutex> lock(*mutex_);
            finalizeEvictionLocked(*task, task_result);
        }
        metrics_reporter_->reportEvictionFinished(*task, task_result, tree_->groupSets());
    };

    bool submitted = false;
    try {
        submitted = task_pool_ != nullptr && task_pool_->submitCompletion(settle);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to enqueue eviction settlement with unknown exception");
    }
    if (!submitted) {
        settle();
    }
}

void BlockTreeEvictor::finalizeEvictionLocked(const EvictionTask&       task,
                                              const EvictionTaskResult& task_result) noexcept {
    const bool has_pending_releases = task.needsCopy();
    settleEvictionLocked(task, task_result);
    if (has_pending_releases) {
        updatePendingReleases(task, false);
    }
    const bool task_succeeded = task_result.primary_success
                                && std::all_of(task_result.cascade_success.begin(),
                                               task_result.cascade_success.end(),
                                               [](bool success) { return success; });
    settled_(task_result.primary_success, has_pending_releases && task_succeeded);
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
    auto                        update_block = [this, reserve](IBlockPool* pool, BlockIdxType block) {
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
void BlockTreeEvictor::collectFullPrune(const TransferDescriptor&                  eviction_desc,
                                        EvictionTask&                              task,
                                        std::vector<std::pair<TreeNode*, size_t>>& detached_resources) const {
    if (task.hasFullPrune() || eviction_desc.target_tier != Tier::NONE
        || tree_->groupSets()[eviction_desc.group_set_id]->groupType() != CacheGroupType::FULL) {
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

std::optional<EvictionTask> BlockTreeEvictor::prepareEvictionLocked(TransferDescriptor eviction_desc) {
    EvictionTask task;
    task.primary_timing =
        EvictionTimingSnapshot(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    std::vector<std::pair<TreeNode*, size_t>> detached_resources;
    collectFullPrune(eviction_desc, task, detached_resources);
    task.primary_desc = std::move(eviction_desc);
    selectCascades(task, detached_resources);
    task.dependent_prune_timings.resize(task.dependent_prune_descs.size());
    if (!allocateTargets(task)) {
        return std::nullopt;
    }

    activateTaskLocked(task, detached_resources);
    if (!task.needsCopy()) {
        EvictionTaskResult task_result;
        task_result.primary_success = true;
        task_result.cascade_success.assign(task.cascade_descs.size(), true);
        finalizeEvictionLocked(task, task_result);
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
    RTP_LLM_CHECK_WITH_INFO(
        resource.transfer_state == GroupSetTransferState::DEMOTING,
        "eviction completion state mismatch: group_set=%zu node_key=%ld state=%d source=%s target=%s",
        desc.group_set_id,
        desc.node->cache_key,
        static_cast<int>(resource.transfer_state),
        tierName(desc.source_tier),
        tierName(desc.target_tier));

    if (desc.target_tier != Tier::NONE) {
        MultiNodeResource target_holder{desc.group_set_id, desc.target_tier, {{desc.node, desc.target_blocks}}};
        resource.setBlocks(desc.target_tier, desc.target_blocks);
        group_set->referenceBlocks(target_holder, BlockTreeRefType::CACHE);
        group_set->unreferenceBlocks(target_holder, BlockTreeRefType::EVICTION);
    }

    // DEMOTING is the operation's ownership token. Release its saved source
    // cache hold before clearing the corresponding resource tier. The target is
    // installed while the state is still non-IDLE, then IDLE is published last.
    const MultiNodeResource source_holder{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
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
        admitCandidate(desc.node, desc.group_set_id, desc.target_tier);
    }
}

void BlockTreeEvictor::abortEvictionLocked(const EvictionTask& task) {
    bool detached = task.primary_desc.node->group_set_resources[task.primary_desc.group_set_id].transfer_detached;
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        detached = detached || cascade_desc.node->group_set_resources[cascade_desc.group_set_id].transfer_detached;
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

bool BlockTreeEvictor::allocateTargets(EvictionTask& task) {
    for (size_t i = 0; i < task.dependent_prune_descs.size(); ++i) {
        const TransferDescriptor& desc = task.dependent_prune_descs[i];
        task.dependent_prune_timings[i] =
            EvictionTimingSnapshot(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
    }

    if (task.primary_desc.target_tier == Tier::NONE) {
        task.cascade_timings.reserve(task.cascade_descs.size());
        for (const TransferDescriptor& desc : task.cascade_descs) {
            task.cascade_timings.emplace_back(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
        }
        return true;
    }

    const auto allocate_target = [this](TransferDescriptor& desc) {
        BlockIdxType target =
            tree_->groupSets()[desc.group_set_id]->allocateSingleBlock(desc.target_tier, BlockTreeRefType::EVICTION);
        if (isNullBlockIdx(target)) {
            return false;
        }
        desc.target_blocks = {target};
        return true;
    };

    if (!allocate_target(task.primary_desc)) {
        return false;
    }

    std::vector<TransferDescriptor>     retained_descs;
    std::vector<EvictionTimingSnapshot> retained_timings;
    retained_descs.reserve(task.cascade_descs.size());
    retained_timings.reserve(task.cascade_descs.size());
    const auto retain = [&](TransferDescriptor& desc) {
        retained_timings.emplace_back(desc.node->group_set_resources[desc.group_set_id].candidate_meta);
        retained_descs.push_back(std::move(desc));
    };

    // A task with a real target only contains same-node cascades; each target
    // allocation is independent, so retain the descriptors that succeed.
    for (TransferDescriptor& desc : task.cascade_descs) {
        if (allocate_target(desc)) {
            retain(desc);
        }
    }

    task.cascade_descs   = std::move(retained_descs);
    task.cascade_timings = std::move(retained_timings);
    return true;
}

void BlockTreeEvictor::activateTaskLocked(const EvictionTask&                              task,
                                          const std::vector<std::pair<TreeNode*, size_t>>& detached_resources) {
    reserveSource(task.primary_desc);
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        reserveSource(cascade_desc);
    }
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        reserveSource(dependent_desc);
    }
    for (const auto& [node, group_set_id] : detached_resources) {
        node->group_set_resources[group_set_id].transfer_detached = true;
    }
    if (task.hasFullPrune()) {
        RTP_LLM_LOG_WARNING("event=block_tree_full_prune root_key=%ld trigger_group_set_id=%zu source_tier=%s "
                            "closure_nodes=%zu dependent_resources=%zu detached_resources=%zu",
                            task.primary_desc.node->cache_key,
                            task.primary_desc.group_set_id,
                            tierName(task.primary_desc.source_tier),
                            task.full_prune_nodes_bottom_up.size(),
                            task.dependent_prune_descs.size(),
                            detached_resources.size());
    }
}

// Reserve the source: remove it from its source heap and mark the in-flight state so
// no other selector can pick it. The caller must first verify that the resource
// is IDLE; assigning DEMOTING is not safe for an already reserved resource.
void BlockTreeEvictor::reserveSource(const TransferDescriptor& eviction_desc) {
    eviction_desc.node->group_set_resources[eviction_desc.group_set_id].transfer_state =
        GroupSetTransferState::DEMOTING;
    suspendCandidate(eviction_desc.node, eviction_desc.group_set_id, eviction_desc.source_tier);
}

// Restore a reserved source after a failed/aborted move: clear the in-flight
// state and re-evaluate candidacy at the source tier.
void BlockTreeEvictor::restoreSource(const TransferDescriptor& eviction_desc) {
    auto              group_set_id = eviction_desc.group_set_id;
    GroupSetResource& resource     = eviction_desc.node->group_set_resources[group_set_id];
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
    RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue(),
                            "eviction rollback produced invalid steady state: group_set_id=%zu node_key=%ld",
                            eviction_desc.group_set_id,
                            eviction_desc.node->cache_key);
    admitCandidate(eviction_desc.node, group_set_id, eviction_desc.source_tier);
}

void BlockTreeEvictor::discardDetachedTransfer(const TransferDescriptor& transfer_desc) {
    GroupSetResource&       resource  = transfer_desc.node->group_set_resources[transfer_desc.group_set_id];
    const GroupSetPtr&      group_set = tree_->groupSets()[transfer_desc.group_set_id];
    const MultiNodeResource source_holder{
        transfer_desc.group_set_id, transfer_desc.source_tier, {{transfer_desc.node, transfer_desc.source_blocks}}};
    group_set->unreferenceBlocks(source_holder, BlockTreeRefType::CACHE);
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
        group_set->releaseSingleBlock(eviction_desc.target_tier, block, BlockTreeRefType::EVICTION);
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

void BlockTreeEvictor::selectCascades(EvictionTask&                              task,
                                      std::vector<std::pair<TreeNode*, size_t>>& detached_resources) {
    const TransferDescriptor& primary_desc     = task.primary_desc;
    TreeNode* const           primary_node     = primary_desc.node;
    const size_t              primary_group_id = primary_desc.group_set_id;
    const Tier                source_tier      = primary_desc.source_tier;
    const Tier                target_tier      = primary_desc.target_tier;
    const auto&               group_sets       = tree_->groupSets();
    std::vector<bool>         selected(group_sets.size(), false);
    selected[primary_group_id] = true;

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
        selected[group_set_id] = true;
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

    if (target_tier == Tier::NONE) {
        for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
            const GroupSetResource& resource = primary_node->group_set_resources[group_set_id];
            if (!resource.is_empty() && (resource.getTopTier() != source_tier || !selected[group_set_id])) {
                return;
            }
        }
        if (!detached_resources.empty() || (!task.hasFullPrune() && !primary_node->children.empty())) {
            return;
        }
        selectUpwardCascades(task);
    }
}

void BlockTreeEvictor::selectUpwardCascades(EvictionTask& task) {
    const TransferDescriptor& primary_desc = task.primary_desc;
    TreeNode* const           primary_node = primary_desc.node;
    const Tier                target_tier  = primary_desc.target_tier;
    const auto&               group_sets   = tree_->groupSets();

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
