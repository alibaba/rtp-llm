#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>               tree,
                               std::vector<GroupSetPtr>                 group_sets,
                               BlockTreeCacheConfig                     config,
                               std::shared_ptr<StorageBackend>          storage_backend,
                               std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>       task_pool):
    config_(std::move(config)),
    tree_(std::move(tree)),
    group_sets_(std::move(group_sets)),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    evictor_(
        group_sets_,
        [this](const TransferDescriptor& descriptor) { return executeTransfer(descriptor); },
        config_.enable_reverse_eviction,
        tree_.get(),
        transfer_dispatcher_.get(),
        task_pool_.get(),
        metrics_reporter_,
        mutex_,
        config_.memory_cache_sync_timeout_ms,
        config_.memory_cache_disk_sync_timeout_ms,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](const std::vector<EvictionReleaseCredit>& credits) {
            reserveInFlightDeviceReleaseCreditsLocked(credits);
        },
        [this](const std::vector<EvictionReleaseCredit>& credits) {
            settleInFlightDeviceReleaseCreditsLocked(credits);
        },
        [this](bool tree_data_mutated, bool check_watermark) {
            if (tree_data_mutated) {
                ++mutation_version_;
            }
            if (check_watermark) {
                checkWatermark();
            }
        },
        [this](CacheKeyType cache_key, size_t group_set_id) {
            if (config_.enable_remote_cache) {
                evictor_.writeRemoteThrough(storage_backend_, cache_key, group_set_id);
            }
        }),
    matcher_(tree_.get(), group_sets_, reusable_group_locations_, evictor_),
    loader_(
        group_sets_,
        evictor_,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        metrics_reporter_,
        mutex_,
        config_.memory_cache_disk_sync_timeout_ms,
        config_.memory_cache_sync_timeout_ms,
        config_.enable_device_cache,
        [this](bool tree_data_mutated, bool check_watermark) {
            if (tree_data_mutated) {
                ++mutation_version_;
            }
            if (check_watermark) {
                checkWatermark();
            }
        }) {
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const auto& group_ids = group_sets_[group_set_id]->groupIds();
        for (size_t member_index = 0; member_index < group_ids.size(); ++member_index) {
            reusable_group_locations_.emplace(group_ids[member_index],
                                              ReusableGroupLocation{group_set_id, member_index});
        }
    }
}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    evictor_.init(config_.device_eviction_policy, config_.host_eviction_policy, config_.disk_eviction_policy);
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.eviction_thread_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     group_sets_.size(),
                     reusable_group_locations_.size(),
                     config_.eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_memory_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : group_sets_) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    initialized_ = true;
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    loader_.shutdown();
    if (!initialized_) {
        RTP_LLM_LOG_INFO("destroyed");
        return;
    }
    RTP_LLM_LOG_INFO("load tickets closed, waiting for pending tasks...");
    waitForPendingTasks();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RTP_LLM_CHECK_WITH_INFO(
            in_flight_device_release_credits_.empty(),
            "BlockTreeCache: in-flight DEVICE release credits remain after pending tasks drained: %zu",
            in_flight_device_release_credits_.size());
    }
    task_pool_->shutdown();
    drainTreeHolds();
    RTP_LLM_LOG_INFO("destroyed");
}

void BlockTreeCache::drainTreeHolds() {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK_WITH_INFO(tree_ != nullptr && tree_->root() != nullptr,
                            "BlockTreeCache::drainTreeHolds: tree and root must be valid");

    const auto drain_node = [this](TreeNode* node) {
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetPtr& group_set = group_sets_[group_set_id];

            GroupSetResource&               resource      = node->group_set_resources[group_set_id];
            const std::vector<BlockIdxType> device_blocks = resource.device_blocks;
            if (!device_blocks.empty()) {
                // Keep shutdown symmetric with referenceBlocks/unreferenceBlocks:
                // pool-less structural resources carry no hold, while real pools are released exactly once.
                group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::DEVICE, {device_blocks}},
                                             BlockRefType::BLOCK_CACHE);
                std::fill(resource.device_blocks.begin(), resource.device_blocks.end(), NULL_BLOCK_IDX);
            }

            if (!isNullBlockIdx(resource.host_block)) {
                const BlockIdxType host_block = resource.host_block;
                group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::HOST, {{host_block}}},
                                             BlockRefType::BLOCK_CACHE);
                resource.host_block = NULL_BLOCK_IDX;
            }

            if (!isNullBlockIdx(resource.disk_slot)) {
                const BlockIdxType disk_block = resource.disk_slot;
                group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::DISK, {{disk_block}}},
                                             BlockRefType::BLOCK_CACHE);
                resource.disk_slot = NULL_BLOCK_IDX;
            }

            resource.transfer_state = GroupSetTransferState::IDLE;
        }
    };

    drain_node(tree_->root());
    for (const std::unique_ptr<TreeNode>& node : tree_->nodes()) {
        drain_node(node.get());
    }
}

bool BlockTreeCache::executeTransfer(const TransferDescriptor& descriptor) {
    return transfer_dispatcher_->executePerRank(descriptor);
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType& cache_keys) {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("empty cache_keys, returning empty result");
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto [result, logical_matched_path] = matcher_.matchLocked(cache_keys);
    if (config_.enable_load) {
        BlockTreeLoadResult load_result = loader_.prepareLoadLocked(logical_matched_path, result.matched_blocks);
        result.load_blocks              = load_result.load_blocks;
        result.host_load_blocks         = load_result.host_load_blocks;
        result.disk_load_blocks         = load_result.disk_load_blocks;
        result.load_ticket              = std::move(load_result.load_ticket);
    }
    return std::move(result);
}

void BlockTreeCache::insert(TreeNode*                                         parent,
                            const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources) {
    insertImpl(parent, cache_keys, resources, false);
}

void BlockTreeCache::insertSparse(TreeNode*                                         parent,
                                  const CacheKeysType&                              cache_keys,
                                  const std::vector<std::vector<GroupSetResource>>& resources) {
    insertImpl(parent, cache_keys, resources, true);
}

void BlockTreeCache::insertImpl(TreeNode*                                         parent,
                                const CacheKeysType&                              cache_keys,
                                const std::vector<std::vector<GroupSetResource>>& resources,
                                bool                                              allow_sparse_resources) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    if (resources.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/resource size mismatch, keys=%zu resources=%zu", cache_keys.size(), resources.size());
        return;
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        if (resources[i].size() != group_sets_.size()) {
            RTP_LLM_LOG_WARNING("GroupSetResource mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                group_sets_.size(),
                                resources[i].size());
            return;
        }
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const auto& group_set           = group_sets_[group_set_id];
            const auto& resource            = resources[i][group_set_id];
            const bool  structurally_absent = resource.device_blocks.empty() && resource.is_empty();
            const bool  allowed_sparse_absence =
                allow_sparse_resources && group_set->groupType() != CacheGroupType::FULL && structurally_absent;
            RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState()
                                        && (allowed_sparse_absence || group_set->hasCompleteDeviceValue(resource)),
                                    "BlockTreeCache insert requires an IDLE complete DEVICE resource: "
                                    "key=%ld group_set_id=%zu state=%d tiers=%zu expected_width=%zu actual_width=%zu",
                                    cache_keys[i],
                                    group_set_id,
                                    static_cast<int>(resource.transfer_state),
                                    resource.servingTierCount(),
                                    group_set->devicePoolCount(),
                                    resource.device_blocks.size());
        }
    }

    BlockTreeInsertResult insert_result = tree_->insertNode(parent, cache_keys, resources);

    // incRef cache-hold on new nodes' device blocks (balanced by unreferenceBlocks on
    // eviction). Reused nodes keep theirs; their demoted data comes from load.
    for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
        TreeNode* node = inserted.node;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetPtr& group_set = group_sets_[group_set_id];
            GroupSetResource&  resource  = node->group_set_resources[group_set_id];
            if (group_set->hasCompleteDeviceValue(resource)) {
                const std::vector<BlockIdxType> blocks = group_set->getBlocks(resource, Tier::DEVICE);
                group_set->referenceBlocks(MultiNodeResource{group_set_id, Tier::DEVICE, {blocks}},
                                           BlockRefType::BLOCK_CACHE);
            }
        }
    }

    // Existing nodes may independently refill one empty GroupSetResource. Take a tree
    // holder only for that adopted resource; other resources already own theirs.
    for (const BlockTreeAdoptedResource& adopted : insert_result.adopted_resources) {
        const size_t       group_set_id = adopted.group_set_id;
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        GroupSetResource&  resource  = adopted.node->group_set_resources[group_set_id];
        RTP_LLM_CHECK_WITH_INFO(group_set->hasCompleteDeviceValue(resource),
                                "BlockTreeCache adopted incomplete DEVICE resource: key=%ld group_set_id=%zu",
                                adopted.node->cache_key,
                                group_set_id);
        group_set->referenceBlocks(
            MultiNodeResource{group_set_id, Tier::DEVICE, {group_set->getBlocks(resource, Tier::DEVICE)}},
            BlockRefType::BLOCK_CACHE);
    }

    const bool changed = !insert_result.inserted_nodes.empty() || !insert_result.adopted_resources.empty();
    if (!changed) {
        return;
    }

    // Stamp and refresh only newly created nodes and exact adopted GroupSet resources.
    evictor_.onInsertCommitted(insert_result);
    ++mutation_version_;
    RTP_LLM_LOG_DEBUG("created=%zu adopted=%zu tree_nodes=%zu",
                      insert_result.inserted_nodes.size(),
                      insert_result.adopted_resources.size(),
                      tree_->nodeCount());
    checkWatermark();
}

int BlockTreeCache::evictForGroup(size_t group_id, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const auto location_it = reusable_group_locations_.find(group_id);
    if (location_it == reusable_group_locations_.end()) {
        return 0;
    }
    const ReusableGroupLocation& location    = location_it->second;
    const GroupSetPtr&           group_set   = group_sets_[location.group_set_id];
    const auto&                  device_pool = group_set->devicePools()[location.member_index];

    const size_t initial_free = device_pool->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        auto eviction_move = evictor_.chooseVictim(location.group_set_id, Tier::DEVICE);
        if (!eviction_move.has_value()) {
            break;
        }
        eviction_move->target_tier = Tier::NONE;
        if (!evictor_.submitLocked(*eviction_move)) {
            break;
        }
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    RTP_LLM_LOG_DEBUG("group_id=%zu group_set[%zu] reclaimed %zu/%zu device blocks",
                      group_id,
                      location.group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

void BlockTreeCache::releaseMatchedResources(const std::vector<MultiNodeResource>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);
    matcher_.releaseMatchedResourcesLocked(resources);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(size_t                                group_id,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return matcher_.matchedBlocksForGroup(group_id, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->nodeCount();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<BlockTreePoolMetricsSnapshot> BlockTreeCache::poolMetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_reporter_.collectPoolMetricsSnapshots(group_sets_, evictor_);
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_.collectEvictableMetricsSnapshots(group_sets_, evictor_);
    }
    metrics_reporter_.reportEvictableBlockCount(snapshots);
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;
    if (limit == 0 || !tree_ || !tree_->root()) {
        return snapshot;
    }

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->nodeCount());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty() && snapshot.keys.size() < limit) {
        const TreeNode* node = pending.back();
        pending.pop_back();
        const bool reusable = std::any_of(node->group_set_resources.begin(),
                                          node->group_set_resources.end(),
                                          [](const GroupSetResource& resource) { return !resource.is_empty(); });
        if (reusable) {
            snapshot.keys.push_back(node->cache_key);
        }
        for (const auto& [cache_key, child] : node->children) {
            (void)cache_key;
            if (child) {
                pending.push_back(child);
            }
        }
    }
    return snapshot;
}

void BlockTreeCache::waitForPendingTasks() {
    task_pool_->waitForIdle();
}

void BlockTreeCache::onBlocksReleased() {
    std::lock_guard<std::mutex> lock(mutex_);
    // After external refcount changes (e.g. request free), blocks that were
    // non-evictable at insert time (refcount > 1) may now have refcount == 1
    // and thus become eviction candidates.  Refresh the eviction heap before
    // checking watermark so that pending evictions can find victims.
    evictor_.refreshAllCandidates(*tree_);
    checkWatermark();
}

bool BlockTreeCache::cancelLoad(const std::shared_ptr<AsyncContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    return loader_.cancelLoadLocked(context);
}

void BlockTreeCache::reserveInFlightDeviceReleaseCreditsLocked(
    const std::vector<EvictionReleaseCredit>& release_credits) {
    for (const EvictionReleaseCredit& credit : release_credits) {
        if (credit.pool != nullptr) {
            ++in_flight_device_release_credits_[credit.pool];
        }
    }
}

void BlockTreeCache::settleInFlightDeviceReleaseCreditsLocked(
    const std::vector<EvictionReleaseCredit>& release_credits) noexcept {
    for (const EvictionReleaseCredit& credit : release_credits) {
        const auto it = in_flight_device_release_credits_.find(credit.pool);
        RTP_LLM_CHECK_WITH_INFO(it != in_flight_device_release_credits_.end() && it->second > 0,
                                "missing in-flight DEVICE release credit while settling pool=%p block=%d",
                                static_cast<void*>(credit.pool.get()),
                                credit.block);
        if (--it->second == 0) {
            in_flight_device_release_credits_.erase(it);
        }
    }
}
void BlockTreeCache::checkWatermark() {
    if (config_.enable_device_cache && config_.device_min_free_blocks > 0) {
        struct PoolDeficit {
            DeviceBlockPoolPtr pool;
            size_t             deficit{0};
            size_t             accepted_credits{0};
        };
        std::vector<PoolDeficit>                     pool_deficits;
        std::unordered_map<DeviceBlockPool*, size_t> pool_indices;
        for (const auto& group_set : group_sets_) {
            for (const auto& pool : group_set->devicePools()) {
                if (pool_indices.count(pool.get()) != 0) {
                    continue;
                }
                const size_t capacity     = pool->totalBlocksNum();
                const size_t min_free     = std::min(config_.device_min_free_blocks, capacity);
                const size_t free_blocks  = pool->freeBlocksNum();
                const size_t deficit      = free_blocks < min_free ? min_free - free_blocks : 0;
                const auto   in_flight_it = in_flight_device_release_credits_.find(pool);
                const size_t in_flight_credits =
                    in_flight_it == in_flight_device_release_credits_.end() ? 0 : in_flight_it->second;
                pool_indices.emplace(pool.get(), pool_deficits.size());
                pool_deficits.push_back({pool, deficit, in_flight_credits});
            }
        }

        auto has_uncovered_deficit = [&]() {
            return std::any_of(pool_deficits.begin(), pool_deficits.end(), [](const PoolDeficit& state) {
                return state.accepted_credits < state.deficit;
            });
        };
        auto group_set_has_uncovered_deficit = [&](const GroupSetPtr& group_set) {
            for (const auto& pool : group_set->devicePools()) {
                const auto it = pool_indices.find(pool.get());
                if (it != pool_indices.end()) {
                    const auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        return true;
                    }
                }
            }
            return false;
        };

        std::vector<bool> unavailable(group_sets_.size(), false);
        while (has_uncovered_deficit()) {
            bool round_progress = false;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                const auto& group_set = group_sets_[group_set_id];
                if (unavailable[group_set_id] || !group_set_has_uncovered_deficit(group_set)) {
                    continue;
                }
                auto eviction_move = evictor_.chooseVictim(group_set_id, Tier::DEVICE);
                if (!eviction_move.has_value()) {
                    unavailable[group_set_id] = true;
                    continue;
                }
                std::vector<EvictionReleaseCredit> release_credits;
                if (!evictor_.submitLocked(*eviction_move, &release_credits)) {
                    unavailable[group_set_id] = true;
                    continue;
                }
                bool credited_uncovered_pool = false;
                for (const EvictionReleaseCredit& credit : release_credits) {
                    const auto it = pool_indices.find(credit.pool.get());
                    if (it == pool_indices.end()) {
                        continue;
                    }
                    auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        ++state.accepted_credits;
                        credited_uncovered_pool = true;
                    }
                }
                if (credited_uncovered_pool) {
                    round_progress = true;
                } else {
                    unavailable[group_set_id] = true;
                }
            }
            if (!round_progress) {
                break;
            }
        }
    }

    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (tier == Tier::DEVICE && config_.device_min_free_blocks > 0) {
            continue;
        }
        auto wm = config_.watermarkForTier(tier);
        if (wm.ratio <= 0.0 || !config_.isTierEnabled(tier))
            continue;

        for (auto& group_set : group_sets_) {
            auto victims = evictor_.chooseWatermarkVictims(*group_set, tier, wm.ratio);
            for (auto& eviction_move : victims) {
                evictor_.submitLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
