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
        [this](size_t group_set_id, Tier tier) { return reclaimOneForGroup(group_set_id, tier); },
        [this](bool tree_data_mutated, bool check_watermark) {
            if (tree_data_mutated) {
                ++mutation_version_;
            }
            if (check_watermark) {
                checkWatermark();
            }
        }) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    if (transfer_dispatcher_ == nullptr || task_pool_ == nullptr) {
        RTP_LLM_LOG_ERROR("transfer dispatcher and task pool must be initialized");
        return false;
    }
    if (!initializeConfiguration()) {
        RTP_LLM_LOG_ERROR("invalid configuration");
        return false;
    }
    if (!evictor_.init(config_.device_eviction_policy, config_.host_eviction_policy, config_.disk_eviction_policy)) {
        RTP_LLM_LOG_ERROR("failed to initialize BlockTreeEvictor");
        return false;
    }
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
        RTP_LLM_LOG_INFO("  group[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    initialized_ = true;
    return true;
}

bool BlockTreeCache::initializeConfiguration() {
    if (tree_ == nullptr) {
        RTP_LLM_LOG_ERROR("tree must be initialized");
        return false;
    }
    if (tree_->groupSetResourceCount() != group_sets_.size()) {
        RTP_LLM_LOG_ERROR(
            "tree/group set count mismatch: tree=%zu registry=%zu", tree_->groupSetResourceCount(), group_sets_.size());
        return false;
    }
    if (config_.enable_disk_cache && !config_.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("disk cache requires memory cache");
        return false;
    }
    if (config_.enable_load && !config_.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("load requires memory cache");
        return false;
    }

    std::shared_ptr<const CacheTopology> topology;
    ReusableGroupLocations                reusable_group_locations;
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        if (group_set == nullptr || group_set->topology() == nullptr || group_set->groupSetId() != group_set_id) {
            RTP_LLM_LOG_ERROR("group set must be initialized and indexed by id, index=%zu", group_set_id);
            return false;
        }
        if (topology == nullptr) {
            topology = group_set->topology();
        } else if (group_set->topology() != topology) {
            RTP_LLM_LOG_ERROR("all group sets must share one CacheTopology");
            return false;
        }

        const auto host_pool = group_set->hostPool();
        const auto disk_pool = group_set->diskPool();
        if (config_.enable_memory_cache && host_pool == nullptr) {
            RTP_LLM_LOG_ERROR("memory cache group %zu has no host pool", group_set_id);
            return false;
        }
        if (config_.enable_disk_cache && disk_pool == nullptr) {
            RTP_LLM_LOG_ERROR("disk cache group %zu has no disk pool", group_set_id);
            return false;
        }
        if (host_pool != nullptr && host_pool->payloadBytes() != group_set->payloadBytes()) {
            RTP_LLM_LOG_ERROR("group %zu host/logical payload mismatch: %zu/%zu",
                              group_set_id,
                              host_pool->payloadBytes(),
                              group_set->payloadBytes());
            return false;
        }
        if (disk_pool != nullptr && disk_pool->payloadBytes() != group_set->payloadBytes()) {
            RTP_LLM_LOG_ERROR("group %zu disk/logical payload mismatch: %zu/%zu",
                              group_set_id,
                              disk_pool->payloadBytes(),
                              group_set->payloadBytes());
            return false;
        }

        const auto& group_ids = group_set->groupIds();
        for (size_t local_group_index = 0; local_group_index < group_ids.size(); ++local_group_index) {
            if (!reusable_group_locations
                     .emplace(group_ids[local_group_index], ReusableGroupLocation{group_set_id, local_group_index})
                     .second) {
                RTP_LLM_LOG_ERROR("duplicate reusable group_id=%zu", group_ids[local_group_index]);
                return false;
            }
        }
    }

    if (topology != nullptr) {
        for (size_t group_id = 0; group_id < topology->groups().size(); ++group_id) {
            if (topology->groupById(group_id).policy.enable_prefix_reuse
                && reusable_group_locations.count(group_id) == 0) {
                RTP_LLM_LOG_ERROR("reusable topology group_id=%zu is missing from GroupSet registry", group_id);
                return false;
            }
        }
    }
    reusable_group_locations_ = std::move(reusable_group_locations);
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
        RTP_LLM_CHECK_WITH_INFO(node != nullptr, "BlockTreeCache::drainTreeHolds: node must be valid");
        RTP_LLM_CHECK_WITH_INFO(node->group_set_resources.size() == group_sets_.size(),
                                "BlockTreeCache::drainTreeHolds: slot count mismatch, slots=%zu groups=%zu",
                                node->group_set_resources.size(),
                                group_sets_.size());

        for (size_t group_set_index = 0; group_set_index < group_sets_.size(); ++group_set_index) {
            const GroupSetPtr& group_set = group_sets_[group_set_index];

            GroupSetResource&               slot          = node->group_set_resources[group_set_index];
            const std::vector<BlockIdxType> device_blocks = slot.device_blocks;
            if (!device_blocks.empty()) {
                // Keep shutdown symmetric with referenceBlocks/unreferenceBlocks:
                // pool-less structural slots carry no hold, while real pools are released exactly once.
                group_set->unreferenceBlocks(MultiNodeResource{group_set_index, Tier::DEVICE, {device_blocks}},
                                             BlockRefType::BLOCK_CACHE);
                std::fill(slot.device_blocks.begin(), slot.device_blocks.end(), NULL_BLOCK_IDX);
            }

            if (!isNullBlockIdx(slot.host_block)) {
                const BlockIdxType host_block = slot.host_block;
                group_set->unreferenceBlocks(MultiNodeResource{group_set_index, Tier::HOST, {{host_block}}},
                                             BlockRefType::BLOCK_CACHE);
                slot.host_block = NULL_BLOCK_IDX;
            }

            if (!isNullBlockIdx(slot.disk_slot)) {
                const BlockIdxType disk_block = slot.disk_slot;
                group_set->unreferenceBlocks(MultiNodeResource{group_set_index, Tier::DISK, {{disk_block}}},
                                             BlockRefType::BLOCK_CACHE);
                slot.disk_slot = NULL_BLOCK_IDX;
            }

            slot.transfer_state = GroupSetTransferState::IDLE;
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
                            const std::vector<std::vector<GroupSetResource>>& slots) {
    insertImpl(parent, cache_keys, slots, false);
}

void BlockTreeCache::insertSparse(TreeNode*                                         parent,
                                  const CacheKeysType&                              cache_keys,
                                  const std::vector<std::vector<GroupSetResource>>& slots) {
    insertImpl(parent, cache_keys, slots, true);
}

void BlockTreeCache::insertImpl(TreeNode*                                         parent,
                                const CacheKeysType&                              cache_keys,
                                const std::vector<std::vector<GroupSetResource>>& slots,
                                bool                                              allow_sparse_slots) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    if (slots.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/slot size mismatch, keys=%zu slots=%zu", cache_keys.size(), slots.size());
        return;
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].size() != group_sets_.size()) {
            RTP_LLM_LOG_WARNING(
                "GroupSetResource mismatch, index=%zu expected=%zu actual=%zu", i, group_sets_.size(), slots[i].size());
            return;
        }
        for (size_t group_set_index = 0; group_set_index < group_sets_.size(); ++group_set_index) {
            const auto& group               = group_sets_[group_set_index];
            const auto& slot                = slots[i][group_set_index];
            const bool  structurally_absent = slot.device_blocks.empty() && slot.is_empty();
            const bool  allowed_sparse_absence =
                allow_sparse_slots && group->groupType() != CacheGroupType::FULL && structurally_absent;
            RTP_LLM_CHECK_WITH_INFO(slot.isValidSteadyState()
                                        && (allowed_sparse_absence || group->hasCompleteDeviceValue(slot)),
                                    "BlockTreeCache insert requires an IDLE complete DEVICE resource: "
                                    "key=%ld group_set_id=%zu state=%d tiers=%zu expected_width=%zu actual_width=%zu",
                                    cache_keys[i],
                                    group_set_index,
                                    static_cast<int>(slot.transfer_state),
                                    slot.servingTierCount(),
                                    group->devicePoolCount(),
                                    slot.device_blocks.size());
        }
    }

    BlockTreeInsertResult insert_result = tree_->insertNode(parent, cache_keys, slots);

    // incRef cache-hold on new nodes' device blocks (balanced by unreferenceBlocks on
    // eviction). Reused nodes keep theirs; their demoted data comes from load.
    for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
        TreeNode* node = inserted.node;
        RTP_LLM_CHECK_WITH_INFO(
            node != nullptr && node->group_set_resources.size() == group_sets_.size(),
            "BlockTreeCache received malformed inserted node: node=%p expected_resources=%zu actual_resources=%zu",
            static_cast<void*>(node),
            group_sets_.size(),
            node == nullptr ? 0 : node->group_set_resources.size());
        for (size_t group_set_index = 0; group_set_index < group_sets_.size(); ++group_set_index) {
            const GroupSetPtr& group = group_sets_[group_set_index];
            GroupSetResource&  slot  = node->group_set_resources[group_set_index];
            if (group->hasCompleteDeviceValue(slot)) {
                const std::vector<BlockIdxType> blocks = group->getBlocks(slot, Tier::DEVICE);
                group->referenceBlocks(MultiNodeResource{group_set_index, Tier::DEVICE, {blocks}},
                                       BlockRefType::BLOCK_CACHE);
            }
        }
    }

    // Existing nodes may independently refill one empty GroupSetResource. Take a tree
    // holder only for that adopted resource; other resources already own theirs.
    for (const BlockTreeAdoptedSlot& adopted : insert_result.adopted_slots) {
        const size_t group_set_index = adopted.group_set_id;
        RTP_LLM_CHECK_WITH_INFO(
            adopted.node != nullptr && group_set_index < group_sets_.size()
                && adopted.node->group_set_resources.size() == group_sets_.size(),
            "BlockTreeCache received malformed adopted resource: node=%p group_set_id=%zu group_set_count=%zu",
            static_cast<void*>(adopted.node),
            group_set_index,
            group_sets_.size());
        const GroupSetPtr& group = group_sets_[group_set_index];
        GroupSetResource&  slot  = adopted.node->group_set_resources[group_set_index];
        RTP_LLM_CHECK_WITH_INFO(group->hasCompleteDeviceValue(slot),
                                "BlockTreeCache adopted incomplete DEVICE resource: key=%ld group_set_id=%zu",
                                adopted.node->cache_key,
                                group_set_index);
        group->referenceBlocks(MultiNodeResource{group_set_index, Tier::DEVICE, {group->getBlocks(slot, Tier::DEVICE)}},
                               BlockRefType::BLOCK_CACHE);
    }

    const bool changed = !insert_result.inserted_nodes.empty() || !insert_result.adopted_slots.empty();
    if (!changed) {
        return;
    }

    // Stamp and refresh only newly created nodes and exact adopted GroupSet resources.
    evictor_.onInsertCommitted(insert_result);
    ++mutation_version_;
    RTP_LLM_LOG_DEBUG("created=%zu adopted=%zu tree_nodes=%zu",
                      insert_result.inserted_nodes.size(),
                      insert_result.adopted_slots.size(),
                      tree_->nodeCount());
    checkWatermark();
}

int BlockTreeCache::evictForTag(const std::string& tag, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE) || tag.empty()) {
        return 0;
    }
    const ReusableGroupLocation* location = nullptr;
    for (const auto& [group_id, candidate] : reusable_group_locations_) {
        const GroupSetPtr& group_set = group_sets_[candidate.group_set_id];
        if (group_set->topology()->groupById(group_id).tag == tag) {
            location = &candidate;
            break;
        }
    }
    if (location == nullptr) {
        return 0;
    }
    const GroupSetPtr& group_set   = group_sets_[location->group_set_id];
    const auto&        device_pool = group_set->devicePools()[location->local_group_index];

    const size_t initial_free = device_pool->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        auto eviction_move = evictor_.chooseVictim(location->group_set_id, Tier::DEVICE);
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
    RTP_LLM_LOG_DEBUG("tag=%s group_set[%zu] reclaimed %zu/%zu device blocks",
                      tag.c_str(),
                      location->group_set_id,
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
                                          [](const GroupSetResource& slot) { return !slot.is_empty(); });
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

bool BlockTreeCache::reclaimOneForGroup(size_t group_set_id, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (group_set_id >= group_sets_.size()) {
        return false;
    }
    auto eviction_move = evictor_.chooseVictim(group_set_id, tier);
    if (!eviction_move.has_value()) {
        return false;
    }
    eviction_move->target_tier = Tier::NONE;
    return evictor_.submitLocked(*eviction_move);
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
        for (const auto& group : group_sets_) {
            for (const auto& pool : group->devicePools()) {
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
        auto group_has_uncovered_deficit = [&](const GroupSetPtr& group) {
            for (const auto& pool : group->devicePools()) {
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
            for (size_t group_index = 0; group_index < group_sets_.size(); ++group_index) {
                const auto& group = group_sets_[group_index];
                if (unavailable[group_index] || !group_has_uncovered_deficit(group)) {
                    continue;
                }
                auto eviction_move = evictor_.chooseVictim(group->groupSetId(), Tier::DEVICE);
                if (!eviction_move.has_value()) {
                    unavailable[group_index] = true;
                    continue;
                }
                std::vector<EvictionReleaseCredit> release_credits;
                if (!evictor_.submitLocked(*eviction_move, &release_credits)) {
                    unavailable[group_index] = true;
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
                    unavailable[group_index] = true;
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

        for (auto& group : group_sets_) {
            auto victims = evictor_.chooseWatermarkVictims(*group, tier, wm.ratio);
            for (auto& eviction_move : victims) {
                evictor_.submitLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
