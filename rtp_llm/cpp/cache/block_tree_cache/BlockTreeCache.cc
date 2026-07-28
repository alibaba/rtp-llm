#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
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

    std::shared_ptr<const CacheTopology>      topology;
    std::unordered_map<size_t, GroupLocation> reusable_group_locations;
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
                     .emplace(group_ids[local_group_index], GroupLocation{group_set_id, local_group_index})
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
    BlockTreeMatchResult result;
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("empty cache_keys, returning empty result");
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeFindResult         tree_find_result = tree_->findNode(cache_keys);
    if (tree_find_result.matched_node == nullptr) {
        RTP_LLM_LOG_DEBUG("no match found for %zu cache_keys", cache_keys.size());
        return result;
    }

    for (TreeNode* path_node : tree_find_result.path) {
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& resource = path_node->group_set_resources[group_set_id];
            RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE)
                                        || group_sets_[group_set_id]->hasCompleteDeviceValue(resource),
                                    "BlockTreeCache partial DEVICE resource: node_key=%ld group_set_id=%zu "
                                    "device_width=%zu expected_width=%zu",
                                    path_node->cache_key,
                                    group_set_id,
                                    resource.device_blocks.size(),
                                    group_sets_[group_set_id]->devicePoolCount());
        }
    }

    size_t            valid_matched_block_count = 0;
    std::vector<bool> candidate_logically_valid;
    candidate_logically_valid.reserve(tree_find_result.path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(group_sets_.size());
    for (const GroupSetPtr& group_set : group_sets_) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    for (size_t i = 0; i < tree_find_result.path.size(); ++i) {
        TreeNode* path_node        = tree_find_result.path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            const bool        group_valid = match_validators[group_set_id]->validate(path_node, group_set_resource);
            if (!group_valid) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_matched_block_count = i + 1;
        }
        candidate_logically_valid.push_back(all_groups_valid);
    }

    std::vector<TreeNode*> matched_path(tree_find_result.path.begin(),
                                        tree_find_result.path.begin()
                                            + static_cast<ptrdiff_t>(valid_matched_block_count));
    candidate_logically_valid.resize(valid_matched_block_count);
    prepareMatchedBlocks(matched_path, candidate_logically_valid, result);
    if (config_.enable_load) {
        BlockTreeLoadResult load_result = loader_.prepareLoadLocked(matched_path, result.matched_blocks);
        result.load_blocks              = load_result.load_blocks;
        result.host_load_blocks         = load_result.host_load_blocks;
        result.disk_load_blocks         = load_result.disk_load_blocks;
        result.load_ticket              = std::move(load_result.load_ticket);
    }

    RTP_LLM_LOG_DEBUG("matched %zu blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_blocks,
                      cache_keys.size(),
                      tree_->nodeCount());
    return result;
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
    const GroupLocation* location = nullptr;
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

void BlockTreeCache::validateMatchedResource(const MultiNodeResource& resource) const {
    RTP_LLM_CHECK_WITH_INFO(resource.group_set_id < group_sets_.size(),
                            "invalid matched group_set_id=%zu group_set_count=%zu",
                            resource.group_set_id,
                            group_sets_.size());
    RTP_LLM_CHECK_WITH_INFO(resource.tier == Tier::DEVICE,
                            "matched resource requires DEVICE tier, group_set_id=%zu tier=%s",
                            resource.group_set_id,
                            tierName(resource.tier));

    const GroupSetPtr& group_set = group_sets_[resource.group_set_id];
    for (const auto& node_blocks : resource.per_node) {
        RTP_LLM_CHECK_WITH_INFO(node_blocks.size() == group_set->devicePoolCount()
                                    && std::all_of(node_blocks.begin(),
                                                   node_blocks.end(),
                                                   [](BlockIdxType block) { return !isNullBlockIdx(block); }),
                                "malformed matched DEVICE blocks, group_set_id=%zu expected_width=%zu actual_width=%zu",
                                resource.group_set_id,
                                group_set->devicePoolCount(),
                                node_blocks.size());
    }
    RTP_LLM_CHECK_WITH_INFO(resource.tree_nodes.empty()
                                || (resource.tree_nodes.size() == resource.per_node.size()
                                    && std::all_of(resource.tree_nodes.begin(),
                                                   resource.tree_nodes.end(),
                                                   [](const TreeNode* node) { return node != nullptr; })),
                            "malformed matched tree-node alignment, group_set_id=%zu nodes=%zu blocks=%zu",
                            resource.group_set_id,
                            resource.tree_nodes.size(),
                            resource.per_node.size());
}

void BlockTreeCache::releaseMatchedResources(const std::vector<MultiNodeResource>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<size_t>  seen_group_set_ids;
    for (const auto& resource : resources) {
        RTP_LLM_CHECK_WITH_INFO(seen_group_set_ids.emplace(resource.group_set_id).second,
                                "releaseMatchedResources duplicate group_set_id=%zu",
                                resource.group_set_id);
        validateMatchedResource(resource);
    }
    for (const auto& resource : resources) {
        group_sets_[resource.group_set_id]->unreferenceBlocks(resource, BlockRefType::REQUEST);
        // Releasing a match reference may make the node evictable again.
        evictor_.refreshCandidatesAfterRelease(resource);
    }
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(size_t                                group_id,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    const auto location_it = reusable_group_locations_.find(group_id);
    if (location_it == reusable_group_locations_.end()) {
        return {};
    }
    const GroupLocation& location = location_it->second;
    for (const auto& resource : matched_resources) {
        if (resource.group_set_id != location.group_set_id) {
            continue;
        }
        validateMatchedResource(resource);
        BlockIndicesType blocks;
        blocks.reserve(resource.per_node.size());
        for (const auto& node_blocks : resource.per_node) {
            blocks.push_back(node_blocks[location.local_group_index]);
        }
        return blocks;
    }
    return {};
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

void BlockTreeCache::prepareMatchedBlocks(const std::vector<TreeNode*>& matched_path,
                                          const std::vector<bool>&      candidate_logically_valid,
                                          BlockTreeMatchResult&         result) {
    const size_t logical_matched_block_count = matched_path.size();
    if (logical_matched_block_count == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(candidate_logically_valid.size() == logical_matched_block_count,
                            "candidate validity size mismatch, path=%zu valid=%zu",
                            logical_matched_block_count,
                            candidate_logically_valid.size());

    const size_t ready_matched_block_count = computeReadyMatchedBlockCount(matched_path, candidate_logically_valid);
    if (ready_matched_block_count > 0) {
        result.matched_node   = matched_path[ready_matched_block_count - 1];
        result.matched_blocks = ready_matched_block_count;
        evictor_.onMatched(std::vector<TreeNode*>(
            matched_path.begin(), matched_path.begin() + static_cast<ptrdiff_t>(ready_matched_block_count)));
    }

    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        MultiNodeResource  matched_device_blocks{group_set_id, Tier::DEVICE};

        const size_t ready_reuse_count = std::min(
            group_set->computeReuseBlockCount(ready_matched_block_count, matched_path), ready_matched_block_count);
        const size_t ready_reuse_begin = ready_matched_block_count - ready_reuse_count;
        for (size_t i = ready_reuse_begin; i < ready_matched_block_count; ++i) {
            TreeNode*                       path_node          = matched_path[i];
            GroupSetResource&               group_set_resource = path_node->group_set_resources[group_set_id];
            const std::vector<BlockIdxType> device_blocks      = group_set->getBlocks(group_set_resource, Tier::DEVICE);
            matched_device_blocks.per_node.push_back(device_blocks);
            matched_device_blocks.tree_nodes.push_back(path_node);
        }

        if (!matched_device_blocks.per_node.empty()) {
            group_set->referenceBlocks(matched_device_blocks, BlockRefType::REQUEST);
            result.matched_resources.push_back(std::move(matched_device_blocks));
        }
    }
}

size_t BlockTreeCache::computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                                     const std::vector<bool>&      candidate_logically_valid) const {
    for (size_t candidate_count = matched_path.size(); candidate_count > 0; --candidate_count) {
        if (!candidate_logically_valid[candidate_count - 1]) {
            continue;
        }
        bool all_groups_ready = true;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetPtr& group_set = group_sets_[group_set_id];
            const size_t       reuse_count =
                std::min(group_set->computeReuseBlockCount(candidate_count, matched_path), candidate_count);
            for (size_t path_index = candidate_count - reuse_count; path_index < candidate_count; ++path_index) {
                TreeNode* path_node = matched_path[path_index];
                if (!group_set->hasCompleteDeviceValue(path_node->group_set_resources[group_set_id])) {
                    all_groups_ready = false;
                    break;
                }
            }
            if (!all_groups_ready) {
                break;
            }
        }
        if (all_groups_ready) {
            return candidate_count;
        }
    }
    return 0;
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
