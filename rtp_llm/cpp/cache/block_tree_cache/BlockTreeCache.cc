#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

template<typename Cleanup>
class ScopeRollback {
public:
    explicit ScopeRollback(Cleanup cleanup): cleanup_(std::move(cleanup)) {}

    ~ScopeRollback() {
        run();
    }

    ScopeRollback(const ScopeRollback&)            = delete;
    ScopeRollback& operator=(const ScopeRollback&) = delete;
    ScopeRollback(ScopeRollback&&)                 = delete;
    ScopeRollback& operator=(ScopeRollback&&)      = delete;

    void run() {
        if (!active_) {
            return;
        }
        active_ = false;
        cleanup_();
    }

    void dismiss() noexcept {
        active_ = false;
    }

private:
    Cleanup cleanup_;
    bool    active_{true};
};

}  // anonymous namespace

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>               tree,
                               std::vector<GroupSetPtr>                 group_sets,
                               BlockTreeCacheConfig                     config,
                               std::shared_ptr<StorageBackend>          storage_backend,
                               std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>       task_pool):
    config_(std::move(config)),
    tree_(std::move(tree)),
    group_sets_(std::move(group_sets)),
    load_back_ticket_registry_(std::make_shared<LoadBackTicketRegistry>(
        [this](const LoadBackTicket& ticket) { return commitLoadBack(ticket); },
        [this](const LoadBackTicket& ticket) { abortLoadBack(ticket); })),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    evictor_(
        group_sets_,
        [this](const TransferDescriptor& descriptor) { return executeTransfer(descriptor); },
        config_.enable_reverse_eviction) {}

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
    if (config_.enable_load_back && !config_.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("load back requires memory cache");
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
    RTP_LLM_LOG_INFO("destroying, closing load-back tickets...");
    load_back_ticket_registry_->shutdown();
    if (!initialized_) {
        RTP_LLM_LOG_INFO("destroyed");
        return;
    }
    RTP_LLM_LOG_INFO("load-back tickets closed, waiting for pending tasks...");
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
    LoadBackTicket::PendingLoadBackItems pending_load_back_items;
    prepareMatchedBlocks(matched_path, candidate_logically_valid, result, pending_load_back_items);
    if (config_.enable_load_back && !pending_load_back_items.empty()) {
        result.load_back_ticket = prepareLoadBackTicket(pending_load_back_items, valid_matched_block_count);
        if (result.load_back_ticket == nullptr) {
            result.load_back_blocks      = 0;
            result.host_load_back_blocks = 0;
            result.disk_load_back_blocks = 0;
        }
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
    // eviction). Reused nodes keep theirs; their demoted data comes from load_back.
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
        if (!submitEvictionLocked(*eviction_move)) {
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

bool BlockTreeCache::cancelLoadBack(const std::shared_ptr<AsyncContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    return load_back_worker_.cancelLoadBackNolock(context);
}

void BlockTreeCache::prepareMatchedBlocks(const std::vector<TreeNode*>&         matched_path,
                                          const std::vector<bool>&              candidate_logically_valid,
                                          BlockTreeMatchResult&                 result,
                                          LoadBackTicket::PendingLoadBackItems& pending_load_back_items) {
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

        if (!config_.enable_load_back) {
            continue;
        }
        const size_t logical_reuse_count = std::min(
            group_set->computeReuseBlockCount(logical_matched_block_count, matched_path), logical_matched_block_count);
        for (size_t i = logical_matched_block_count - logical_reuse_count; i < logical_matched_block_count; ++i) {
            if (i >= ready_reuse_begin && i < ready_matched_block_count) {
                continue;
            }
            TreeNode*         path_node          = matched_path[i];
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            prepareMatchedLoadBackItem(path_node, group_set, group_set_resource, i, result, pending_load_back_items);
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

void BlockTreeCache::prepareMatchedLoadBackItem(TreeNode*                             path_node,
                                                const GroupSetPtr&                    group_set,
                                                const GroupSetResource&               group_set_resource,
                                                size_t                                path_index,
                                                BlockTreeMatchResult&                 result,
                                                LoadBackTicket::PendingLoadBackItems& pending_load_back_items) {
    const Tier source_tier = group_set->getTopTier(group_set_resource);
    if (source_tier == Tier::NONE) {
        return;
    }

    const std::vector<BlockIdxType> source_blocks = group_set->getBlocks(group_set_resource, source_tier);

    LoadBackTicket::PendingLoadBackItem pending_item;
    pending_item.node             = path_node;
    pending_item.group_set_id     = group_set->groupSetId();
    pending_item.path_index       = path_index;
    pending_item.source_tier      = source_tier;
    pending_item.source_blocks    = source_blocks;
    pending_item.joined_load_back = group_set_resource.transfer_state == GroupSetTransferState::LOADING_BACK;
    pending_load_back_items.push_back(std::move(pending_item));

    if (source_tier == Tier::HOST) {
        result.host_load_back_blocks++;
        result.load_back_blocks++;
    } else if (source_tier == Tier::DISK) {
        result.disk_load_back_blocks++;
        result.load_back_blocks++;
    }

    RTP_LLM_LOG_DEBUG("planned logical settlement from %s group[%zu] node_key=%ld",
                      tierName(source_tier),
                      group_set->groupSetId(),
                      path_node->cache_key);
    if (group_set_resource.transfer_state == GroupSetTransferState::LOADING_BACK) {
        RTP_LLM_LOG_DEBUG(
            "match joined LOADING_BACK, node_key=%ld group_set=%zu", path_node->cache_key, group_set->groupSetId());
    }
}

std::shared_ptr<LoadBackTicket> BlockTreeCache::prepareLoadBackTicket(LoadBackTicket::PendingLoadBackItems& items,
                                                                      size_t logical_matched_blocks) {
    if (!reserveLoadBackItems(items)) {
        return nullptr;
    }

    size_t pending_transfer_count = 0;
    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.source_tier == Tier::HOST || item.source_tier == Tier::DISK) {
            ++pending_transfer_count;
        }
    }
    const std::shared_ptr<LoadBackAsyncContext> context =
        std::make_shared<LoadBackAsyncContext>(pending_transfer_count);
    for (LoadBackTicket::PendingLoadBackItem& item : items) {
        if (!item.joined_load_back) {
            continue;
        }
        if (!prepareJoinedLoadBackItem(item, context)) {
            abortLoadBackUnsafe(items, 0, context);
            return nullptr;
        }
    }

    std::shared_ptr<LoadBackTicket> ticket =
        load_back_ticket_registry_->createTicket(items, logical_matched_blocks, context);
    if (ticket == nullptr) {
        abortLoadBackUnsafe(items, 0, context);
    }
    return ticket;
}

bool BlockTreeCache::prepareJoinedLoadBackItem(LoadBackTicket::PendingLoadBackItem&         item,
                                               const std::shared_ptr<LoadBackAsyncContext>& context) {
    const std::optional<std::vector<BlockIdxType>> target_blocks =
        load_back_worker_.joinLoading(item.node, item.group_set_id, context);
    if (!target_blocks.has_value()) {
        RTP_LLM_LOG_WARNING("failed to join active load-back, group_set=%zu", item.group_set_id);
        return false;
    }
    item.target_device_blocks = target_blocks.value();
    if (item.target_device_blocks.size() != group_sets_[item.group_set_id]->devicePoolCount()) {
        item.target_device_blocks.clear();
        return false;
    }
    group_sets_[item.group_set_id]->referenceBlocks(
        MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
    return true;
}

bool BlockTreeCache::reserveLoadBackItems(const LoadBackTicket::PendingLoadBackItems& items) {
    const bool has_lower_tier_item =
        std::any_of(items.begin(), items.end(), [](const LoadBackTicket::PendingLoadBackItem& item) {
            return item.source_tier == Tier::HOST || item.source_tier == Tier::DISK;
        });
    if (items.empty() || !has_lower_tier_item) {
        return false;
    }

    for (const auto& item : items) {
        if (item.node == nullptr || item.group_set_id >= group_sets_.size()
            || (item.source_tier != Tier::DEVICE && item.source_tier != Tier::HOST && item.source_tier != Tier::DISK)) {
            return false;
        }
        const GroupSetPtr& group = group_sets_[item.group_set_id];
        if (group == nullptr || item.group_set_id >= item.node->group_set_resources.size()) {
            return false;
        }
        const GroupSetTransferState expected_state =
            item.joined_load_back ? GroupSetTransferState::LOADING_BACK : GroupSetTransferState::IDLE;
        if (item.node->group_set_resources[item.group_set_id].transfer_state != expected_state) {
            return false;
        }
        const size_t expected_source_count = item.source_tier == Tier::DEVICE ? group->devicePoolCount() : 1;
        if (item.source_blocks.size() != expected_source_count
            || group->getTopTier(item.node->group_set_resources[item.group_set_id]) != item.source_tier
            || group->getBlocks(item.node->group_set_resources[item.group_set_id], item.source_tier)
                   != item.source_blocks) {
            return false;
        }
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.joined_load_back) {
            continue;
        }
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, item.source_tier, {item.source_blocks}}, BlockRefType::REQUEST);
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.source_tier == Tier::DEVICE || item.joined_load_back) {
            continue;
        }
        if (!evictor_.reserveLoadBack(item.node, item.group_set_id, item.source_tier, item.source_blocks)) {
            abortLoadBackUnsafe(items, /*prepared_item_count=*/0, nullptr);
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncContext> BlockTreeCache::commitLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex>                 lock(mutex_);
    const LoadBackTicket::PendingLoadBackItems& items = ticket.items();

    size_t                                      prepared_item_count = 0;
    const std::shared_ptr<LoadBackAsyncContext> context             = ticket.context();
    ScopeRollback                               rollback_guard(
        [this, &items, &prepared_item_count, &context]() { abortLoadBackUnsafe(items, prepared_item_count, context); });
    if (context == nullptr) {
        RTP_LLM_LOG_WARNING("load-back ticket has no context");
        return nullptr;
    }

    LoadBackWorker::TaskPtr task;
    if (!load_back_worker_.createTask(items, group_sets_, context, task)) {
        return nullptr;
    }
    if (task != nullptr) {
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            const LoadBackTicket::PendingLoadBackItem& item = task->items[item_index];
            if (item.source_tier != Tier::DEVICE
                && !task->item_groups[item_index]->hasAllocatedDeviceBlocks(item.target_device_blocks)) {
                RTP_LLM_LOG_WARNING("invalid load-back target blocks, group_set=%zu", item.group_set_id);
                return nullptr;
            }
        }
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.source_tier == Tier::DEVICE || item.joined_load_back) {
            ++prepared_item_count;
            continue;
        }
        if (!load_back_worker_.startLoading(item.node, item.group_set_id, item.target_device_blocks, context)) {
            RTP_LLM_LOG_WARNING("failed to create loading record, group_set=%zu", item.group_set_id);
            return nullptr;
        }
        if (!evictor_.beginLoadBack(item.node, item.group_set_id, item.source_tier)) {
            const bool erased = load_back_worker_.eraseLoadingForOneContext(item.node, item.group_set_id, context);
            if (!erased) {
                RTP_LLM_LOG_ERROR("failed to erase load-back context, group_set=%zu", item.group_set_id);
            }
            RTP_LLM_LOG_WARNING("pending-to-loading transition failed, rolled back all %zu load_back items",
                                items.size());
            return nullptr;
        }
        // Add an in-flight copy holder. It becomes a cache holder only after
        // the target blocks are installed into the tree slot.
        group_sets_[item.group_set_id]->referenceBlocks(
            MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        ++prepared_item_count;
    }

    if (task != nullptr) {
        const bool submitted = task_pool_->submit([this, task]() { runLoadBackTask(task); });
        if (!submitted) {
            rollback_guard.run();
            const bool completed = context->onTaskFail();
            if (!completed) {
                RTP_LLM_LOG_ERROR("failed to complete rejected load-back task");
            }
            return context;
        }
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.joined_load_back) {
            group_sets_[item.group_set_id]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        }
    }
    rollback_guard.dismiss();
    return context;
}

void BlockTreeCache::abortLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadBackUnsafe(ticket.items(), 0, ticket.context());
}

void BlockTreeCache::abortLoadBackUnsafe(const LoadBackTicket::PendingLoadBackItems&  items,
                                         size_t                                       prepared_item_count,
                                         const std::shared_ptr<LoadBackAsyncContext>& context) {
    if (prepared_item_count > 0 && context == nullptr) {
        RTP_LLM_LOG_ERROR("missing context while aborting %zu prepared load-back items", prepared_item_count);
    }

    bool device_refs_released = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const auto&  item            = items[item_index];
        const size_t group_set_index = item.group_set_id;
        if (group_set_index >= group_sets_.size() || group_sets_[group_set_index] == nullptr) {
            continue;
        }
        const bool fully_prepared = item_index < prepared_item_count;
        if (item.joined_load_back) {
            if (context != nullptr) {
                const bool erased = load_back_worker_.eraseLoadingForOneContext(item.node, item.group_set_id, context);
                if (!erased) {
                    RTP_LLM_LOG_DEBUG("joined load-back context is no longer registered, group_set=%zu",
                                      item.group_set_id);
                }
            }
            if (!item.target_device_blocks.empty()) {
                group_sets_[group_set_index]->unreferenceBlocks(
                    MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}},
                    BlockRefType::REQUEST);
                device_refs_released = true;
            }
            continue;
        }
        if (item.source_tier != Tier::DEVICE && fully_prepared) {
            if (context != nullptr) {
                const bool erased = load_back_worker_.eraseLoadingForOneContext(item.node, item.group_set_id, context);
                if (!erased) {
                    RTP_LLM_LOG_WARNING("failed to erase aborted load-back context, group_set=%zu", item.group_set_id);
                }
            }
            group_sets_[group_set_index]->unreferenceBlocks(
                MultiNodeResource{item.group_set_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        }

        MultiNodeResource source_set{item.group_set_id, item.source_tier, {item.source_blocks}};
        if (item.node != nullptr) {
            source_set.tree_nodes = {item.node};
        }
        group_sets_[group_set_index]->unreferenceBlocks(source_set, BlockRefType::REQUEST);
        if (item.source_tier != Tier::DEVICE) {
            if (fully_prepared) {
                if (!evictor_.finishLoadBack(item.node, item.group_set_id, item.source_tier, false)) {
                    RTP_LLM_LOG_WARNING(
                        "loading state mismatch, group=%zu source=%s", item.group_set_id, tierName(item.source_tier));
                }
            } else {
                if (!evictor_.abortPendingLoadBack(
                        item.node, item.group_set_id, item.source_tier, item.source_blocks)) {
                    RTP_LLM_LOG_WARNING("reservation state mismatch, "
                                        "group=%zu source=%s",
                                        item.group_set_id,
                                        tierName(item.source_tier));
                }
                evictor_.refreshCandidatesAfterRelease(source_set);
            }
        } else {
            evictor_.refreshCandidatesAfterRelease(source_set);
            device_refs_released = true;
        }
    }
    if (device_refs_released) {
        checkWatermark();
    }
}

void BlockTreeCache::runLoadBackTask(const LoadBackWorker::TaskPtr& task) {
    if (task == nullptr || task->context == nullptr) {
        RTP_LLM_LOG_ERROR("invalid load-back task");
        return;
    }

    bool copy_success = false;
    try {
        bool prepared = !task->items.empty();
        for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
            LoadBackWorker::PrepareStatus status = load_back_worker_.prepareTransferItem(*task, item_index);
            if (status == LoadBackWorker::PrepareStatus::NEED_HOST_RECLAIM) {
                const size_t group_set_id = task->items[item_index].group_set_id;
                if (reclaimOneForGroup(group_set_id, Tier::HOST)) {
                    status = load_back_worker_.prepareTransferItem(*task, item_index);
                }
            }
            if (status != LoadBackWorker::PrepareStatus::READY) {
                if (status == LoadBackWorker::PrepareStatus::NEED_HOST_RECLAIM) {
                    RTP_LLM_LOG_WARNING("failed to prepare host staging block, group_set=%zu",
                                        task->items[item_index].group_set_id);
                }
                prepared = false;
            }
        }

        copy_success = load_back_worker_.runTransfer(*task,
                                                     *transfer_dispatcher_,
                                                     metrics_reporter_,
                                                     config_.memory_cache_disk_sync_timeout_ms,
                                                     config_.memory_cache_sync_timeout_ms,
                                                     prepared);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("load-back worker failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("load-back worker failed with unknown exception");
    }

    // Commit the copied batch only while every stateful item still belongs
    // to this load-back operation.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool                  settlement_success = settleLoadBackNolock(*task, copy_success);
        if (!settlement_success) {
            RTP_LLM_LOG_DEBUG("load-back task settled unsuccessfully");
        }
    }
}

bool BlockTreeCache::settleLoadBackNolock(LoadBackWorker::Task& task, bool copy_success) {
    bool settlement_success   = copy_success && task.context != nullptr;
    bool state_settled        = false;
    bool tree_data_mutated    = false;
    bool device_refs_released = false;

    RTP_LLM_CHECK_WITH_INFO(task.items.size() == task.item_groups.size()
                                && task.items.size() == task.target_installed.size(),
                            "malformed load-back task: items=%zu groups=%zu targets=%zu",
                            task.items.size(),
                            task.item_groups.size(),
                            task.target_installed.size());
    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
        const GroupSetPtr&                         group = task.item_groups[item_index];
        RTP_LLM_CHECK_WITH_INFO(group != nullptr && group->groupSetId() == item.group_set_id && item.node != nullptr
                                    && item.group_set_id < item.node->group_set_resources.size(),
                                "malformed load-back item: index=%zu group_set_id=%zu node=%p",
                                item_index,
                                item.group_set_id,
                                static_cast<void*>(item.node));
        if (settlement_success && item.source_tier != Tier::DEVICE
            && (item.target_device_blocks.size() != group->devicePoolCount()
                || item.node->group_set_resources[item.group_set_id].transfer_state
                       != GroupSetTransferState::LOADING_BACK)) {
            RTP_LLM_LOG_WARNING("completion state mismatch, group_set=%zu", item.group_set_id);
            settlement_success = false;
        }
    }

    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item         = task.items[item_index];
        const GroupSetPtr&                         group        = task.item_groups[item_index];
        const size_t                               group_set_id = item.group_set_id;

        MultiNodeResource source_protection{group_set_id, item.source_tier, {item.source_blocks}};
        source_protection.tree_nodes = {item.node};
        group->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        if (item.source_tier == Tier::DEVICE) {
            evictor_.refreshCandidatesAfterRelease(source_protection);
            device_refs_released = true;
            continue;
        }
        GroupSetResource& resource = item.node->group_set_resources[group_set_id];
        if (settlement_success) {
            MultiNodeResource target_holder{group_set_id, Tier::DEVICE, {item.target_device_blocks}};
            group->setBlocks(resource, Tier::DEVICE, item.target_device_blocks);
            group->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
            group->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
            group->unreferenceBlocks(MultiNodeResource{group_set_id, item.source_tier, {item.source_blocks}},
                                     BlockRefType::BLOCK_CACHE);
            group->evictFromTier(item.node, resource, item.source_tier);
            task.target_installed[item_index] = true;
            tree_data_mutated                 = true;
            RTP_LLM_CHECK_WITH_INFO(evictor_.finishLoadBack(item.node, group_set_id, item.source_tier, true),
                                    "load-back state changed after locked preflight, group_set_id=%zu",
                                    group_set_id);
            state_settled = true;
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!evictor_.finishLoadBack(item.node, group_set_id, item.source_tier, false)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group_set=%zu source=%s", group_set_id, tierName(item.source_tier));
        } else {
            state_settled = true;
        }
    }
    if (tree_data_mutated) {
        ++mutation_version_;
    }
    if (device_refs_released || state_settled) {
        checkWatermark();
    }
    load_back_worker_.releaseTaskResources(task);
    for (const LoadBackTicket::PendingLoadBackItem& item : task.items) {
        if (item.source_tier == Tier::DEVICE) {
            continue;
        }
        const bool completed = load_back_worker_.finishLoading(item.node, item.group_set_id, settlement_success);
        if (!completed) {
            RTP_LLM_LOG_WARNING("failed to finish loading record, group_set=%zu", item.group_set_id);
        }
    }
    return settlement_success;
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
    return submitEvictionLocked(*eviction_move);
}

void BlockTreeCache::reserveInFlightDeviceReleaseCreditsLocked(
    const std::vector<DeviceReleaseCredit>& release_credits) {
    for (const DeviceReleaseCredit& credit : release_credits) {
        if (credit.pool != nullptr) {
            ++in_flight_device_release_credits_[credit.pool];
        }
    }
}

void BlockTreeCache::settleInFlightDeviceReleaseCreditsLocked(
    const std::vector<DeviceReleaseCredit>& release_credits) noexcept {
    for (const DeviceReleaseCredit& credit : release_credits) {
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

bool BlockTreeCache::submitEvictionLocked(EvictionMove&                     eviction_move,
                                          std::vector<DeviceReleaseCredit>* release_credits) {
    if (release_credits != nullptr) {
        release_credits->clear();
    }
    if (eviction_move.target_tier != Tier::NONE && !config_.isTierEnabled(eviction_move.target_tier)) {
        eviction_move.target_tier = Tier::NONE;
    }

    auto plan = evictor_.buildPlan(eviction_move);
    if (!plan.has_value()) {
        return false;
    }

    std::vector<DeviceReleaseCredit>                    accepted_release_credits;
    std::set<std::pair<DeviceBlockPool*, BlockIdxType>> accepted_physical_releases;
    auto                                                collect_device_credits = [&](const EvictionMove& move) {
        if (move.source_tier != Tier::DEVICE) {
            return;
        }
        const size_t group_set_index = move.group_set_id;
        RTP_LLM_CHECK_WITH_INFO(group_set_index < group_sets_.size(),
                                "eviction plan has invalid group_set_id=%zu group_set_count=%zu",
                                group_set_index,
                                group_sets_.size());
        const auto& pools = group_sets_[group_set_index]->devicePools();
        RTP_LLM_CHECK_WITH_INFO(move.source_blocks.size() == pools.size(),
                                "eviction plan DEVICE width mismatch: group_set_id=%zu expected=%zu actual=%zu",
                                group_set_index,
                                pools.size(),
                                move.source_blocks.size());
        for (size_t i = 0; i < pools.size(); ++i) {
            const auto& pool = pools[i];
            if (!isNullBlockIdx(move.source_blocks[i])
                && accepted_physical_releases.emplace(pool.get(), move.source_blocks[i]).second) {
                accepted_release_credits.push_back({pool, move.source_blocks[i]});
            }
        }
    };
    collect_device_credits(plan->primary);
    for (const EvictionMove& cascade_move : plan->cascade_moves) {
        collect_device_credits(cascade_move);
    }

    if (!plan->needsCopy()) {
        BlockTreeEvictor::CopyResultSet results;
        results.primary_success = true;
        results.cascade_success.assign(plan->cascade_moves.size(), true);
        evictor_.complete(*tree_, *plan, results);
        metrics_reporter_.reportEvictionFinished(*plan, results, group_sets_);
        ++mutation_version_;
        if (release_credits != nullptr) {
            *release_credits = std::move(accepted_release_credits);
        }
        return true;
    }

    auto       plan_ptr                  = std::make_shared<BlockTreeEvictor::EvictionPlan>(std::move(*plan));
    auto       in_flight_release_credits = accepted_release_credits;
    const bool submitted =
        task_pool_->submit([this, plan_ptr, in_flight_release_credits = std::move(in_flight_release_credits)]() {
            performEvictionCopy(*plan_ptr, in_flight_release_credits);
        });
    if (!submitted) {
        evictor_.rollbackPreparedPlan(*plan_ptr);
        return false;
    }
    reserveInFlightDeviceReleaseCreditsLocked(accepted_release_credits);
    if (release_credits != nullptr) {
        *release_credits = std::move(accepted_release_credits);
    }
    return true;
}

void BlockTreeCache::performEvictionCopy(const BlockTreeEvictor::EvictionPlan&   plan,
                                         const std::vector<DeviceReleaseCredit>& release_credits) {
    const Tier    source_tier            = plan.primary.source_tier;
    const Tier    target_tier            = plan.primary.target_tier;
    const size_t  transfer_block_count   = plan.cascade_moves.size() + 1;
    const int64_t transfer_begin_time_us = metrics_reporter_.reportTransferStarted(source_tier, target_tier);
    BlockTreeEvictor::CopyResultSet copy_results;
    copy_results.primary_success = false;
    copy_results.cascade_success.assign(plan.cascade_moves.size(), false);

    auto worker_finalization_action = [this,
                                       &plan,
                                       &release_credits,
                                       &copy_results,
                                       source_tier,
                                       target_tier,
                                       transfer_block_count,
                                       transfer_begin_time_us]() noexcept {
        const bool transfer_success = copy_results.primary_success
                                      && std::all_of(copy_results.cascade_success.begin(),
                                                     copy_results.cascade_success.end(),
                                                     [](bool success) { return success; });
        metrics_reporter_.reportTransferFinished(
            source_tier, target_tier, transfer_block_count, transfer_begin_time_us, transfer_success);

        bool credit_settlement_attempted = false;
        auto credit_settlement_action    = [this, &release_credits, &credit_settlement_attempted]() noexcept {
            if (credit_settlement_attempted) {
                return;
            }
            credit_settlement_attempted = true;
            std::lock_guard<std::mutex> lock(mutex_);
            settleInFlightDeviceReleaseCreditsLocked(release_credits);
        };
        ScopeRollback<decltype(credit_settlement_action)> credit_settlement_guard(std::move(credit_settlement_action));

        bool completion_succeeded = false;
        bool plan_terminalized    = false;
        bool plan_succeeded       = false;
        bool copy_ok              = copy_results.primary_success;

        CacheKeyType          remote_cache_key = 0;
        std::optional<size_t> remote_group_set_id;
        if (copy_ok && plan.primary.node != nullptr) {
            remote_cache_key    = plan.primary.node->cache_key;
            remote_group_set_id = plan.primary.group_set_id;
        }

        try {
            std::lock_guard<std::mutex> lock(mutex_);
            try {
                evictor_.complete(*tree_, plan, copy_results);
                completion_succeeded = true;
                plan_terminalized    = true;
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("eviction completion failed; rolling back accepted plan: %s", error.what());
                evictor_.rollbackPreparedPlan(plan);
                plan_terminalized = true;
            } catch (...) {
                RTP_LLM_LOG_ERROR("eviction completion failed with unknown exception; rolling back "
                                  "accepted plan");
                evictor_.rollbackPreparedPlan(plan);
                plan_terminalized = true;
            }

            // Credits are accounting-only. The completed or rolled-back evictor plan above owns all
            // pool reference transitions; settlement must never add another decRef.
            credit_settlement_attempted = true;
            settleInFlightDeviceReleaseCreditsLocked(release_credits);

            const bool mutated = plan_terminalized && completion_succeeded
                                 && (copy_results.primary_success
                                     || std::any_of(copy_results.cascade_success.begin(),
                                                    copy_results.cascade_success.end(),
                                                    [](bool success) { return success; }));
            plan_succeeded = plan_terminalized && completion_succeeded && copy_results.primary_success
                             && copy_results.cascade_success.size() == plan.cascade_moves.size()
                             && std::all_of(copy_results.cascade_success.begin(),
                                            copy_results.cascade_success.end(),
                                            [](bool success) { return success; });
            if (mutated) {
                ++mutation_version_;
            }
            if (plan_succeeded) {
                // A fully completed device->host or host->disk plan changes the target tier's
                // pressure. This remains under the cache lock, after this plan's credits settle.
                checkWatermark();
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed with unknown exception");
        }
        metrics_reporter_.reportEvictionFinished(plan, copy_results, group_sets_);

        // If an exception escaped before the in-lock settlement attempt, perform that accounting step
        // now. The no-throw guard records one attempt and prevents a duplicate decrement.
        credit_settlement_guard.run();

        if (plan_terminalized && completion_succeeded && copy_ok && config_.enable_remote_cache
            && remote_group_set_id.has_value()) {
            try {
                evictor_.writeRemoteThrough(storage_backend_, remote_cache_key, *remote_group_set_id);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("remote eviction write-through failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("remote eviction write-through failed with unknown exception");
            }
        }
    };
    ScopeRollback<decltype(worker_finalization_action)> worker_finalization_guard(
        std::move(worker_finalization_action));

    try {
        if (!transfer_dispatcher_->hasMultiRankEngine()) {
            copy_results = evictor_.performCopy(plan);
        } else {
            std::vector<TransferDescriptor> descriptors;
            const bool                      batch_ready = buildEvictionTransferBatch(plan, descriptors);
            const bool                      transfer_success =
                batch_ready && transfer_dispatcher_->executeMultiRank(descriptors, evictionTransferTimeoutMs(plan));
            copy_results.primary_success = transfer_success;
            copy_results.cascade_success.assign(plan.cascade_moves.size(), transfer_success);
        }
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
}

bool BlockTreeCache::buildEvictionTransferBatch(const BlockTreeEvictor::EvictionPlan& plan,
                                                std::vector<TransferDescriptor>&      descriptors) const {
    descriptors.clear();
    descriptors.reserve(1 + plan.cascade_moves.size());

    TransferDescriptor primary_descriptor;
    if (!BlockTreeEvictor::buildTransferDescriptor(plan.primary, primary_descriptor)) {
        return false;
    }
    descriptors.push_back(std::move(primary_descriptor));

    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        TransferDescriptor cascade_descriptor;
        if (!BlockTreeEvictor::buildTransferDescriptor(cascade_move, cascade_descriptor)) {
            descriptors.clear();
            return false;
        }
        descriptors.push_back(std::move(cascade_descriptor));
    }
    return true;
}

int BlockTreeCache::evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const {
    bool uses_disk = plan.primary.source_tier == Tier::DISK || plan.primary.target_tier == Tier::DISK;
    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        if (cascade_move.source_tier == Tier::DISK || cascade_move.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    if (!uses_disk) {
        return config_.memory_cache_sync_timeout_ms;
    }
    return std::max(config_.memory_cache_sync_timeout_ms, config_.memory_cache_disk_sync_timeout_ms);
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
                std::vector<DeviceReleaseCredit> release_credits;
                if (!submitEvictionLocked(*eviction_move, &release_credits)) {
                    unavailable[group_index] = true;
                    continue;
                }
                bool credited_uncovered_pool = false;
                for (const DeviceReleaseCredit& credit : release_credits) {
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
                submitEvictionLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
