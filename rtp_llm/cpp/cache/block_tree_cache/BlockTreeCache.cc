#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockCacheTaskPool.h"
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

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                    tree,
                               std::vector<ComponentGroupPtr>                component_groups,
                               std::shared_ptr<const std::vector<Component>> components,
                               BlockTreeCacheConfig                          config,
                               std::shared_ptr<StorageBackend>               storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>      transfer_dispatcher,
                               std::unique_ptr<BlockCacheTaskPool>           task_pool,
                               std::vector<std::string>                      per_tag_tags,
                               std::vector<DeviceKVCacheGroupPtr>            per_tag_device_groups,
                               std::vector<PerTagMapping>                    per_tag_mapping):
    config_(std::move(config)),
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::move(components)),
    per_tag_tags_(std::move(per_tag_tags)),
    per_tag_device_groups_(std::move(per_tag_device_groups)),
    per_tag_mapping_(std::move(per_tag_mapping)),
    load_back_ticket_registry_(std::make_shared<LoadBackTicketRegistry>(
        [this](const LoadBackTicket& ticket) { return commitLoadBack(ticket); },
        [this](const LoadBackTicket& ticket) { abortLoadBack(ticket); })),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    evictor_(
        component_groups_,
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
    if (!validateConfiguration()) {
        RTP_LLM_LOG_ERROR("invalid configuration");
        return false;
    }
    if (!initDeviceGroupTags()) {
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
    RTP_LLM_LOG_INFO("initialized with %zu component groups, %zu components, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     component_groups_.size(),
                     components_->size(),
                     config_.eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_memory_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const ComponentGroupPtr& component_group : component_groups_) {
        RTP_LLM_LOG_INFO("  group[%d] type=%s host_pool=%s disk_pool=%s",
                         component_group->component_group_id,
                         cacheGroupTypeName(component_group->group_type),
                         component_group->hostPool() ? "enabled" : "null",
                         component_group->diskPool() ? "enabled" : "null");
    }
    initialized_ = true;
    return true;
}

bool BlockTreeCache::initDeviceGroupTags() {
    device_group_tags_.clear();
    device_group_tags_.resize(component_groups_.size());
    if (per_tag_mapping_.empty() || per_tag_tags_.empty()) {
        RTP_LLM_LOG_ERROR("declarative mapping registry must not be empty");
        return false;
    }
    if (per_tag_tags_.size() != per_tag_mapping_.size()) {
        RTP_LLM_LOG_ERROR(
            "tag/mapping size mismatch, tags=%zu mappings=%zu", per_tag_tags_.size(), per_tag_mapping_.size());
        return false;
    }
    for (size_t group_id = 0; group_id < component_groups_.size(); ++group_id) {
        const ComponentGroupPtr& component_group = component_groups_[group_id];
        if (component_group == nullptr || component_group->component_group_id != static_cast<int>(group_id)
            || component_group->devicePoolCount() == 0) {
            RTP_LLM_LOG_ERROR("invalid component group, group=%zu", group_id);
            return false;
        }
        device_group_tags_[group_id].assign(component_group->devicePoolCount(), {});
    }

    for (size_t tag_index = 0; tag_index < per_tag_mapping_.size(); ++tag_index) {
        const PerTagMapping& mapping = per_tag_mapping_[tag_index];
        if (mapping.component_group_id == -1) {
            if (mapping.local_pool_index != -1) {
                RTP_LLM_LOG_ERROR("invalid non-reusable mapping, tag=%s", per_tag_tags_[tag_index].c_str());
                return false;
            }
            continue;
        }
        if (mapping.component_group_id < 0 || mapping.local_pool_index < 0) {
            RTP_LLM_LOG_ERROR("negative mapping index, tag=%s", per_tag_tags_[tag_index].c_str());
            return false;
        }
        const size_t group_id      = static_cast<size_t>(mapping.component_group_id);
        const size_t local_pool_id = static_cast<size_t>(mapping.local_pool_index);
        if (group_id >= device_group_tags_.size() || local_pool_id >= device_group_tags_[group_id].size()
            || !device_group_tags_[group_id][local_pool_id].empty() || per_tag_tags_[tag_index].empty()) {
            RTP_LLM_LOG_ERROR("invalid mapping, tag=%s group=%zu local=%zu",
                              per_tag_tags_[tag_index].c_str(),
                              group_id,
                              local_pool_id);
            return false;
        }
        device_group_tags_[group_id][local_pool_id] = per_tag_tags_[tag_index];
    }

    for (size_t group_id = 0; group_id < device_group_tags_.size(); ++group_id) {
        const std::vector<std::string>& device_group_tags = device_group_tags_[group_id];
        if (std::any_of(device_group_tags.begin(), device_group_tags.end(), [](const std::string& tag) {
                return tag.empty();
            })) {
            RTP_LLM_LOG_ERROR("incomplete mapping, group=%zu", group_id);
            return false;
        }
        if (device_group_tags != component_groups_[group_id]->tags()) {
            RTP_LLM_LOG_ERROR("component group tag mapping mismatch, group=%zu", group_id);
            return false;
        }
    }
    return true;
}

bool BlockTreeCache::validateConfiguration() const {
    if (tree_ == nullptr || components_ == nullptr) {
        RTP_LLM_LOG_ERROR("tree and component registry must be initialized");
        return false;
    }
    if (component_groups_.empty() && !components_->empty()) {
        RTP_LLM_LOG_ERROR("empty component groups require an empty component registry");
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

    for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
        const ComponentGroupPtr& group = component_groups_[group_index];
        if (group == nullptr || group->component_group_id != static_cast<int>(group_index)) {
            RTP_LLM_LOG_ERROR("component group must be non-null and indexed by id, index=%zu", group_index);
            return false;
        }
        if (!group->hasLayout()) {
            RTP_LLM_LOG_ERROR("component group %zu has no finalized layout", group_index);
            return false;
        }

        const auto& membership = group->componentIndices();
        const auto& layout     = group->layout();
        if (membership.empty() || membership.size() != layout.componentCount()
            || membership.size() != group->devicePoolCount()) {
            RTP_LLM_LOG_ERROR("group %zu membership/layout/pool counts differ: %zu/%zu/%zu",
                              group_index,
                              membership.size(),
                              layout.componentCount(),
                              group->devicePoolCount());
            return false;
        }

        std::unordered_set<std::string> tags;
        size_t                          slice_index     = 0;
        size_t                          expected_offset = 0;
        for (size_t component_position = 0; component_position < membership.size(); ++component_position) {
            const int component_index = membership[component_position];
            if (component_index < 0 || static_cast<size_t>(component_index) >= components_->size()) {
                RTP_LLM_LOG_ERROR("group %zu has invalid component index %d", group_index, component_index);
                return false;
            }
            const Component& component = (*components_)[static_cast<size_t>(component_index)];
            if (component.component_id != component_index
                || component.component_group_id != static_cast<int>(group_index) || component.tag.empty()
                || !tags.insert(component.tag).second || component.layerCount() == 0
                || component.model_layer_ids.size() != component.layerCount()) {
                RTP_LLM_LOG_ERROR("invalid component binding index=%d group=%zu pool_position=%zu",
                                  component_index,
                                  group_index,
                                  component_position);
                return false;
            }
            for (size_t layer_index = 0; layer_index < component.layerCount(); ++layer_index) {
                if (slice_index >= layout.slices().size()) {
                    RTP_LLM_LOG_ERROR("group %zu layout has too few slices", group_index);
                    return false;
                }
                const auto&  slice = layout.slices()[slice_index++];
                const size_t bytes = component.layerBytes(layer_index);
                if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - expected_offset
                    || slice.component_idx != component_position || slice.layer_idx != layer_index
                    || slice.offset_bytes != expected_offset) {
                    RTP_LLM_LOG_ERROR("group %zu layout drift at component=%zu layer=%zu",
                                      group_index,
                                      component_position,
                                      layer_index);
                    return false;
                }
                expected_offset += bytes;
            }
        }
        if (slice_index != layout.slices().size() || expected_offset != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("group %zu layout slice count or payload drift", group_index);
            return false;
        }

        const auto host_pool = group->hostPool();
        const auto disk_pool = group->diskPool();
        if (config_.enable_memory_cache && host_pool == nullptr) {
            RTP_LLM_LOG_ERROR("memory cache group %zu has no host pool", group_index);
            return false;
        }
        if (config_.enable_disk_cache && disk_pool == nullptr) {
            RTP_LLM_LOG_ERROR("disk cache group %zu has no disk pool", group_index);
            return false;
        }
        if (host_pool != nullptr && host_pool->payloadBytes() != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("group %zu host/layout payload mismatch: %zu/%zu",
                              group_index,
                              host_pool->payloadBytes(),
                              layout.payloadBytes());
            return false;
        }
        if (disk_pool != nullptr && disk_pool->payloadBytes() != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("group %zu disk/layout payload mismatch: %zu/%zu",
                              group_index,
                              disk_pool->payloadBytes(),
                              layout.payloadBytes());
            return false;
        }
    }
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
    if (initialized_) {
        drainTreeHolds();
    }
    RTP_LLM_LOG_INFO("destroyed");
}

void BlockTreeCache::drainTreeHolds() {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK_WITH_INFO(tree_ != nullptr && tree_->root() != nullptr,
                            "BlockTreeCache::drainTreeHolds: tree and root must be valid");

    const auto drain_node = [this](TreeNode* node) {
        RTP_LLM_CHECK_WITH_INFO(node != nullptr, "BlockTreeCache::drainTreeHolds: node must be valid");
        RTP_LLM_CHECK_WITH_INFO(node->group_slots.size() == component_groups_.size(),
                                "BlockTreeCache::drainTreeHolds: slot count mismatch, slots=%zu groups=%zu",
                                node->group_slots.size(),
                                component_groups_.size());

        for (size_t component_group_index = 0; component_group_index < component_groups_.size();
             ++component_group_index) {
            const ComponentGroupPtr& component_group = component_groups_[component_group_index];
            RTP_LLM_CHECK_WITH_INFO(component_group != nullptr
                                        && component_group->component_group_id
                                               == static_cast<int>(component_group_index),
                                    "BlockTreeCache::drainTreeHolds: component group is not indexed by id, index=%zu",
                                    component_group_index);

            GroupSlot&                      slot          = node->group_slots[component_group_index];
            const std::vector<BlockIdxType> device_blocks = slot.device_blocks;
            if (!device_blocks.empty()) {
                RTP_LLM_CHECK_WITH_INFO(
                    device_blocks.size() == component_group->devicePoolCount(),
                    "BlockTreeCache::drainTreeHolds: device slot/pool count mismatch, group=%zu slots=%zu pools=%zu",
                    component_group_index,
                    device_blocks.size(),
                    component_group->devicePoolCount());
                // Keep shutdown symmetric with referenceBlocks/unreferenceBlocks:
                // pool-less structural slots carry no hold, while real pools are released exactly once.
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::DEVICE, {device_blocks}},
                    BlockRefType::BLOCK_CACHE);
                std::fill(slot.device_blocks.begin(), slot.device_blocks.end(), NULL_BLOCK_IDX);
            }

            if (!isNullBlockIdx(slot.host_block)) {
                const BlockIdxType host_block = slot.host_block;
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::HOST, {{host_block}}},
                    BlockRefType::BLOCK_CACHE);
                slot.host_block = NULL_BLOCK_IDX;
            }

            if (!isNullBlockIdx(slot.disk_slot)) {
                const BlockIdxType disk_block = slot.disk_slot;
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::DISK, {{disk_block}}},
                    BlockRefType::BLOCK_CACHE);
                slot.disk_slot = NULL_BLOCK_IDX;
            }

            slot.transfer_state = SlotTransferState::IDLE;
        }
    };

    drain_node(tree_->root());
    for (const std::unique_ptr<TreeNode>& node : tree_->nodes()) {
        drain_node(node.get());
    }
}

TransferStatus BlockTreeCache::executeTransfer(const TransferDescriptor& descriptor) {
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

    size_t structurally_matchable_count = 0;
    for (TreeNode* path_node : tree_find_result.path) {
        if (!isNodeStructurallyMatchable(path_node)) {
            break;
        }
        ++structurally_matchable_count;
    }
    tree_find_result.path.resize(structurally_matchable_count);
    if (tree_find_result.path.empty()) {
        return result;
    }

    size_t            valid_matched_block_count = 0;
    std::vector<bool> candidate_logically_valid;
    candidate_logically_valid.reserve(tree_find_result.path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(component_groups_.size());
    for (ComponentGroupPtr& component_group : component_groups_) {
        match_validators.push_back(component_group->createMatchValidator());
    }
    for (size_t i = 0; i < tree_find_result.path.size(); ++i) {
        TreeNode* path_node        = tree_find_result.path[i];
        bool      all_groups_valid = true;
        for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
            const size_t component_group_index =
                static_cast<size_t>(component_groups_[group_index]->component_group_id);
            GroupSlot& group_slot  = path_node->group_slots[component_group_index];
            const bool group_valid = match_validators[group_index]->validate(path_node, group_slot);
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
        if (!reserveLoadBackItems(pending_load_back_items)) {
            pending_load_back_items.clear();
            result.load_back_blocks      = 0;
            result.host_load_back_blocks = 0;
            result.disk_load_back_blocks = 0;
            return result;
        }
        result.load_back_ticket =
            load_back_ticket_registry_->createTicket(pending_load_back_items, valid_matched_block_count);
        if (result.load_back_ticket == nullptr) {
            abortLoadBackNolock(pending_load_back_items, 0);
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

bool BlockTreeCache::isNodeStructurallyMatchable(const TreeNode* node) const {
    if (node == nullptr || node->group_slots.size() != component_groups_.size()) {
        RTP_LLM_LOG_WARNING("malformed group slot count, node_key=%ld expected=%zu actual=%zu",
                            node == nullptr ? 0 : node->cache_key,
                            component_groups_.size(),
                            node == nullptr ? 0 : node->group_slots.size());
        return false;
    }

    for (const ComponentGroupPtr& group : component_groups_) {
        if (group == nullptr || group->component_group_id < 0) {
            return false;
        }
        const size_t gid = static_cast<size_t>(group->component_group_id);
        if (gid >= node->group_slots.size()) {
            return false;
        }

        const GroupSlot& slot                = node->group_slots[gid];
        const bool       has_device_storage  = slot.has_value(Tier::DEVICE);
        const bool       has_complete_device = group->hasCompleteDeviceValue(slot);
        if (has_device_storage != has_complete_device) {
            RTP_LLM_LOG_WARNING(
                "partial DEVICE slot, node_key=%ld group_id=%d", node->cache_key, group->component_group_id);
            return false;
        }

        const int serving_tier_count = static_cast<int>(has_complete_device)
                                       + static_cast<int>(slot.has_value(Tier::HOST))
                                       + static_cast<int>(slot.has_value(Tier::DISK));
        if (serving_tier_count > 1) {
            RTP_LLM_LOG_WARNING(
                "multiple serving tiers, node_key=%ld group_id=%d", node->cache_key, group->component_group_id);
            return false;
        }
    }
    return true;
}

void BlockTreeCache::insert(TreeNode*                                  parent,
                            const CacheKeysType&                       cache_keys,
                            const std::vector<std::vector<GroupSlot>>& slots) {
    insertImpl(parent, cache_keys, slots, false);
}

void BlockTreeCache::insertSparse(TreeNode*                                  parent,
                                  const CacheKeysType&                       cache_keys,
                                  const std::vector<std::vector<GroupSlot>>& slots) {
    insertImpl(parent, cache_keys, slots, true);
}

void BlockTreeCache::insertImpl(TreeNode*                                  parent,
                                const CacheKeysType&                       cache_keys,
                                const std::vector<std::vector<GroupSlot>>& slots,
                                bool                                       allow_sparse_slots) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    if (slots.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/slot size mismatch, keys=%zu slots=%zu", cache_keys.size(), slots.size());
        return;
    }
    std::vector<std::vector<GroupSlot>> sanitized_slots = slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].size() != component_groups_.size()) {
            RTP_LLM_LOG_WARNING("component slot mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                component_groups_.size(),
                                slots[i].size());
            return;
        }
        for (size_t component_group_index = 0; component_group_index < component_groups_.size();
             ++component_group_index) {
            const auto& group                  = component_groups_[component_group_index];
            const auto& slot                   = slots[i][component_group_index];
            const bool  structurally_absent    = slot.device_blocks.empty() && slot.is_empty();
            const bool  allowed_sparse_absence = allow_sparse_slots && group != nullptr
                                                && group->group_type != CacheGroupType::FULL && structurally_absent;
            if (group == nullptr) {
                RTP_LLM_LOG_WARNING("null component group, index=%zu component=%zu", i, component_group_index);
                return;
            }
            if (allowed_sparse_absence || group->hasCompleteDeviceValue(slot)) {
                continue;
            }

            if (slot.has_value(Tier::DEVICE)) {
                RTP_LLM_LOG_WARNING("partial DEVICE input rejected, "
                                    "key=%ld component=%zu expected=%zu actual=%zu",
                                    cache_keys[i],
                                    component_group_index,
                                    group->devicePoolCount(),
                                    slot.device_blocks.size());
            }

            // Invalid or absent DEVICE payload is group-local: preserve the
            // topology insertion, but give the tree an exact-size empty slot so
            // no partial ownership can become visible or gain a cache holder.
            sanitized_slots[i][component_group_index] = GroupSlot{};
            sanitized_slots[i][component_group_index].device_blocks.assign(group->devicePoolCount(), NULL_BLOCK_IDX);
        }
    }

    BlockTreeInsertResult insert_result = tree_->insertNode(parent, cache_keys, sanitized_slots);

    // incRef cache-hold on new nodes' device blocks (balanced by unreferenceBlocks on
    // eviction). Reused nodes keep theirs; their demoted data comes from load_back.
    for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
        TreeNode* node = inserted.node;
        if (node == nullptr) {
            continue;
        }
        for (ComponentGroupPtr& group : component_groups_) {
            const size_t gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            GroupSlot& slot = node->group_slots[gid];
            if (group->hasCompleteDeviceValue(slot)) {
                const std::vector<BlockIdxType> blocks = group->getBlocks(slot, Tier::DEVICE);
                if (!blocks.empty()) {
                    group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}},
                                           BlockRefType::BLOCK_CACHE);
                }
            }
        }
    }

    // Existing nodes may independently refill one empty component. Take a tree
    // holder only for the exact adopted component; other slots already own theirs.
    for (const BlockTreeAdoptedSlot& adopted : insert_result.adopted_slots) {
        if (adopted.node == nullptr || adopted.component_group_id < 0) {
            continue;
        }
        const size_t gid = static_cast<size_t>(adopted.component_group_id);
        if (gid >= component_groups_.size() || component_groups_[gid] == nullptr
            || gid >= adopted.node->group_slots.size()) {
            continue;
        }
        auto&      group = component_groups_[gid];
        GroupSlot& slot  = adopted.node->group_slots[gid];
        if (!group->hasCompleteDeviceValue(slot)) {
            RTP_LLM_LOG_WARNING(
                "incomplete adopted slot event, key=%ld group=%d", adopted.node->cache_key, adopted.component_group_id);
            continue;
        }
        group->referenceBlocks(
            GroupBlockSet{adopted.component_group_id, Tier::DEVICE, {group->getBlocks(slot, Tier::DEVICE)}},
            BlockRefType::BLOCK_CACHE);
    }

    const bool changed = !insert_result.inserted_nodes.empty() || !insert_result.adopted_slots.empty();
    if (!changed) {
        return;
    }

    // Stamp and refresh only newly created nodes and exact adopted components.
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
    const auto tag_it = std::find(per_tag_tags_.begin(), per_tag_tags_.end(), tag);
    if (tag_it == per_tag_tags_.end()) {
        return 0;
    }
    const size_t tag_index = static_cast<size_t>(std::distance(per_tag_tags_.begin(), tag_it));
    if (tag_index >= per_tag_mapping_.size()) {
        return 0;
    }
    const PerTagMapping& mapping = per_tag_mapping_[tag_index];
    if (mapping.component_group_id < 0 || mapping.local_pool_index < 0) {
        return 0;
    }
    const size_t gid = static_cast<size_t>(mapping.component_group_id);
    if (gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
        return 0;
    }
    const auto&  device_pools     = component_groups_[gid]->devicePools();
    const size_t local_pool_index = static_cast<size_t>(mapping.local_pool_index);
    if (local_pool_index >= device_pools.size() || device_pools[local_pool_index] == nullptr) {
        return 0;
    }

    const size_t initial_free = device_pools[local_pool_index]->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        auto eviction_move = evictor_.chooseVictim(mapping.component_group_id, Tier::DEVICE);
        if (!eviction_move.has_value()) {
            break;
        }
        eviction_move->target_tier = Tier::NONE;
        if (!submitEvictionLocked(*eviction_move)) {
            break;
        }
        const size_t current_free = device_pools[local_pool_index]->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    RTP_LLM_LOG_DEBUG("tag=%s component_group[%d] reclaimed %zu/%zu device blocks",
                      tag.c_str(),
                      mapping.component_group_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

void BlockTreeCache::releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& set : sets) {
        auto gid = static_cast<size_t>(set.component_group_id);
        if (gid < component_groups_.size()) {
            component_groups_[gid]->unreferenceBlocks(set, BlockRefType::REQUEST);
            // Releasing a match reference may make the node evictable again.
            evictor_.refreshCandidatesAfterRelease(set);
        }
    }
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
    return metrics_reporter_.collectPoolMetricsSnapshots(component_groups_, evictor_);
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_.collectEvictableMetricsSnapshots(component_groups_, evictor_);
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
        const bool reusable = std::any_of(
            node->group_slots.begin(), node->group_slots.end(), [](const GroupSlot& slot) { return !slot.is_empty(); });
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
    if (candidate_logically_valid.size() != logical_matched_block_count) {
        RTP_LLM_LOG_WARNING("candidate validity size mismatch, path=%zu valid=%zu",
                            logical_matched_block_count,
                            candidate_logically_valid.size());
        return;
    }

    const size_t ready_matched_block_count = computeReadyMatchedBlockCount(matched_path, candidate_logically_valid);
    if (ready_matched_block_count > 0) {
        result.matched_node   = matched_path[ready_matched_block_count - 1];
        result.matched_blocks = ready_matched_block_count;
        evictor_.onMatched(std::vector<TreeNode*>(
            matched_path.begin(), matched_path.begin() + static_cast<ptrdiff_t>(ready_matched_block_count)));
    }

    for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
        ComponentGroupPtr& component_group       = component_groups_[group_index];
        const size_t       component_group_index = static_cast<size_t>(component_group->component_group_id);
        GroupBlockSet      matched_device_blocks{component_group->component_group_id, Tier::DEVICE};
        const std::vector<std::string>& device_group_tags = device_group_tags_[component_group_index];

        const size_t ready_reuse_count =
            std::min(component_group->computeReuseBlockCount(ready_matched_block_count, matched_path),
                     ready_matched_block_count);
        const size_t ready_reuse_begin = ready_matched_block_count - ready_reuse_count;
        for (size_t i = ready_reuse_begin; i < ready_matched_block_count; ++i) {
            TreeNode*                       path_node     = matched_path[i];
            GroupSlot&                      group_slot    = path_node->group_slots[component_group_index];
            const std::vector<BlockIdxType> device_blocks = component_group->getBlocks(group_slot, Tier::DEVICE);
            for (size_t tag_group_index = 0; tag_group_index < per_tag_mapping_.size(); ++tag_group_index) {
                const PerTagMapping& tag_mapping = per_tag_mapping_[tag_group_index];
                if (tag_mapping.component_group_id != component_group->component_group_id
                    || tag_mapping.local_pool_index < 0) {
                    continue;
                }
                const size_t local_pool_index = static_cast<size_t>(tag_mapping.local_pool_index);
                if (local_pool_index >= device_blocks.size() || device_blocks[local_pool_index] == NULL_BLOCK_IDX) {
                    continue;
                }
                result.group_block_indices[per_tag_tags_[tag_group_index]].push_back(device_blocks[local_pool_index]);
            }
            matched_device_blocks.per_node.push_back(device_blocks);
            matched_device_blocks.nodes.push_back(path_node);
        }

        if (!matched_device_blocks.per_node.empty()) {
            component_group->referenceBlocks(matched_device_blocks, BlockRefType::REQUEST);
            result.matched_block_sets.push_back(std::move(matched_device_blocks));
        }

        if (!config_.enable_load_back) {
            continue;
        }
        const size_t logical_reuse_count =
            std::min(component_group->computeReuseBlockCount(logical_matched_block_count, matched_path),
                     logical_matched_block_count);
        for (size_t i = logical_matched_block_count - logical_reuse_count; i < logical_matched_block_count; ++i) {
            if (i >= ready_reuse_begin && i < ready_matched_block_count) {
                continue;
            }
            TreeNode*  path_node  = matched_path[i];
            GroupSlot& group_slot = path_node->group_slots[component_group_index];
            prepareMatchedLoadBackItem(
                path_node, component_group, group_slot, i, device_group_tags, result, pending_load_back_items);
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
        for (const ComponentGroupPtr& component_group : component_groups_) {
            const size_t reuse_count =
                std::min(component_group->computeReuseBlockCount(candidate_count, matched_path), candidate_count);
            const size_t component_group_index = static_cast<size_t>(component_group->component_group_id);
            for (size_t path_index = candidate_count - reuse_count; path_index < candidate_count; ++path_index) {
                TreeNode* path_node = matched_path[path_index];
                if (component_group_index >= path_node->group_slots.size()
                    || !component_group->hasCompleteDeviceValue(path_node->group_slots[component_group_index])) {
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
                                                const ComponentGroupPtr&              component_group,
                                                const GroupSlot&                      group_slot,
                                                size_t                                path_index,
                                                const std::vector<std::string>&       device_group_tags,
                                                BlockTreeMatchResult&                 result,
                                                LoadBackTicket::PendingLoadBackItems& pending_load_back_items) {
    Tier source_tier = Tier::NONE;
    if (component_group->hasCompleteDeviceValue(group_slot)) {
        source_tier = Tier::DEVICE;
    } else if (group_slot.has_value(Tier::HOST)) {
        source_tier = Tier::HOST;
    } else if (group_slot.has_value(Tier::DISK)) {
        source_tier = Tier::DISK;
    }
    if (source_tier == Tier::NONE) {
        return;
    }

    const std::vector<BlockIdxType> source_blocks = component_group->getBlocks(group_slot, source_tier);
    if (source_blocks.empty()) {
        return;
    }

    if (device_group_tags.empty()
        || !validateDeviceGroupTagsForComponentGroup(component_group->component_group_id, device_group_tags)) {
        RTP_LLM_LOG_WARNING("boundary=producer "
                            "failure=invalid_device_group_mapping component_group_id=%d expected=%zu actual=%zu",
                            component_group->component_group_id,
                            component_group->devicePoolCount(),
                            device_group_tags.size());
        return;
    }

    LoadBackTicket::PendingLoadBackItem pending_item;
    pending_item.node              = path_node;
    pending_item.group_id          = component_group->component_group_id;
    pending_item.path_index        = path_index;
    pending_item.source_tier       = source_tier;
    pending_item.source_blocks     = source_blocks;
    pending_item.device_group_tags = device_group_tags;
    pending_load_back_items.push_back(std::move(pending_item));

    if (source_tier == Tier::HOST) {
        result.host_load_back_blocks++;
        result.load_back_blocks++;
    } else if (source_tier == Tier::DISK) {
        result.disk_load_back_blocks++;
        result.load_back_blocks++;
    }

    RTP_LLM_LOG_DEBUG("planned logical settlement from %s group[%d] node_key=%ld",
                      tierName(source_tier),
                      component_group->component_group_id,
                      path_node->cache_key);
}

bool BlockTreeCache::validateDeviceGroupTagsForComponentGroup(int                             component_group_id,
                                                              const std::vector<std::string>& device_group_tags) const {
    if (!initialized_ || component_group_id < 0
        || static_cast<size_t>(component_group_id) >= device_group_tags_.size()) {
        RTP_LLM_LOG_WARNING("invalid component group mapping request, group=%d", component_group_id);
        return false;
    }
    const std::vector<std::string>& expected = device_group_tags_[static_cast<size_t>(component_group_id)];
    if (device_group_tags != expected) {
        RTP_LLM_LOG_WARNING("device group tag mapping mismatch, group=%d expected=%zu actual=%zu",
                            component_group_id,
                            expected.size(),
                            device_group_tags.size());
        return false;
    }
    return true;
}

bool BlockTreeCache::changeLoadBackStateNolock(TreeNode*         node,
                                               int               group_id,
                                               SlotTransferState from,
                                               SlotTransferState to) {
    const size_t gid = static_cast<size_t>(group_id);
    if (group_id < 0 || node == nullptr || gid >= component_groups_.size() || gid >= node->group_slots.size()) {
        return false;
    }
    const ComponentGroupPtr& group = component_groups_[gid];
    GroupSlot&               slot  = node->group_slots[gid];
    if (group == nullptr || slot.transfer_state != from) {
        return false;
    }
    slot.transfer_state = to;
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

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.node == nullptr || item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()
            || (item.source_tier != Tier::DEVICE && item.source_tier != Tier::HOST && item.source_tier != Tier::DISK)) {
            return false;
        }
        const ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (group == nullptr || static_cast<size_t>(item.group_id) >= item.node->group_slots.size()
            || item.node->group_slots[static_cast<size_t>(item.group_id)].transfer_state != SlotTransferState::IDLE) {
            return false;
        }
        const size_t expected_source_count = item.source_tier == Tier::DEVICE ? group->devicePoolCount() : 1;
        if (item.source_blocks.size() != expected_source_count
            || group->getTopTier(item.node->group_slots[static_cast<size_t>(item.group_id)]) != item.source_tier
            || group->getBlocks(item.node->group_slots[static_cast<size_t>(item.group_id)], item.source_tier)
                   != item.source_blocks) {
            return false;
        }
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        component_groups_[static_cast<size_t>(item.group_id)]->referenceBlocks(
            GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}}, BlockRefType::REQUEST);
    }

    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.source_tier == Tier::DEVICE) {
            continue;
        }
        if (!changeLoadBackStateNolock(
                item.node, item.group_id, SlotTransferState::IDLE, SlotTransferState::LOAD_BACK_PENDING)) {
            abortLoadBackNolock(items, 0);
            return false;
        }
        evictor_.eraseNode(item.node, item.group_id, item.source_tier);
    }
    return true;
}

std::shared_ptr<AsyncContext> BlockTreeCache::commitLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    const LoadBackTicket::PendingLoadBackItems& items = ticket.items_;

    std::vector<ComponentGroupPtr> item_groups;
    item_groups.reserve(items.size());
    for (const LoadBackTicket::PendingLoadBackItem& item : items) {
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()
            || component_groups_[static_cast<size_t>(item.group_id)] == nullptr) {
            abortLoadBackNolock(items, 0);
            RTP_LLM_LOG_WARNING("invalid load-back group while committing, group=%d", item.group_id);
            return nullptr;
        }
        item_groups.push_back(component_groups_[static_cast<size_t>(item.group_id)]);
    }

    LoadBackWorker::TaskPtr task = load_back_worker_.createTask(items, item_groups);
    if (task == nullptr) {
        abortLoadBackNolock(items, 0);
        return nullptr;
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item = items[item_index];
        if (item.source_tier == Tier::DEVICE) {
            continue;
        }
        const GroupBlockSet target_holder{item.group_id, Tier::DEVICE, {item.target_device_blocks}};
        if (!changeLoadBackStateNolock(item.node,
                                      item.group_id,
                                      SlotTransferState::LOAD_BACK_PENDING,
                                      SlotTransferState::LOADING_BACK)) {
            abortLoadBackNolock(items, item_index);
            RTP_LLM_LOG_WARNING(
                "pending-to-loading transition failed, rolled back all %zu load_back items", items.size());
            return nullptr;
        }
        task->item_groups[item_index]->referenceBlocks(target_holder, BlockRefType::REQUEST);
    }

    const bool submitted = task_pool_->submit([this, task]() { runLoadBackTask(task); });
    if (!submitted) {
        abortLoadBackNolock(items, items.size());
        task->context->onTaskComplete(false);
        return task->context;
    }
    return task->context;
}

void BlockTreeCache::abortLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadBackNolock(ticket.items_, 0);
}

void BlockTreeCache::abortLoadBackNolock(const LoadBackTicket::PendingLoadBackItems& items,
                                         size_t                                     prepared_item_count) {
    bool global_refresh_required = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item = items[item_index];
        const size_t gid = static_cast<size_t>(item.group_id);
        if (item.group_id < 0 || gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
            continue;
        }
        const bool fully_prepared = item_index < prepared_item_count;
        if (item.source_tier != Tier::DEVICE && fully_prepared) {
            component_groups_[gid]->unreferenceBlocks(
                GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}}, BlockRefType::REQUEST);
        }

        GroupBlockSet source_set{item.group_id, item.source_tier, {item.source_blocks}};
        if (item.source_tier != Tier::DEVICE) {
            source_set.nodes = {item.node};
        }
        component_groups_[gid]->unreferenceBlocks(source_set, BlockRefType::REQUEST);
        if (item.source_tier != Tier::DEVICE) {
            if (fully_prepared) {
                if (!changeLoadBackStateNolock(
                        item.node, item.group_id, SlotTransferState::LOADING_BACK, SlotTransferState::IDLE)) {
                    RTP_LLM_LOG_WARNING(
                        "loading state mismatch, group=%d source=%s", item.group_id, tierName(item.source_tier));
                } else {
                    evictor_.refreshNode(item.node, item.group_id, item.source_tier, false);
                }
            } else {
                if (!changeLoadBackStateNolock(
                        item.node, item.group_id, SlotTransferState::LOAD_BACK_PENDING, SlotTransferState::IDLE)) {
                    RTP_LLM_LOG_WARNING("reservation state mismatch, "
                                        "group=%d source=%s",
                                        item.group_id,
                                        tierName(item.source_tier));
                }
                evictor_.refreshCandidatesAfterRelease(source_set);
            }
        } else {
            global_refresh_required = true;
        }
    }
    if (global_refresh_required) {
        evictor_.refreshAllCandidates(*tree_);
        checkWatermark();
    }
}

void BlockTreeCache::runLoadBackTask(const LoadBackWorker::TaskPtr& task) {
    if (task == nullptr || task->context == nullptr) {
        RTP_LLM_LOG_ERROR("invalid load-back task");
        return;
    }

    bool prepared = !task->items.empty();
    for (size_t item_index = 0; item_index < task->items.size(); ++item_index) {
        LoadBackWorker::PrepareStatus status = load_back_worker_.prepareTransferItem(*task, item_index);
        if (status == LoadBackWorker::PrepareStatus::NEED_HOST_RECLAIM) {
            const int group_id = task->items[item_index].group_id;
            if (reclaimOneForGroup(group_id, Tier::HOST)) {
                status = load_back_worker_.prepareTransferItem(*task, item_index);
            }
        }
        if (status != LoadBackWorker::PrepareStatus::READY) {
            if (status == LoadBackWorker::PrepareStatus::NEED_HOST_RECLAIM) {
                RTP_LLM_LOG_WARNING("failed to prepare host staging block, group=%d",
                                    task->items[item_index].group_id);
            }
            prepared = false;
        }
    }

    const bool copy_success = load_back_worker_.runTransfer(*task,
                                                            *transfer_dispatcher_,
                                                            metrics_reporter_,
                                                            config_.memory_cache_disk_sync_timeout_ms,
                                                            config_.memory_cache_sync_timeout_ms,
                                                            prepared);

    bool settlement_success = false;
    // Commit the copied batch only while every stateful item still belongs
    // to this load-back operation.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settlement_success = settleLoadBackNolock(*task, copy_success);
    }
    load_back_worker_.completeTask(*task, settlement_success);
}

bool BlockTreeCache::settleLoadBackNolock(LoadBackWorker::Task& task, bool copy_success) {
    const bool has_device_items = std::any_of(
        task.items.begin(), task.items.end(), [](const LoadBackTicket::PendingLoadBackItem& item) {
            return item.source_tier == Tier::DEVICE;
        });
    bool settlement_success = copy_success && task.context != nullptr && !task.context->cancelRequested();
    bool state_settled      = false;
    bool tree_data_mutated  = false;

    if (settlement_success) {
        for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
            const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
            const ComponentGroupPtr&                   group = task.item_groups[item_index];
            if (item.source_tier == Tier::DEVICE) {
                continue;
            }
            const size_t gid = static_cast<size_t>(item.group_id);
            if (item.group_id < 0 || group == nullptr || item.node == nullptr || gid >= item.node->group_slots.size()
                || item.node->group_slots[gid].transfer_state != SlotTransferState::LOADING_BACK) {
                RTP_LLM_LOG_WARNING("completion state mismatch, group=%d", item.group_id);
                settlement_success = false;
                break;
            }
        }
    }

    for (size_t item_index = 0; item_index < task.items.size(); ++item_index) {
        const LoadBackTicket::PendingLoadBackItem& item  = task.items[item_index];
        const ComponentGroupPtr&                   group = task.item_groups[item_index];
        const size_t                               gid   = static_cast<size_t>(item.group_id);
        if (item.group_id < 0 || group == nullptr) {
            continue;
        }

        GroupBlockSet source_protection{item.group_id, item.source_tier, {item.source_blocks}};
        if (item.source_tier != Tier::DEVICE && item.node != nullptr) {
            source_protection.nodes = {item.node};
        }
        group->unreferenceBlocks(source_protection, BlockRefType::REQUEST);

        if (item.source_tier == Tier::DEVICE || item.node == nullptr || gid >= item.node->group_slots.size()) {
            continue;
        }
        GroupSlot& slot = item.node->group_slots[gid];
        if (settlement_success) {
            GroupBlockSet target_holder{item.group_id, Tier::DEVICE, {item.target_device_blocks}};
            group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
            group->referenceBlocks(target_holder, BlockRefType::BLOCK_CACHE);
            group->unreferenceBlocks(target_holder, BlockRefType::REQUEST);
            group->unreferenceBlocks(
                GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}}, BlockRefType::BLOCK_CACHE);
            group->evictFromTier(item.node, slot, item.source_tier);
            task.target_installed[item_index] = true;
            tree_data_mutated                 = true;
            if (!changeLoadBackStateNolock(
                    item.node, item.group_id, SlotTransferState::LOADING_BACK, SlotTransferState::IDLE)) {
                RTP_LLM_LOG_ERROR("exact-state transition failed after preflight, group=%d", item.group_id);
                settlement_success = false;
            } else {
                evictor_.refreshNode(item.node, item.group_id, Tier::DEVICE, true);
                state_settled = true;
            }
            continue;
        }

        // On copy/batch-settlement failure, leave the source data untouched.
        if (!changeLoadBackStateNolock(
                item.node, item.group_id, SlotTransferState::LOADING_BACK, SlotTransferState::IDLE)) {
            RTP_LLM_LOG_WARNING(
                "loading state mismatch, group=%d source=%s", item.group_id, tierName(item.source_tier));
        } else {
            evictor_.refreshNode(item.node, item.group_id, item.source_tier, false);
            state_settled = true;
        }
    }
    if (has_device_items) {
        evictor_.refreshAllCandidates(*tree_);
    }
    if (tree_data_mutated) {
        ++mutation_version_;
    }
    if (has_device_items || state_settled) {
        checkWatermark();
    }
    return settlement_success;
}

bool BlockTreeCache::reclaimOneForGroup(int component_group_id, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= component_groups_.size()) {
        return false;
    }
    auto eviction_move = evictor_.chooseVictim(component_group_id, tier);
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

void BlockTreeCache::settleInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits) {
    for (const DeviceReleaseCredit& credit : release_credits) {
        const auto it = in_flight_device_release_credits_.find(credit.pool);
        if (it == in_flight_device_release_credits_.end() || it->second == 0) {
            RTP_LLM_LOG_ERROR("missing in-flight DEVICE release credit while settling pool=%p block=%d",
                              static_cast<void*>(credit.pool.get()),
                              credit.block);
            continue;
        }
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
        if (move.source_tier != Tier::DEVICE || move.component_group_id < 0) {
            return;
        }
        const size_t gid = static_cast<size_t>(move.component_group_id);
        if (gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
            return;
        }
        const auto&  pools = component_groups_[gid]->devicePools();
        const size_t slot_count = std::min(pools.size(), move.source_blocks.size());
        for (size_t i = 0; i < slot_count; ++i) {
            const auto& pool = pools[i];
            if (!isNullBlockIdx(move.source_blocks[i]) && pool != nullptr
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
        metrics_reporter_.reportEvictionFinished(*plan, results, component_groups_);
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
            try {
                std::lock_guard<std::mutex> lock(mutex_);
                settleInFlightDeviceReleaseCreditsLocked(release_credits);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("DEVICE release-credit settlement failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("DEVICE release-credit settlement failed with unknown exception");
            }
        };
        ScopeRollback<decltype(credit_settlement_action)> credit_settlement_guard(std::move(credit_settlement_action));

        bool completion_succeeded = false;
        bool plan_terminalized    = false;
        bool plan_succeeded       = false;
        bool copy_ok              = copy_results.primary_success;

        CacheKeyType remote_cache_key = 0;
        int          remote_group_id  = -1;
        if (copy_ok && plan.primary.node != nullptr) {
            remote_cache_key = plan.primary.node->cache_key;
            remote_group_id  = plan.primary.component_group_id;
        }

        try {
            std::lock_guard<std::mutex> lock(mutex_);
            try {
                evictor_.complete(*tree_, plan, copy_results);
                completion_succeeded = true;
                plan_terminalized    = true;
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("eviction completion failed; rolling back accepted plan: %s", error.what());
                try {
                    evictor_.rollbackPreparedPlan(plan);
                    plan_terminalized = true;
                } catch (const std::exception& rollback_error) {
                    RTP_LLM_LOG_ERROR("accepted eviction rollback failed: %s", rollback_error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("accepted eviction rollback failed with unknown exception");
                }
            } catch (...) {
                RTP_LLM_LOG_ERROR("eviction completion failed with unknown exception; rolling back "
                                  "accepted plan");
                try {
                    evictor_.rollbackPreparedPlan(plan);
                    plan_terminalized = true;
                } catch (const std::exception& rollback_error) {
                    RTP_LLM_LOG_ERROR("accepted eviction rollback failed: %s", rollback_error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("accepted eviction rollback failed with unknown exception");
                }
            }

            // Credits are accounting-only. The completed or rolled-back evictor plan above owns all
            // pool reference transitions; settlement must never add another decRef.
            credit_settlement_attempted = true;
            try {
                settleInFlightDeviceReleaseCreditsLocked(release_credits);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("DEVICE release-credit settlement failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("DEVICE release-credit settlement failed with unknown exception");
            }

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
                try {
                    checkWatermark();
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_ERROR("post-eviction watermark check failed: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("post-eviction watermark check failed with unknown exception");
                }
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed with unknown exception");
        }
        metrics_reporter_.reportEvictionFinished(plan, copy_results, component_groups_);

        // If an exception escaped before the in-lock settlement attempt, perform that accounting step
        // now. The no-throw guard records one attempt and prevents a duplicate decrement.
        credit_settlement_guard.run();

        if (plan_terminalized && completion_succeeded && copy_ok && config_.enable_remote_cache
            && remote_group_id >= 0) {
            try {
                evictor_.writeRemoteThrough(storage_backend_, remote_cache_key, remote_group_id);
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
        for (const auto& group : component_groups_) {
            if (group == nullptr) {
                continue;
            }
            for (const auto& pool : group->devicePools()) {
                if (pool == nullptr || pool_indices.count(pool.get()) != 0) {
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
        auto group_has_uncovered_deficit = [&](const ComponentGroupPtr& group) {
            if (group == nullptr) {
                return false;
            }
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

        std::vector<bool> unavailable(component_groups_.size(), false);
        while (has_uncovered_deficit()) {
            bool round_progress = false;
            for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
                const auto& group = component_groups_[group_index];
                if (unavailable[group_index] || !group_has_uncovered_deficit(group)) {
                    continue;
                }
                auto eviction_move = evictor_.chooseVictim(group->component_group_id, Tier::DEVICE);
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

        for (auto& group : component_groups_) {
            auto victims = evictor_.chooseWatermarkVictims(*group, tier, wm.ratio);
            for (auto& eviction_move : victims) {
                submitEvictionLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
