#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

PressureTransferKey
pressureTransferKey(const TransferDescriptor& desc, const std::string& pool, CacheTransferOperation operation) {
    // Remote descriptors carry a session handle in place of a local pool index.
    return {desc.group_set_id,
            pool,
            operation,
            desc.isRemoteMock() ? Tier::REMOTE : desc.source_tier,
            desc.isRemoteWrite() ? Tier::REMOTE : desc.target_tier};
}

// Match DeviceHostCopyGeometry's group-local layer ordinals and KV/scale
// buffer strides. Compute this immutable geometry once per cache instance.
size_t memberCopyBytes(const GroupSet& group, size_t member) {
    if (member >= group.groupTags().size()) {
        return 0;
    }
    if (group.groupTags().size() == 1) {
        return group.payloadBytes();
    }
    const auto& tag  = group.groupTags()[member];
    const auto& pool = group.devicePools()[member];
    if (!group.usesPhysicalPayloadGeometry() || !pool) {
        return group.topologyPtr()->blockSizeBytesForGroup(tag);
    }
    size_t remaining = group.topologyPtr()->layerIdsForGroup(tag).size();
    size_t bytes     = 0;
    for (const auto& layout : pool->config().memory_layouts) {
        const auto layers = std::min<size_t>(remaining, layout.layer_num);
        bytes += layers * (layout.kv_block_stride_bytes + layout.kv_scale_stride_bytes);
        remaining -= layers;
    }
    return bytes;
}

}  // namespace

BlockTree::BlockTree(std::vector<GroupSetPtr> group_sets): group_sets_(std::move(group_sets)) {
    pressure_state_     = std::make_unique<CachePressureState>();
    const auto add_pool = [this](Tier tier, const IBlockPool* pool) {
        if (pool && pressure_pool_indices_.emplace(pool, pressure_pools_.size()).second) {
            pressure_pools_.push_back({"pool_" + std::to_string(pressure_pools_.size()), tier, pool});
        }
    };
    for (const auto& group : group_sets_) {
        auto& member_bytes = pressure_device_payloads_.emplace_back();
        for (size_t member = 0; member < group->devicePools().size(); ++member) {
            add_pool(Tier::DEVICE, group->devicePools()[member].get());
            member_bytes.push_back(memberCopyBytes(*group, member));
        }
        add_pool(Tier::HOST, group->hostPool().get());
        add_pool(Tier::DISK, group->diskPool().get());
    }
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const auto& group_tags = group_sets_[group_set_id]->groupTags();
        for (size_t member_group_id = 0; member_group_id < group_tags.size(); ++member_group_id) {
            const auto& tag = group_tags[member_group_id];
            const bool  inserted =
                reusable_group_locations_.emplace(tag, ReusableGroupLocation{group_set_id, member_group_id}).second;
            RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate BlockTree reusable tag=%s", tag.c_str());
        }
    }

    root_            = std::make_unique<TreeNode>();
    root_->cache_key = 0;
    root_->parent    = nullptr;
    root_->group_set_resources.resize(group_sets_.size());
}

const ReusableGroupLocation* BlockTree::reusableGroupLocation(std::string_view group_tag) const {
    const auto location_it = reusable_group_locations_.find(std::string(group_tag));
    return location_it == reusable_group_locations_.end() ? nullptr : &location_it->second;
}

BlockTreeNodeRangeResult BlockTree::visitNodeRangeLocked(size_t                                      cursor,
                                                         size_t                                      cycle_end_index,
                                                         size_t                                      max_nodes,
                                                         const std::function<void(const TreeNode&)>& visitor) const {
    BlockTreeNodeRangeResult result;
    result.tree_size = node_pool_.size();

    const size_t end   = std::min(cycle_end_index, node_pool_.size());
    size_t       index = cursor;
    while (index < end && result.visited < max_nodes) {
        visitor(*node_pool_[index]);
        ++index;
        ++result.visited;
    }
    result.next_cursor    = index;
    result.cycle_complete = index >= end;
    return result;
}

BlockTree::~BlockTree() {
    releaseNode(root_.get());
    for (const std::unique_ptr<TreeNode>& node : node_pool_) {
        releaseNode(node.get());
    }
    for (const auto& [_, node] : pending_node_pool_) {
        releaseNode(node.get());
    }
    for (auto& node : node_pool_) {
        node->children.clear();
        node->parent = nullptr;
    }
    root_->children.clear();
}

void BlockTree::releaseNode(TreeNode* node) {
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        GroupSetResource&  resource  = node->group_set_resources[group_set_id];
        if (resource.hasTier(Tier::DEVICE)) {
            const std::vector<BlockIdxType> device_blocks = resource.device_blocks;
            const MultiNodeResource         device_resource{group_set_id, Tier::DEVICE, {{node, device_blocks}}};
            resource.evictFromTier(Tier::DEVICE);
            group_set->unreferenceBlocks(device_resource, BlockTreeRefType::CACHE);
        }
        if (resource.hasTier(Tier::HOST)) {
            group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::HOST, {{node, {resource.host_block}}}},
                                         BlockTreeRefType::CACHE);
            resource.host_block = NULL_BLOCK_IDX;
        }
        if (resource.hasTier(Tier::DISK)) {
            group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::DISK, {{node, {resource.disk_block}}}},
                                         BlockTreeRefType::CACHE);
            resource.disk_block = NULL_BLOCK_IDX;
        }
        resource.transfer_state = GroupSetTransferState::IDLE;
        refreshPressureHostOwner(node, group_set_id);
    }
}

TreeNode* BlockTree::createNode(CacheKeyType key, TreeNode* parent) {
    auto node       = std::make_unique<TreeNode>();
    node->cache_key = key;
    node->parent    = parent;
    node->index     = node_pool_.size();
    node->group_set_resources.resize(group_sets_.size());
    auto* raw = node.get();
    node_pool_.push_back(std::move(node));
    cache_keys_.push_back(key);
    ++key_version_;
    return raw;
}

std::vector<TreeNode*> BlockTree::findNode(const CacheKeysType& cache_keys) const {
    std::vector<TreeNode*> path;
    TreeNode*              current = root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        auto it = current->children.find(cache_keys[i]);
        if (it == current->children.end()) {
            break;
        }
        TreeNode* child = it->second;
        for (const GroupSetResource& resource : child->group_set_resources) {
            if (resource.transfer_state == GroupSetTransferState::IDLE) {
                RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState(),
                                        "BlockTree encountered invalid IDLE multi-tier resource, node_key=%ld",
                                        child->cache_key);
            }
        }
        current = child;
        path.push_back(current);
    }

    return path;
}

CacheKeysType BlockTree::cacheKeys() const {
    return cache_keys_;
}

bool BlockTree::isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const {
    for (const auto& [_, child] : node->children) {
        if (child->group_set_resources[group_set_id].hasTier(tier)) {
            return false;
        }
    }
    return true;
}

BlockTreeInsertResult BlockTree::insertNode(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            bool                                              collect_path,
                                            bool                                              is_resident) {
    BlockTreeInsertResult result;
    if (resources.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/resource size mismatch, keys=%zu resources=%zu", cache_keys.size(), resources.size());
        return result;
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        if (resources[i].size() != group_sets_.size()) {
            RTP_LLM_LOG_WARNING("GroupSetResource mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                group_sets_.size(),
                                resources[i].size());
            return result;
        }
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& resource = resources[i][group_set_id];
            RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState()
                                        && (!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue()),
                                    "BlockTree insert requires an IDLE steady resource and complete DEVICE value: "
                                    "key=%ld group_set_id=%zu state=%d tiers=%zu",
                                    cache_keys[i],
                                    group_set_id,
                                    static_cast<int>(resource.transfer_state),
                                    resource.servingTierCount());
        }
    }

    return insertNodeImpl(cache_keys, resources, collect_path, is_resident);
}

BlockTreeInsertResult BlockTree::insertNodeImpl(const CacheKeysType&                              cache_keys,
                                                const std::vector<std::vector<GroupSetResource>>& resources,
                                                bool                                              collect_path,
                                                bool                                              is_resident) {
    BlockTreeInsertResult result;
    if (collect_path) {
        result.path.reserve(cache_keys.size());
    }

    TreeNode* current                = root_.get();
    size_t    inserted_prefix_length = 0;
    std::vector<std::array<BlockIdList, 2>> replaced_blocks;
    block_tree_cache_detail::ScopeRollback  settle_references([&]() {
        std::map<std::pair<size_t, Tier>, MultiNodeResource> revoked_sources;
        for (const auto& desc : result.revoked_evictions) {
            auto& holder = revoked_sources
                               .try_emplace(std::make_pair(desc.group_set_id, desc.source_tier),
                                            MultiNodeResource{desc.group_set_id, desc.source_tier, {}})
                               .first->second;
            holder.node_blocks.emplace_back(desc.node, desc.source_blocks);
        }
        for (const auto& [key, holder] : revoked_sources) {
            group_sets_[holder.group_set_id]->referenceBlocks(holder, BlockTreeRefType::EVICT);
        }
        for (size_t group_set_id = 0; group_set_id < replaced_blocks.size(); ++group_set_id) {
            for (size_t tier_index = 0; tier_index < 2; ++tier_index) {
                const auto& blocks = replaced_blocks[group_set_id][tier_index];
                if (!blocks.empty()) {
                    group_sets_[group_set_id]->releaseBlocks(
                        tier_index == 0 ? Tier::HOST : Tier::DISK, blocks, BlockTreeRefType::CACHE);
                }
            }
        }
    });

    auto publishTier = [this](TreeNode* node, size_t group_set_id, const GroupSetResource& resource, Tier tier) {
        const MultiNodeResource published{group_set_id, tier, {{node, resource.getBlocks(tier)}}};
        group_sets_[group_set_id]->referenceBlocks(published, BlockTreeRefType::CACHE);
        if (tier == Tier::HOST) {
            refreshPressureHostOwner(node, group_set_id);
        }
    };

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        if (pending_node_pool_.count(key)) {
            result.hard_stop = true;
            break;
        }
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            TreeNode* child = it->second;
            if (is_resident
                && std::any_of(child->group_set_resources.begin(),
                               child->group_set_resources.end(),
                               [](const GroupSetResource& resource) {
                                   return resource.transfer_state != GroupSetTransferState::IDLE
                                          || resource.transfer_detached;
                               })) {
                RTP_LLM_LOG_WARNING("resident insert stopped at busy prefix: key_index=%zu key=%ld", i, key);
                break;
            }
            current                                = child;
            const auto&         incoming_resources = resources[i];
            std::vector<size_t> adopted_group_set_ids;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                GroupSetResource&       existing      = current->group_set_resources[group_set_id];
                const GroupSetResource& incoming      = incoming_resources[group_set_id];
                const Tier              incoming_tier = incoming.getTopTier();
                if (incoming_tier == Tier::NONE) {
                    continue;
                }
                const Tier previous_tier = existing.getTopTier();
                const bool full_device =
                    incoming_tier == Tier::DEVICE && group_sets_[group_set_id]->groupType() == CacheGroupType::FULL;
                if (!full_device
                    && (previous_tier != Tier::NONE || existing.transfer_state != GroupSetTransferState::IDLE)) {
                    continue;
                }
                if (full_device && existing.transfer_state != GroupSetTransferState::IDLE) {
                    referenceNodeForNonOwnerTransfer(current);
                    if (existing.transfer_state == GroupSetTransferState::EVICTING) {
                        result.revoked_evictions.emplace_back(current,
                                                              group_set_id,
                                                              i,
                                                              previous_tier,
                                                              Tier::NONE,
                                                              existing.getBlocks(previous_tier),
                                                              /*initial_owner=*/true);
                    }
                    ++existing.generation;
                    existing.transfer_state = GroupSetTransferState::IDLE;
                    if (previous_tier == Tier::DEVICE) {
                        existing.candidate_meta = {};
                        adopted_group_set_ids.push_back(group_set_id);
                    }
                }
                if (previous_tier == Tier::DEVICE) {
                    continue;
                }
                BlockIdList* old_blocks     = nullptr;
                BlockIdxType previous_block = NULL_BLOCK_IDX;
                if (previous_tier != Tier::NONE) {
                    if (replaced_blocks.empty()) {
                        replaced_blocks.resize(group_sets_.size());
                    }
                    old_blocks = &replaced_blocks[group_set_id][previous_tier == Tier::HOST ? 0 : 1];
                    if (old_blocks->empty()) {
                        old_blocks->reserve(cache_keys.size() - i);
                    }
                    previous_block = previous_tier == Tier::HOST ? existing.host_block : existing.disk_block;
                    if (!is_resident && !current->is_resident) {
                        result.replaced_candidates.push_back({current, group_set_id, previous_tier});
                    }
                }
                // Incoming DEVICE data is already computed; replace ownership without a transfer.
                publishTier(current, group_set_id, incoming, incoming_tier);
                existing.setBlocks(incoming_tier, incoming.getBlocks(incoming_tier));
                if (old_blocks != nullptr) {
                    old_blocks->push_back(previous_block);
                    existing.evictFromTier(previous_tier);
                    if (previous_tier == Tier::HOST) {
                        refreshPressureHostOwner(current, group_set_id);
                    }
                }
                existing.candidate_meta = {};
                ++result.accepted_resource_count;
                adopted_group_set_ids.push_back(group_set_id);
            }
            if (!adopted_group_set_ids.empty()) {
                result.adopted_nodes.emplace_back(current, std::move(adopted_group_set_ids));
            }
        } else {
            TreeNode* child              = createNode(key, current);
            current->children[key]       = child;
            current                      = child;
            current->group_set_resources = resources[i];
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                const GroupSetResource& resource = current->group_set_resources[group_set_id];
                const Tier              tier     = resource.getTopTier();
                if (tier == Tier::NONE) {
                    continue;
                }
                publishTier(current, group_set_id, resource, tier);
                ++result.accepted_resource_count;
            }
            result.inserted_nodes.push_back(current);
        }
        if (is_resident && !current->is_resident) {
            assert(current->parent == root_.get() || current->parent->is_resident);
            current->is_resident = true;
            result.newly_resident_nodes.push_back(current);
        }
        if (is_resident) {
            result.resident_prefix_length = i + 1;
        }
        if (collect_path) {
            result.path.push_back(current);
        }
        refreshPublishedState(current);
        inserted_prefix_length = i + 1;
    }

    settle_references.run();
    RTP_LLM_LOG_DEBUG("keys=%zu created=%zu adopted=%zu tree_nodes=%zu",
                      inserted_prefix_length,
                      result.inserted_nodes.size(),
                      result.adopted_nodes.size(),
                      node_pool_.size());
    return result;
}

void BlockTree::referenceNodeForNonOwnerTransfer(TreeNode* node) {
    RTP_LLM_CHECK(node && node->parent);
    ++node->non_owner_transfer_ref_count;
}

bool BlockTree::isRemovable(TreeNode* node) const {
    return node != nullptr && node != root_.get() && !node->is_resident && node->children.empty()
           && std::all_of(node->group_set_resources.begin(),
                          node->group_set_resources.end(),
                          [](const GroupSetResource& resource) { return resource.is_removable(); });
}

bool BlockTree::canReleaseNode(TreeNode* node) const {
    return isRemovable(node) && node->non_owner_transfer_ref_count == 0;
}

void BlockTree::removeNode(TreeNode* node) {
    for (size_t group_id = 0; group_id < group_sets_.size(); ++group_id) {
        pressure_state_->setHostOwner({group_id, node->cache_key}, std::nullopt);
    }
    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    if (node->parent == nullptr) {
        RTP_LLM_CHECK(node->non_owner_transfer_ref_count == 0);
        pending_node_pool_.erase(node->cache_key);
        return;
    }
    removePublishedKey(node->cache_key);
    node->parent->children.erase(node->cache_key);
    node->parent            = nullptr;
    const size_t index      = node->index;
    const size_t last_index = node_pool_.size() - 1;
    if (index != last_index) {
        std::swap(node_pool_[index], node_pool_[last_index]);
        node_pool_[index]->index = index;
        cache_keys_[index]       = cache_keys_.back();
    }
    node->index = std::numeric_limits<size_t>::max();
    if (!canReleaseNode(node)) {
        pending_node_pool_.emplace(node->cache_key, std::move(node_pool_.back()));
    }
    node_pool_.pop_back();
    cache_keys_.pop_back();
    ++key_version_;
}

void BlockTree::removePendingNodes(const std::unordered_map<TreeNode*, size_t>& nodes) {
    for (const auto& [node, non_owner_transfer_refs] : nodes) {
        RTP_LLM_CHECK(node && node->non_owner_transfer_ref_count >= non_owner_transfer_refs);
        node->non_owner_transfer_ref_count -= non_owner_transfer_refs;
        if (node->parent == nullptr && canReleaseNode(node)) {
            removeNode(node);
        }
    }
}

TreeNode* BlockTree::removeNodeAndEmptyAncestors(TreeNode* node) {
    TreeNode* current       = node;
    size_t    removed_count = 0;
    while (isRemovable(current)) {
        TreeNode* parent = current->parent;
        removeNode(current);
        current = parent;
        ++removed_count;
    }
    if (removed_count != 0) {
        RTP_LLM_LOG_DEBUG("removed %zu nodes", removed_count);
    }
    return current;
}

void BlockTree::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                  const std::vector<std::string>& required_group_tags) {
    std::vector<ReusableGroupLocation> locations;
    if (publisher) {
        if (required_group_tags.empty()) {
            throw std::invalid_argument("cache event publication requires reusable groups");
        }
        for (const auto& tag : required_group_tags) {
            const auto* location = reusableGroupLocation(tag);
            if (!location) {
                throw std::invalid_argument("cache event publication group is absent from BlockTree");
            }
            locations.push_back(*location);
        }
    }
    event_publisher_    = std::move(publisher);
    publication_groups_ = std::move(locations);
    published_groups_.clear();
    // Initialization only. Normal mutations update a single affected key.
    // The publisher starts after attachment and captures this initial state.
    for (const auto& node : node_pool_) {
        refreshPublishedState(node.get());
    }
}

KVCacheSnapshot BlockTree::logicalCacheSnapshot() const {
    KVCacheSnapshot snapshot;
    snapshot.version = publication_version_;
    snapshot.block_groups.insert(published_groups_.begin(), published_groups_.end());
    return snapshot;
}

void BlockTree::removePublishedKey(CacheKeyType key) {
    const auto it = published_groups_.find(key);
    if (event_publisher_ && it != published_groups_.end()) {
        for (const auto group_index : it->second) {
            ++publication_version_;
            (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, key, group_index});
        }
        published_groups_.erase(it);
    }
}

void BlockTree::refreshPublishedState(const TreeNode* node) {
    if (!event_publisher_ || node->parent == nullptr) {
        return;
    }
    const auto          it = published_groups_.find(node->cache_key);
    std::vector<size_t> device_ready_groups;
    for (size_t group_index = 0; group_index < publication_groups_.size(); ++group_index) {
        const auto& location = publication_groups_[group_index];
        const auto& resource = node->group_set_resources.at(location.group_set_id);
        const bool  device_ready =
            !resource.transfer_detached
            && resource.device_blocks.size() == group_sets_[location.group_set_id]->groupTags().size()
            && resource.hasCompleteDeviceValue();
        if (device_ready) {
            device_ready_groups.push_back(group_index);
        }
        const bool was_device_ready =
            it != published_groups_.end() && std::binary_search(it->second.begin(), it->second.end(), group_index);
        if (device_ready != was_device_ready) {
            ++publication_version_;
            (void)event_publisher_->tryPublish(
                {device_ready ? KVCacheEventType::BLOCK_ADD : KVCacheEventType::BLOCK_DELETE,
                 node->cache_key,
                 group_index});
        }
    }
    if (!device_ready_groups.empty()) {
        published_groups_[node->cache_key] = std::move(device_ready_groups);
    } else if (it != published_groups_.end()) {
        published_groups_.erase(it);
    }
}

CachePressureState& BlockTree::pressureState() const {
    return *pressure_state_;
}

const std::string& BlockTree::pressurePoolId(const IBlockPool* pool) const {
    return pressure_pools_.at(pressure_pool_indices_.at(pool)).id;
}

void BlockTree::refreshPressureHostOwner(const TreeNode* node, size_t group_id) {
    const auto& pool = group_sets_.at(group_id)->hostPool();
    if (!pool) {
        return;
    }
    const auto& resource = node->group_set_resources.at(group_id);
    if (resource.transfer_detached || !resource.hasTier(Tier::HOST)) {
        pressure_state_->setHostOwner({group_id, node->cache_key}, std::nullopt);
    } else {
        pressure_state_->setHostOwner({group_id, node->cache_key},
                                      PressureBlock{pressurePoolId(pool.get()), resource.host_block});
    }
}

// Every endpoint is reported in its own physical pool units (packed HOST/DISK
// record versus individual DEVICE member blocks). No request-derived keys.
void BlockTree::recordPressurePublication(CacheTransferOperation operation, const TransferDescriptor& desc) {
    const auto& group = group_sets_.at(desc.group_set_id);
    for (Tier tier : {desc.source_tier, desc.target_tier}) {
        const auto record = [&](const IBlockPool* pool, uint64_t blocks) {
            if (pool && blocks) {
                pressure_state_->recordPublication(pressureTransferKey(desc, pressurePoolId(pool), operation),
                                                   blocks,
                                                   blocks * pool->blockSizeBytes());
            }
        };
        if (tier == Tier::DEVICE) {
            const auto& blocks = desc.blocksAt(tier);
            for (size_t i = 0; i < group->devicePools().size() && i < blocks.size(); ++i) {
                if (!isNullBlockIdx(blocks[i])) {
                    record(group->devicePools()[i].get(), 1);
                }
            }
        } else if (tier == Tier::HOST) {
            record(group->hostPool().get(), desc.blocksAt(tier).size());
        } else if (tier == Tier::DISK) {
            record(group->diskPool().get(), desc.blocksAt(tier).size());
        }
    }
}

void BlockTree::recordPressureCopy(CacheTransferOperation operation,
                                   DescriptorView         descriptors,
                                   const TransferResult&  result) {
    recordPressureResult(operation, descriptors, result, true);
}

void BlockTree::recordPressureWorkflowFailure(CacheTransferOperation operation,
                                              DescriptorView         descriptors,
                                              const TransferResult&  result) {
    if (!result.ok()) {
        try {
            recordPressureResult(operation, descriptors, result, false);
        } catch (...) {
            workflow_evidence_complete_.store(false);
            RTP_LLM_LOG_ERROR("cache pressure workflow failure could not be recorded");
        }
    }
}

void BlockTree::recordPressureResult(CacheTransferOperation operation,
                                     DescriptorView         descriptors,
                                     const TransferResult&  result,
                                     bool                   physical) {
    // Aggregate once per logical operation/key so failed_operations is not a block count.
    struct CopyTotal {
        uint64_t blocks{0};
        uint64_t bytes{0};
        bool     failed{false};
    };
    std::map<PressureTransferKey, CopyTotal> totals;
    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto& desc   = descriptors[index];
        const auto& group  = group_sets_.at(desc.group_set_id);
        const bool  copied = result.blockError(index) == CacheCopyError::NONE;
        if (!physical && copied) {
            continue;
        }
        for (Tier tier : {desc.source_tier, desc.target_tier}) {
            const auto add = [&](const IBlockPool* pool, uint64_t blocks, size_t bytes_per_block) {
                if (!pool) {
                    return;
                }
                auto& total = totals[pressureTransferKey(desc, pressurePoolId(pool), operation)];
                if (copied) {
                    total.blocks += blocks;
                    total.bytes += blocks * bytes_per_block;
                } else {
                    total.failed = true;
                }
            };
            if (tier == Tier::DEVICE) {
                const auto& blocks = desc.blocksAt(tier);
                for (size_t i = 0; i < group->devicePools().size() && i < blocks.size(); ++i) {
                    if (!isNullBlockIdx(blocks[i])) {
                        add(group->devicePools()[i].get(), 1, pressure_device_payloads_[desc.group_set_id][i]);
                    }
                }
            } else if (tier == Tier::HOST) {
                const bool disk_io = desc.source_tier == Tier::DISK || desc.target_tier == Tier::DISK;
                const auto bytes =
                    disk_io && group->diskPool() ? group->diskPool()->strideBytes() : group->payloadBytes();
                add(group->hostPool().get(), desc.blocksAt(tier).size(), bytes);
            } else if (tier == Tier::DISK) {
                if (const auto& disk = group->diskPool()) {
                    add(disk.get(), desc.blocksAt(tier).size(), disk->strideBytes());
                }
            }
        }
    }
    for (const auto& [key, total] : totals) {
        if (physical) {
            pressure_state_->recordCopy(key, total.blocks, total.bytes, !total.failed);
        } else {
            pressure_state_->recordWorkflowFailure(key);
        }
    }
}

}  // namespace rtp_llm
