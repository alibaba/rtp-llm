#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTree::BlockTree(std::vector<GroupSetPtr> group_sets): group_sets_(std::move(group_sets)) {
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

    auto publishTier = [this](TreeNode* node, size_t group_set_id, const GroupSetResource& resource, Tier tier) {
        const MultiNodeResource published{group_set_id, tier, {{node, resource.getBlocks(tier)}}};
        group_sets_[group_set_id]->referenceBlocks(published, BlockTreeRefType::CACHE);
    };

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
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
            bool full_path_ready = true;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                if (group_sets_[group_set_id]->groupType() != CacheGroupType::FULL) {
                    continue;
                }
                const GroupSetResource& incoming = resources[i][group_set_id];
                if (!incoming.hasTier(Tier::DEVICE)) {
                    continue;
                }
                const GroupSetResource& existing  = child->group_set_resources[group_set_id];
                const bool              can_reuse = existing.isValidSteadyState() && existing.hasCompleteDeviceValue();
                const bool              can_adopt = existing.is_removable();
                if (!can_reuse && !can_adopt) {
                    RTP_LLM_LOG_WARNING("event=block_tree_insert_hard_stop key_index=%zu key=%ld "
                                        "group_set_id=%zu existing_tier=%s transfer_state=%d serving_tiers=%zu",
                                        i,
                                        key,
                                        group_set_id,
                                        tierName(existing.getTopTier()),
                                        static_cast<int>(existing.transfer_state),
                                        existing.servingTierCount());
                    full_path_ready = false;
                    break;
                }
            }
            if (!full_path_ready) {
                if (collect_path) {
                    current = child;
                    result.path.push_back(current);
                    for (size_t path_index = i + 1; path_index < cache_keys.size(); ++path_index) {
                        auto path_it = current->children.find(cache_keys[path_index]);
                        if (path_it == current->children.end()) {
                            break;
                        }
                        current = path_it->second;
                        result.path.push_back(current);
                    }
                }
                break;
            }

            current                                = child;
            const auto&         incoming_resources = resources[i];
            std::vector<size_t> adopted_group_set_ids;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                GroupSetResource&       existing      = current->group_set_resources[group_set_id];
                const GroupSetResource& incoming      = incoming_resources[group_set_id];
                const Tier              incoming_tier = incoming.getTopTier();
                if (!existing.is_empty() || existing.transfer_state != GroupSetTransferState::IDLE
                    || incoming_tier == Tier::NONE) {
                    continue;
                }
                existing.setBlocks(incoming_tier, incoming.getBlocks(incoming_tier));
                existing.transfer_state = GroupSetTransferState::IDLE;
                existing.candidate_meta = {};
                publishTier(current, group_set_id, existing, incoming_tier);
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

    RTP_LLM_LOG_DEBUG("keys=%zu created=%zu adopted=%zu tree_nodes=%zu",
                      inserted_prefix_length,
                      result.inserted_nodes.size(),
                      result.adopted_nodes.size(),
                      node_pool_.size());
    return result;
}

bool BlockTree::isRemovable(TreeNode* node) const {
    return node != root_.get() && !node->is_resident && node->children.empty()
           && std::all_of(node->group_set_resources.begin(),
                          node->group_set_resources.end(),
                          [](const GroupSetResource& resource) { return resource.is_removable(); });
}

void BlockTree::removeNode(TreeNode* node) {
    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    removePublishedKey(node->cache_key);
    node->parent->children.erase(node->cache_key);
    const size_t index      = node->index;
    const size_t last_index = node_pool_.size() - 1;
    if (index != last_index) {
        std::swap(node_pool_[index], node_pool_[last_index]);
        node_pool_[index]->index = index;
    }
    node_pool_.pop_back();
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
    published_keys_.clear();
    // Initialization only. Normal mutations update a single affected key.
    // The publisher starts after attachment and reconciles this initial snapshot.
    for (const auto& node : node_pool_) {
        refreshPublishedState(node.get());
    }
}

KVCacheSnapshot BlockTree::logicalCacheSnapshot() const {
    KVCacheSnapshot snapshot;
    snapshot.version = publication_version_;
    snapshot.block_keys.assign(published_keys_.begin(), published_keys_.end());
    std::sort(snapshot.block_keys.begin(), snapshot.block_keys.end());
    return snapshot;
}

void BlockTree::removePublishedKey(CacheKeyType key) {
    if (event_publisher_ && published_keys_.erase(key)) {
        ++publication_version_;
        (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, static_cast<int64_t>(key), 0});
    }
}

void BlockTree::refreshPublishedState(const TreeNode* node) {
    if (!event_publisher_ || node == root_.get()) {
        return;
    }
    for (const auto& location : publication_groups_) {
        const auto& resource = node->group_set_resources.at(location.group_set_id);
        if (resource.transfer_detached || resource.transfer_state != GroupSetTransferState::IDLE
            || resource.device_blocks.size() != group_sets_[location.group_set_id]->groupTags().size()
            || !resource.hasCompleteDeviceValue()) {
            removePublishedKey(node->cache_key);
            return;
        }
    }
    if (published_keys_.insert(node->cache_key).second) {
        ++publication_version_;
        (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_ADD, static_cast<int64_t>(node->cache_key), 0});
    }
}

}  // namespace rtp_llm
