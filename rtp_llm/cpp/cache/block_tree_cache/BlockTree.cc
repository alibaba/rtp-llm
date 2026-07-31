#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTree::BlockTree(std::vector<GroupSetPtr> group_sets): group_sets_(std::move(group_sets)) {
    root_            = std::make_unique<TreeNode>();
    root_->cache_key = 0;
    root_->parent    = nullptr;
    root_->group_set_resources.resize(group_sets_.size());
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
            group_set->unreferenceBlocks(
                MultiNodeResource{group_set_id, Tier::DEVICE, {resource.device_blocks}},
                BlockRefType::BLOCK_CACHE);
            std::fill(resource.device_blocks.begin(), resource.device_blocks.end(), NULL_BLOCK_IDX);
        }
        if (resource.hasTier(Tier::HOST)) {
            group_set->unreferenceBlocks(
                MultiNodeResource{group_set_id, Tier::HOST, {{resource.host_block}}},
                BlockRefType::BLOCK_CACHE);
            resource.host_block = NULL_BLOCK_IDX;
        }
        if (resource.hasTier(Tier::DISK)) {
            group_set->unreferenceBlocks(
                MultiNodeResource{group_set_id, Tier::DISK, {{resource.disk_slot}}},
                BlockRefType::BLOCK_CACHE);
            resource.disk_slot = NULL_BLOCK_IDX;
        }
        resource.transfer_state = GroupSetTransferState::IDLE;
    }
}

TreeNode* BlockTree::createNode(CacheKeyType key, TreeNode* parent) {
    auto node       = std::make_unique<TreeNode>();
    node->cache_key = key;
    node->parent    = parent;
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
    const GroupSetResource& resource = node->group_set_resources[group_set_id];
    if (!(tier == Tier::DEVICE ? resource.hasCompleteDeviceValue() : resource.hasTier(tier))) {
        return false;
    }

    for (const auto& [_, child] : node->children) {
        if (child->group_set_resources[group_set_id].hasTier(tier)) {
            return false;
        }
    }
    return true;
}

BlockTreeInsertResult BlockTree::insertNode(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources) {
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
            const GroupSetPtr&      group_set = group_sets_[group_set_id];
            const GroupSetResource& resource  = resources[i][group_set_id];
            RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState()
                                        && (!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue()),
                                    "BlockTree insert requires an IDLE steady resource and complete DEVICE value: "
                                    "key=%ld group_set_id=%zu state=%d tiers=%zu expected_width=%zu actual_width=%zu",
                                    cache_keys[i],
                                    group_set_id,
                                    static_cast<int>(resource.transfer_state),
                                    resource.servingTierCount(),
                                    group_set->devicePools().size(),
                                    resource.device_blocks.size());
        }
    }

    TreeNode* current = root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            current = it->second;
            const auto& incoming_resources = resources[i];
            std::vector<size_t> adopted_group_set_ids;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                GroupSetResource&       existing     = current->group_set_resources[group_set_id];
                const GroupSetResource& incoming     = incoming_resources[group_set_id];
                if (!existing.is_empty() || existing.transfer_state != GroupSetTransferState::IDLE
                    || !incoming.hasTier(Tier::DEVICE)) {
                    continue;
                }
                existing.device_blocks  = incoming.device_blocks;
                existing.host_block     = NULL_BLOCK_IDX;
                existing.disk_slot      = NULL_BLOCK_IDX;
                existing.transfer_state = GroupSetTransferState::IDLE;
                existing.candidate_meta = {};
                if (existing.hasCompleteDeviceValue()) {
                    group_sets_[group_set_id]->referenceBlocks(
                        MultiNodeResource{group_set_id, Tier::DEVICE, {existing.device_blocks}},
                        BlockRefType::BLOCK_CACHE);
                }
                adopted_group_set_ids.push_back(group_set_id);
            }
            if (!adopted_group_set_ids.empty()) {
                result.adopted_nodes.emplace_back(current, std::move(adopted_group_set_ids));
            }
        } else {
            TreeNode* child        = createNode(key, current);
            current->children[key] = child;
            current                = child;
            current->group_set_resources = resources[i];
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                const GroupSetPtr& group_set = group_sets_[group_set_id];
                GroupSetResource&  resource  = current->group_set_resources[group_set_id];
                if (resource.hasCompleteDeviceValue()) {
                    group_set->referenceBlocks(
                        MultiNodeResource{group_set_id, Tier::DEVICE, {resource.device_blocks}},
                        BlockRefType::BLOCK_CACHE);
                }
            }
            result.inserted_nodes.push_back(current);
        }
    }

    RTP_LLM_LOG_DEBUG("keys=%zu created=%zu adopted=%zu tree_nodes=%zu",
                      cache_keys.size(),
                      result.inserted_nodes.size(),
                      result.adopted_nodes.size(),
                      node_pool_.size());
    return result;
}

bool BlockTree::isRemovable(TreeNode* node) const {
    return node != root_.get() && node->children.empty()
           && std::all_of(node->group_set_resources.begin(),
                          node->group_set_resources.end(),
                          [](const GroupSetResource& resource) { return resource.is_removable(); });
}

TreeNode* BlockTree::removeNodeAndEmptyAncestors(TreeNode* node) {
    TreeNode* current       = node;
    int       removed_count = 0;
    while (isRemovable(current)) {
        TreeNode* parent = current->parent;
        RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", current->cache_key, node_pool_.size());
        parent->children.erase(current->cache_key);
        current->parent = nullptr;
        auto it = std::find_if(node_pool_.begin(), node_pool_.end(), [current](const std::unique_ptr<TreeNode>& ptr) {
            return ptr.get() == current;
        });
        node_pool_.erase(it);
        current = parent;
        removed_count++;
    }
    if (removed_count > 0) {
        RTP_LLM_LOG_DEBUG("removed %d nodes", removed_count);
    }
    return current;
}

}  // namespace rtp_llm
