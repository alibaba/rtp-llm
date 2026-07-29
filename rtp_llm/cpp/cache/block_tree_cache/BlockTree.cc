#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTree::BlockTree(size_t group_set_resource_count): group_set_resource_count_(group_set_resource_count) {
    root_            = std::make_unique<TreeNode>();
    root_->cache_key = 0;
    root_->parent    = nullptr;
    root_->group_set_resources.resize(group_set_resource_count);
}

BlockTree::~BlockTree() {
    for (auto& node : node_pool_) {
        node->children.clear();
        node->parent = nullptr;
    }
    root_->children.clear();
}

TreeNode* BlockTree::createNode(CacheKeyType key, TreeNode* parent) {
    auto node       = std::make_unique<TreeNode>();
    node->cache_key = key;
    node->parent    = parent;
    node->group_set_resources.resize(group_set_resource_count_);
    auto* raw = node.get();
    node_pool_.push_back(std::move(node));
    return raw;
}

BlockTreeFindResult BlockTree::findNode(const CacheKeysType& cache_keys) const {
    BlockTreeFindResult result;
    TreeNode*           current = root_.get();

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
        current               = child;
        result.matched_blocks = i + 1;
        result.matched_node   = current;
        result.path.push_back(current);
    }

    return result;
}

BlockTreeInsertResult BlockTree::insertNode(TreeNode*                                         parent,
                                            const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTreeInsertResult result;
    result.inserted_mask.assign(cache_keys.size(), false);
    TreeNode* current = parent ? parent : root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            current = it->second;
            const auto& incoming_resources = resources[i];
            for (size_t group_set_id = 0; group_set_id < group_set_resource_count_; ++group_set_id) {
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
                result.adopted_resources.push_back(BlockTreeAdoptedResource{current, i, group_set_id});
            }
        } else {
            TreeNode* child        = createNode(key, current);
            current->children[key] = child;
            current                = child;
            current->group_set_resources = resources[i];
            result.inserted_mask[i] = true;
            result.inserted_nodes.push_back(BlockTreeInsertedNode{current, i});
        }
    }

    result.leaf = current;
    return result;
}

void BlockTree::removeNode(TreeNode* node) {
    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    TreeNode* parent = node->parent;
    parent->children.erase(node->cache_key);
    node->parent = nullptr;

    auto it = std::find_if(node_pool_.begin(), node_pool_.end(), [node](const std::unique_ptr<TreeNode>& ptr) {
        return ptr.get() == node;
    });
    node_pool_.erase(it);
}

TreeNode* BlockTree::removeEmptyAncestors(TreeNode* start_node, const std::vector<size_t>& reusable_group_set_ids) {
    TreeNode* current       = start_node;
    int       removed_count = 0;

    while (current != root_.get()) {
        if (!current->children.empty()) {
            break;
        }

        bool removable = true;
        for (size_t group_set_id : reusable_group_set_ids) {
            if (!current->group_set_resources[group_set_id].is_removable()) {
                removable = false;
                break;
            }
        }

        if (!removable) {
            break;
        }

        TreeNode* parent = current->parent;
        removeNode(current);
        current = parent;
        removed_count++;
    }
    if (removed_count > 0) {
        RTP_LLM_LOG_DEBUG("removed %d empty ancestors", removed_count);
    }
    return current;
}

}  // namespace rtp_llm
