#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTree::BlockTree(int group_slot_count): group_slot_count_(group_slot_count) {
    root_            = std::make_unique<TreeNode>();
    root_->cache_key = 0;
    root_->parent    = nullptr;
    root_->group_slots.resize(static_cast<size_t>(group_slot_count));
}

BlockTree::~BlockTree() {
    // node_pool_ cleanup happens automatically via unique_ptr
    // Clear children maps to avoid dangling raw pointers during destruction
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
    node->group_slots.resize(static_cast<size_t>(group_slot_count_));
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
        current               = it->second;
        result.matched_blocks = i + 1;
        result.matched_node   = current;
        result.path.push_back(current);
    }

    return result;
}

BlockTreeInsertResult BlockTree::insertNode(TreeNode*                                  parent,
                                           const CacheKeysType&                       cache_keys,
                                           const std::vector<std::vector<GroupSlot>>& slots) {
    BlockTreeInsertResult result;
    if (cache_keys.empty()) {
        result.leaf = parent ? parent : root_.get();
        return result;
    }
    RTP_LLM_CHECK_WITH_INFO(slots.size() == cache_keys.size(),
                            "BlockTree::insertNode: slots.size() must equal cache_keys.size()");

    result.inserted_mask.assign(cache_keys.size(), false);
    TreeNode* current = parent ? parent : root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            // Existing node — move to it, do NOT overwrite group_slots
            current = it->second;
        } else {
            // Create new child and assign group_slots[i]
            TreeNode* child        = createNode(key, current);
            current->children[key] = child;
            current                = child;
            if (!slots[i].empty()) {
                current->group_slots = slots[i];
            }
            result.inserted_mask[i] = true;
            result.inserted_nodes.push_back(BlockTreeInsertedNode{current, i});
        }
    }

    result.leaf = current;
    return result;
}

void BlockTree::removeNode(TreeNode* node) {
    if (node == nullptr || node == root_.get()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(node->children.empty(), "BlockTree::removeNode called on node with children");

    RTP_LLM_LOG_DEBUG(
        "BlockTree::removeNode: removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    // Remove from parent's children map
    TreeNode* parent = node->parent;
    if (parent != nullptr) {
        parent->children.erase(node->cache_key);
    }
    // Nullify parent pointer so callers can detect deletion.
    node->parent = nullptr;

    // Find and remove from node_pool_
    auto it = std::find_if(node_pool_.begin(), node_pool_.end(), [node](const std::unique_ptr<TreeNode>& ptr) {
        return ptr.get() == node;
    });
    if (it != node_pool_.end()) {
        node_pool_.erase(it);
    }
}

TreeNode* BlockTree::removeEmptyAncestors(TreeNode* start_node, const std::vector<int>& reusable_group_ids) {
    TreeNode* current       = start_node;
    int       removed_count = 0;

    while (current != nullptr && current != root_.get()) {
        // Stop if this node has children
        if (!current->children.empty()) {
            break;
        }

        // Check if any REUSABLE group has data
        bool has_reusable_data = false;
        for (int gid : reusable_group_ids) {
            if (gid >= 0 && static_cast<size_t>(gid) < current->group_slots.size()) {
                if (!current->group_slots[static_cast<size_t>(gid)].is_empty()) {
                    has_reusable_data = true;
                    break;
                }
            }
        }

        if (has_reusable_data) {
            break;
        }

        // This node is empty — remove it
        TreeNode* parent = current->parent;
        removeNode(current);
        current = parent;
        removed_count++;
    }
    if (removed_count > 0) {
        RTP_LLM_LOG_DEBUG("BlockTree::removeEmptyAncestors: removed %d empty ancestors", removed_count);
    }
    return current;
}

}  // namespace rtp_llm
