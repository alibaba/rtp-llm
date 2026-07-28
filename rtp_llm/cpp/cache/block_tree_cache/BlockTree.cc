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
        TreeNode* candidate = it->second;
        if (candidate == nullptr) {
            RTP_LLM_LOG_DEBUG("stop matching at depth=%zu, cache_key=%ld, reason=null candidate", i, cache_keys[i]);
            break;
        }
        // Transfer-state usability is judged per group by the MatchValidators;
        // the tree walk only enforces structural invariants so one busy group
        // resource cannot truncate the whole topological path.
        validateNodeInvariants(*candidate);
        current               = candidate;
        result.matched_blocks = i + 1;
        result.matched_node   = current;
        result.path.push_back(current);
    }

    return result;
}

void BlockTree::validateNodeInvariants(const TreeNode& node) const {
    RTP_LLM_CHECK_WITH_INFO(node.group_set_resources.size() == group_set_resource_count_,
                            "BlockTree node resource count mismatch: node_key=%ld expected=%zu actual=%zu",
                            node.cache_key,
                            group_set_resource_count_,
                            node.group_set_resources.size());
    for (const GroupSetResource& resource : node.group_set_resources) {
        if (resource.transfer_state == GroupSetTransferState::IDLE) {
            RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState(),
                                    "BlockTree encountered invalid IDLE multi-tier resource, node_key=%ld",
                                    node.cache_key);
        }
    }
}

BlockTreeInsertResult BlockTree::insertNode(TreeNode*                                         parent,
                                            const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTreeInsertResult result;
    if (cache_keys.empty()) {
        result.leaf = parent ? parent : root_.get();
        return result;
    }
    RTP_LLM_CHECK_WITH_INFO(resources.size() == cache_keys.size(),
                            "BlockTree::insertNode: resources.size() must equal cache_keys.size()");
    for (size_t i = 0; i < resources.size(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(resources[i].empty() || resources[i].size() == group_set_resource_count_,
                                "BlockTree input resource count mismatch: key=%ld expected=%zu actual=%zu",
                                cache_keys[i],
                                group_set_resource_count_,
                                resources[i].size());
    }

    result.inserted_mask.assign(cache_keys.size(), false);
    TreeNode* current = parent ? parent : root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            TreeNode* existing_node = it->second;
            if (existing_node == nullptr) {
                RTP_LLM_LOG_WARNING("null child, key=%ld", key);
                break;
            }
            current = existing_node;
            RTP_LLM_CHECK_WITH_INFO(current->group_set_resources.size() == group_set_resource_count_,
                                    "BlockTree existing node resource count mismatch: key=%ld expected=%zu actual=%zu",
                                    key,
                                    group_set_resource_count_,
                                    current->group_set_resources.size());
            // An empty per-node input means this topology position carries no
            // resource payload. Keep traversing so callers can append a suffix
            // without manufacturing placeholder GroupSetResources for every existing
            // prefix node.
            if (resources[i].empty()) {
                continue;
            }
            const auto& incoming_resources = resources[i];
            for (size_t group_set_id = 0; group_set_id < group_set_resource_count_; ++group_set_id) {
                GroupSetResource&       existing     = current->group_set_resources[group_set_id];
                const GroupSetResource& incoming     = incoming_resources[group_set_id];
                const bool              source_valid = !incoming.device_blocks.empty()
                                          && std::all_of(incoming.device_blocks.begin(),
                                                         incoming.device_blocks.end(),
                                                         [](BlockIdxType block) { return !isNullBlockIdx(block); });
                if (!existing.is_empty() || existing.transfer_state != GroupSetTransferState::IDLE || !source_valid) {
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
            if (resources[i].size() == group_set_resource_count_) {
                current->group_set_resources = resources[i];
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

    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

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

TreeNode* BlockTree::removeEmptyAncestors(TreeNode* start_node, const std::vector<size_t>& reusable_group_set_ids) {
    TreeNode* current       = start_node;
    int       removed_count = 0;

    while (current != nullptr && current != root_.get()) {
        // Stop if this node has children
        if (!current->children.empty()) {
            break;
        }

        bool removable = true;
        for (size_t group_set_id : reusable_group_set_ids) {
            if (group_set_id >= current->group_set_resources.size()
                || !current->group_set_resources[group_set_id].is_removable()) {
                removable = false;
                break;
            }
        }

        if (!removable) {
            break;
        }

        // This node is empty — remove it
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
