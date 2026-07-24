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
        TreeNode* candidate = it->second;
        if (candidate == nullptr || !isNodeMatchReady(*candidate)) {
            RTP_LLM_LOG_DEBUG("stop matching at depth=%zu, cache_key=%ld, reason=%s",
                              i,
                              cache_keys[i],
                              candidate == nullptr ? "null candidate" : "node not match ready");
            break;
        }
        current               = candidate;
        result.matched_blocks = i + 1;
        result.matched_node   = current;
        result.path.push_back(current);
    }

    return result;
}

bool BlockTree::isNodeMatchReady(const TreeNode& node) const {
    if (node.group_slots.size() != static_cast<size_t>(group_slot_count_)) {
        RTP_LLM_LOG_WARNING("malformed group slot count, node_key=%ld expected=%d actual=%zu",
                            node.cache_key,
                            group_slot_count_,
                            node.group_slots.size());
        return false;
    }
    return std::all_of(node.group_slots.begin(), node.group_slots.end(), [](const GroupSlot& slot) {
        return slot.transfer_state == SlotTransferState::IDLE;
    });
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
            TreeNode* existing_node = it->second;
            if (existing_node == nullptr) {
                RTP_LLM_LOG_WARNING("null child, key=%ld", key);
                break;
            }
            current = existing_node;
            if (current->group_slots.size() != static_cast<size_t>(group_slot_count_)) {
                RTP_LLM_LOG_WARNING("malformed existing node, key=%ld expected=%d actual=%zu",
                                    key,
                                    group_slot_count_,
                                    current->group_slots.size());
                break;
            }
            // An empty per-node input means this topology position carries no
            // group payload. Keep traversing so callers can append a suffix
            // without manufacturing placeholder GroupSlots for every existing
            // prefix node.
            if (slots[i].empty()) {
                continue;
            }
            if (slots[i].size() != static_cast<size_t>(group_slot_count_)) {
                RTP_LLM_LOG_WARNING(
                    "malformed input slots, key=%ld expected=%d actual=%zu", key, group_slot_count_, slots[i].size());
                continue;
            }
            const auto& incoming_slots = slots[i];
            for (size_t gid = 0; gid < static_cast<size_t>(group_slot_count_); ++gid) {
                GroupSlot&       existing     = current->group_slots[gid];
                const GroupSlot& incoming     = incoming_slots[gid];
                const bool       source_valid = !incoming.device_blocks.empty()
                                          && std::all_of(incoming.device_blocks.begin(),
                                                         incoming.device_blocks.end(),
                                                         [](BlockIdxType block) { return !isNullBlockIdx(block); });
                if (!existing.is_empty() || existing.transfer_state != SlotTransferState::IDLE || !source_valid) {
                    continue;
                }
                existing.device_blocks  = incoming.device_blocks;
                existing.host_block     = NULL_BLOCK_IDX;
                existing.disk_slot      = NULL_BLOCK_IDX;
                existing.transfer_state = SlotTransferState::IDLE;
                existing.candidate_meta = {};
                result.adopted_slots.push_back(BlockTreeAdoptedSlot{current, i, static_cast<int>(gid)});
            }
        } else {
            TreeNode* child        = createNode(key, current);
            current->children[key] = child;
            current                = child;
            if (slots[i].size() == static_cast<size_t>(group_slot_count_)) {
                current->group_slots = slots[i];
            } else if (!slots[i].empty()) {
                RTP_LLM_LOG_WARNING(
                    "malformed slot count, key=%ld expected=%d actual=%zu", key, group_slot_count_, slots[i].size());
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

TreeNode* BlockTree::removeEmptyAncestors(TreeNode* start_node, const std::vector<int>& reusable_group_ids) {
    TreeNode* current       = start_node;
    int       removed_count = 0;

    while (current != nullptr && current != root_.get()) {
        // Stop if this node has children
        if (!current->children.empty()) {
            break;
        }

        bool removable = true;
        for (int gid : reusable_group_ids) {
            if (gid < 0 || static_cast<size_t>(gid) >= current->group_slots.size()
                || !current->group_slots[static_cast<size_t>(gid)].is_removable()) {
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
