#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

// Result of a tree traversal for match operations.
struct BlockTreeFindResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};
    // Path from root to matched_node (exclusive of root)
    std::vector<TreeNode*> path;
};

// BlockTree: pure tree topology data structure.
// Manages tree nodes with cache_key-based lookup. No eviction logic.
// The tree owns all node memory. Raw pointers in TreeNode (parent/children)
// are non-owning views into the tree's node pool.
class BlockTree {
public:
    // group_slot_count: number of ComponentGroups (determines group_slots size for new nodes)
    explicit BlockTree(int group_slot_count);
    ~BlockTree();

    // Find the deepest node matching the cache_keys sequence from root.
    BlockTreeFindResult findNode(const CacheKeysType& cache_keys) const;

    // Insert nodes along the cache_keys path starting from parent (nullptr = root).
    // Existing nodes are reused, new nodes are created for unmatched suffix.
    // slots[i] provides the GroupSlot for the i-th node (cache_keys[i]).
    // Returns the deepest node in the path.
    TreeNode*
    insertNode(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);

    // Remove a node from the tree. The node must have no children.
    // The node's parent link is updated accordingly.
    void removeNode(TreeNode* node);

    // Walk up from start_node, removing empty ancestors (no children
    // and all group slots empty). Stops at first non-empty
    // ancestor or root.
    // group_ids: all group IDs (node deletion
    // only considers these when checking emptiness).
    void removeEmptyAncestors(TreeNode* start_node, const std::vector<int>& group_ids);

    // Accessors
    TreeNode* root() const {
        return root_.get();
    }
    int groupSlotCount() const {
        return group_slot_count_;
    }
    size_t nodeCount() const {
        return node_pool_.size();
    }

private:
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);

    std::unique_ptr<TreeNode>              root_;
    std::vector<std::unique_ptr<TreeNode>> node_pool_;
    int                                    group_slot_count_;
};

}  // namespace rtp_llm
