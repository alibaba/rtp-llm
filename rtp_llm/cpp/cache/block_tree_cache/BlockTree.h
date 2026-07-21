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

// A node created (not reused) during insertNode, paired with its input index.
struct BlockTreeInsertedNode {
    TreeNode* node{nullptr};
    size_t    input_index{0};
};

// Result of insertNode: which input slots were consumed by newly created nodes.
struct BlockTreeInsertResult {
    TreeNode*                          leaf{nullptr};
    std::vector<BlockTreeInsertedNode> inserted_nodes;
    std::vector<bool>                  inserted_mask;  // size == cache_keys.size()
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
    // Returns the deepest node plus which input slots were consumed by new nodes.
    BlockTreeInsertResult
    insertNode(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);

    // Remove a node from the tree. The node must have no children.
    // The node's parent link is updated accordingly.
    void removeNode(TreeNode* node);

    // Walk up from start_node, removing empty ancestors (no children
    // and all group slots empty). Returns the first ancestor that was not
    // removed (a non-empty node, a node with children, root, or nullptr).
    // group_ids: all group IDs (node deletion
    // only considers these when checking emptiness).
    TreeNode* removeEmptyAncestors(TreeNode* start_node, const std::vector<int>& group_ids);

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

    // Iterate over all live tree nodes. Used by the evictor to re-evaluate
    // candidacy after external refcount changes (e.g. request free).
    const std::vector<std::unique_ptr<TreeNode>>& nodes() const {
        return node_pool_;
    }

private:
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);

    std::unique_ptr<TreeNode>              root_;
    std::vector<std::unique_ptr<TreeNode>> node_pool_;
    int                                    group_slot_count_;
};

}  // namespace rtp_llm
