#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

struct BlockTreeFindResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};
    std::vector<TreeNode*> path;
};

struct BlockTreeInsertedNode {
    TreeNode* node{nullptr};
    size_t    input_index{0};
};

struct BlockTreeAdoptedResource {
    TreeNode* node{nullptr};
    size_t    input_index{0};
    size_t    group_set_id{0};
};

struct BlockTreeInsertResult {
    TreeNode*                             leaf{nullptr};
    std::vector<BlockTreeInsertedNode>    inserted_nodes;
    std::vector<BlockTreeAdoptedResource> adopted_resources;
    std::vector<bool>                     inserted_mask;
};

class BlockTree {
public:
    explicit BlockTree(size_t group_set_resource_count);
    ~BlockTree();

    BlockTreeFindResult findNode(const CacheKeysType& cache_keys) const;

    BlockTreeInsertResult insertNode(TreeNode*                                         parent,
                                     const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources);

    void removeNode(TreeNode* node);

    TreeNode* removeEmptyAncestors(TreeNode* start_node, const std::vector<size_t>& group_set_ids);

    TreeNode* root() const {
        return root_.get();
    }
    size_t groupSetResourceCount() const {
        return group_set_resource_count_;
    }
    size_t nodeCount() const {
        return node_pool_.size();
    }

    const std::vector<std::unique_ptr<TreeNode>>& nodes() const {
        return node_pool_;
    }

private:
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);

    std::unique_ptr<TreeNode>              root_;
    std::vector<std::unique_ptr<TreeNode>> node_pool_;
    size_t                                 group_set_resource_count_;
};

}  // namespace rtp_llm
