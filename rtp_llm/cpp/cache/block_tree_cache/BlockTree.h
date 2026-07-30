#pragma once

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

struct BlockTreeInsertResult {
    std::vector<TreeNode*> inserted_nodes;
    std::vector<std::pair<TreeNode*, std::vector<size_t>>> adopted_nodes;
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets);
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources);

    void removeNode(TreeNode* node);

    TreeNode* removeEmptyAncestors(TreeNode* start_node, const std::vector<size_t>& group_set_ids);

    TreeNode* root() const {
        return root_.get();
    }
    const std::vector<GroupSetPtr>& groupSets() const {
        return group_sets_;
    }
    const std::vector<std::unique_ptr<TreeNode>>& nodes() const {
        return node_pool_;
    }

private:
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);
    void      releaseNodeHolds(TreeNode* node);

    std::vector<GroupSetPtr>               group_sets_;
    std::unique_ptr<TreeNode>              root_;
    std::vector<std::unique_ptr<TreeNode>> node_pool_;
};

}  // namespace rtp_llm
