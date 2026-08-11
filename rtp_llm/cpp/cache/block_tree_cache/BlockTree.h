#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTreeEvictor;

struct ReusableGroupLocation {
    size_t group_set_id{0};
    size_t member_group_id{0};
};

using ReusableGroupLocations = std::unordered_map<size_t, ReusableGroupLocation>;

struct BlockTreeInsertResult {
    std::vector<TreeNode*>                                 path;
    std::vector<TreeNode*>                                 inserted_nodes;
    std::vector<std::pair<TreeNode*, std::vector<size_t>>> adopted_nodes;
    // Number of logical GroupSetResources the tree took BLOCK_CACHE ownership of.
    size_t accepted_resource_count{0};
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets);
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool                                              collect_path);

    bool      isRemovable(TreeNode* node) const;
    TreeNode* removeNodeAndEmptyAncestors(TreeNode* node);

    TreeNode* root() const {
        return root_.get();
    }
    const std::vector<GroupSetPtr>& groupSets() const {
        return group_sets_;
    }
    const ReusableGroupLocation* reusableGroupLocation(size_t group_id) const;
    size_t                       reusableGroupCount() const {
        return reusable_group_locations_.size();
    }
    size_t size() const {
        return node_pool_.size();
    }

private:
    friend class BlockTreeEvictor;

    BlockTreeInsertResult insertNodeImpl(const CacheKeysType&                              cache_keys,
                                         const std::vector<std::vector<GroupSetResource>>& resources,
                                         bool                                              enable_hard_stop,
                                         bool                                              collect_path);

    void      removeNode(TreeNode* node);
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);
    void      releaseNode(TreeNode* node);

    std::vector<GroupSetPtr>              group_sets_;
    ReusableGroupLocations                reusable_group_locations_;
    std::unique_ptr<TreeNode>             root_;
    std::deque<std::unique_ptr<TreeNode>> node_pool_;
};

}  // namespace rtp_llm
