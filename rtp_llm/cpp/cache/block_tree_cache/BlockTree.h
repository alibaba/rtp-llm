#pragma once

#include <deque>
#include <functional>
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

struct BlockTreeAdoptedNode {
    TreeNode*           node{nullptr};
    std::vector<size_t> group_set_ids;
    std::vector<Tier>   old_top_tiers;
    std::vector<Tier>   new_top_tiers;
};

struct BlockTreeInsertResult {
    std::vector<TreeNode*>            path;
    std::vector<TreeNode*>            inserted_nodes;
    std::vector<BlockTreeAdoptedNode> adopted_nodes;
    // Number of logical GroupSetResources the tree took BLOCK_CACHE ownership of.
    size_t accepted_resource_count{0};
};

struct BlockTreeNodeRangeResult {
    size_t visited{0};
    size_t next_cursor{0};
    size_t tree_size{0};
    bool   cycle_complete{false};
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

    // Diagnostic-only bounded walk over node_pool_[cursor, min(cycle_end_index, size())).
    // The caller must hold BlockTreeCache::mutex_
    BlockTreeNodeRangeResult visitNodeRangeLocked(size_t                                      cursor,
                                                  size_t                                      cycle_end_index,
                                                  size_t                                      max_nodes,
                                                  const std::function<void(const TreeNode&)>& visitor) const;

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
