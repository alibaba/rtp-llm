#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTreeEvictor;

struct ReusableGroupLocation {
    size_t group_set_id{0};
    size_t member_group_id{0};
};

using ReusableGroupLocations = std::unordered_map<std::string, ReusableGroupLocation>;

struct BlockTreeInsertResult {
    std::vector<TreeNode*>                                 path;
    std::vector<TreeNode*>                                 inserted_nodes;
    std::vector<std::pair<TreeNode*, std::vector<size_t>>> adopted_nodes;
    // Borrowed from the tree; consumed under the BlockTreeCache mutex.
    std::vector<TreeNode*> newly_resident_nodes;
    // Number of logical GroupSetResources the tree took BLOCK_CACHE ownership of.
    size_t accepted_resource_count{0};
    size_t resident_prefix_length{0};
};

struct BlockTreeNodeRangeResult {
    size_t visited{0};
    size_t next_cursor{0};
    size_t tree_size{0};
    bool   cycle_complete{false};
};

struct BlockTreeChainSegmentSnapshot {
    size_t              start_tokens{0};
    size_t              end_tokens{0};
    bool                end_is_leaf{false};
    std::vector<size_t> recoverable_points_tokens;
};

struct BlockTreeResourceLeafSnapshot {
    CacheGroupType group_type{CacheGroupType::FULL};
    Tier           tier{Tier::DEVICE};
    size_t         leaf_count{0};
};

struct BlockTreeObservabilitySnapshot {
    size_t  node_count{0};
    size_t  root_child_count{0};
    size_t  leaf_count{0};
    size_t  linear_count{0};
    size_t  branch_count{0};
    size_t  max_child_count{0};
    size_t  chain_segment_count{0};
    size_t  max_chain_tokens{0};
    size_t  max_depth_tokens{0};
    size_t  leaf_depth_sum_tokens{0};
    size_t  max_leaf_depth_tokens{0};
    size_t  branch_depth_sum_tokens{0};
    size_t  max_branch_depth_tokens{0};
    size_t  recoverable_point_count{0};
    size_t  stored_cache_count{0};
    size_t  pure_content_count{0};
    size_t  incomplete_state_count{0};
    size_t  temporarily_busy_count{0};
    size_t  max_recoverable_depth_tokens{0};
    size_t  recoverable_depth_sum_tokens{0};
    size_t  first_recoverable_point_count{0};
    size_t  nearest_ancestor_gap_count{0};
    size_t  nearest_ancestor_gap_sum_tokens{0};
    size_t  nearest_ancestor_gap_max_tokens{0};
    int64_t node_age_sum_us{0};
    int64_t node_age_max_us{0};
    int64_t live_cache_age_sum_us{0};
    int64_t live_cache_age_max_us{0};
    size_t  completed_cache_lifetime_count{0};
    int64_t completed_cache_lifetime_sum_us{0};
    int64_t completed_cache_lifetime_max_us{0};
    size_t  observation_count{0};
    int64_t observation_cpu_sum_us{0};
    int64_t observation_cpu_max_us{0};
    int64_t snapshot_cpu_us{0};
    // Detail is available for explicit baseline capture; online metrics use bounded summaries.
    std::vector<BlockTreeChainSegmentSnapshot> chain_segments;
    std::vector<size_t>                        child_count_histogram;
    std::vector<size_t>                        leaf_depths_tokens;
    std::vector<size_t>                        branch_depths_tokens;
    std::vector<BlockTreeResourceLeafSnapshot> resource_leaves;
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets, size_t reuse_unit_tokens = 1);
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool                                              collect_path,
                                     bool                                              is_resident);

    bool      isRemovable(TreeNode* node) const;
    void      removePendingNodes(const std::unordered_set<TreeNode*>& nodes);
    TreeNode* removeNodeAndEmptyAncestors(TreeNode* node);

    TreeNode* root() const {
        return root_.get();
    }
    const std::vector<GroupSetPtr>& groupSets() const {
        return group_sets_;
    }
    const ReusableGroupLocation* reusableGroupLocation(std::string_view group_tag) const;
    size_t                       reusableGroupCount() const {
        return reusable_group_locations_.size();
    }
    size_t size() const {
        return node_pool_.size();
    }
    CacheKeysType cacheKeys() const;
    int64_t keyVersion() const {
        return key_version_;
    }

    // Diagnostic-only bounded walk over node_pool_[cursor, min(cycle_end_index, size())).
    // The caller must hold BlockTreeCache::mutex_
    BlockTreeNodeRangeResult visitNodeRangeLocked(size_t                                      cursor,
                                                  size_t                                      cycle_end_index,
                                                  size_t                                      max_nodes,
                                                  const std::function<void(const TreeNode&)>& visitor) const;

    // The owning BlockTreeCache mutex protects both state transitions and snapshots.
    void                           observeCacheState(const TreeNode* node);
    BlockTreeObservabilitySnapshot observabilitySnapshot(bool include_detail = true) const;

    // All publication operations require the owning BlockTreeCache mutex.
    // Only complete, matchable DEVICE values in every required group are advertised.
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    void            refreshPublishedState(const TreeNode* node, bool observe_cache_state = true);

private:
    void                               removePublishedKey(CacheKeyType key);
    KVCacheEventPublisherPtr           event_publisher_;
    std::vector<ReusableGroupLocation> publication_groups_;
    std::unordered_set<CacheKeyType>   published_keys_;
    int64_t                            publication_version_{-1};

    friend class BlockTreeEvictor;

    BlockTreeInsertResult insertNodeImpl(const CacheKeysType&                              cache_keys,
                                         const std::vector<std::vector<GroupSetResource>>& resources,
                                         bool                                              collect_path,
                                         bool                                              is_resident);

    void      removeNode(TreeNode* node);
    TreeNode* createNode(CacheKeyType key, TreeNode* parent);
    void      releaseNode(TreeNode* node);

    std::vector<GroupSetPtr>              group_sets_;
    ReusableGroupLocations                reusable_group_locations_;
    std::unique_ptr<TreeNode>             root_;
    std::deque<std::unique_ptr<TreeNode>> node_pool_;
    std::unordered_map<CacheKeyType, std::unique_ptr<TreeNode>> pending_node_pool_;
    // Indexed together with node_pool_ for contiguous snapshot copies.
    CacheKeysType cache_keys_;
    int64_t       key_version_{-1};
    size_t        reuse_unit_tokens_{1};
    size_t        completed_cache_lifetime_count_{0};
    int64_t       completed_cache_lifetime_sum_us_{0};
    int64_t       completed_cache_lifetime_max_us_{0};
    size_t        observation_count_{0};
    int64_t       observation_cpu_sum_us_{0};
    int64_t       observation_cpu_max_us_{0};
};

}  // namespace rtp_llm
