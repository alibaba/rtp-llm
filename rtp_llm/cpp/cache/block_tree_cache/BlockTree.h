#pragma once

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <set>
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

struct BlockTreeContentMatch {
    size_t       pos{0};
    CacheKeyType key{0};
    size_t       node_offset{0};
    bool         partial{false};
};

// A content-matched interval backed by a real tree node. The source endpoint
// can exceed matched_end; dense groups may reuse the shared part of that node.
struct BlockTreeContentRange {
    TreeNode* node{nullptr};
    size_t    matched_end{0};
};

struct BlockTreeAgeSummary {
    size_t                     count{0};
    std::optional<int64_t>     mean_age_us;
    std::optional<int64_t>     max_age_us;
};

struct BlockTreeObservabilitySnapshot {
    size_t                 node_count{0};
    size_t                 leaf_count{0};
    size_t                 branch_count{0};
    BlockTreeAgeSummary    content;
    BlockTreeAgeSummary    recovery;
    std::array<BlockTreeAgeSummary, 3> tier;
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets,
                       size_t                    reuse_unit_tokens = 1,
                       bool                      enable_observability = false,
                       std::string               observability_instance = {});
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;
    size_t                 contentMatchPos(const std::vector<int>& request_tokens) const;
    std::optional<BlockTreeContentMatch> contentMatchInfo(const std::vector<int>& request_tokens) const;
    // Returns all content- and identity-matched endpoints, including terminal short leaves.
    // The caller supplies keys computed with the request's original hash seed and base-block layout.
    std::vector<TreeNode*> contentMatchCandidates(
        const std::vector<int>& request_tokens, const std::function<CacheKeyType(size_t)>& endpoint_key) const;

    std::vector<BlockTreeContentRange>
    contentMatchRanges(const std::vector<int>&                                                   request_tokens,
                       const std::function<CacheKeyType(size_t)>&                                endpoint_key,
                       const std::function<bool(size_t, const std::vector<int>&, CacheKeyType)>& source_identity,
                       std::optional<PrefixCachePlanPoint>* farthest_fork = nullptr) const;
    // Resolve one group's physical source for a shared exact endpoint. The
    // borrowed nodes are valid only while the owning cache lock is held.
    std::optional<std::vector<TreeNode*>> coveredDeviceGroupPath(
        const std::vector<BlockTreeContentRange>& ranges, size_t endpoint, size_t group_id) const;

    std::vector<PrefixCachePlanPoint> planRecoveryPoints(
        const CacheKeysType&                              cache_keys,
        const std::vector<std::vector<GroupSetResource>>& resources,
        const std::vector<std::vector<int>>*              token_segments,
        Tier                                              target_tier,
        const std::vector<size_t>*                         token_ends = nullptr) const;
    std::vector<PrefixCachePlanPoint> unfulfilledPlanPoints(
        const std::vector<PrefixCachePlanPoint>& plan_points,
        const CacheKeysType&                     cache_keys) const;

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool                                              collect_path,
                                     bool                                              is_resident,
                                     const std::vector<std::vector<int>>*              token_segments = nullptr,
                                     const std::vector<size_t>*                         token_ends = nullptr);

    bool      isRemovable(TreeNode* node) const;
    void      removePendingNodes(const std::unordered_set<TreeNode*>& nodes);
    TreeNode* removeNodeAndEmptyAncestors(TreeNode* node, const char* trigger = "mutation");

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
    bool                         observabilityEnabled() const {
        return observability_enabled_;
    }
    const std::string&           observabilityInstance() const {
        return observability_instance_;
    }
    size_t size() const {
        return node_pool_.size();
    }
    size_t reuseUnitTokens() const {
        return reuse_unit_tokens_;
    }
    CacheKeysType cacheKeys() const;
    bool isShortLeaf(const TreeNode* node) const;
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
    void                           observeCacheState(const TreeNode* node,
                                                    const char*      trigger = "mutation");
    BlockTreeObservabilitySnapshot observabilitySnapshot() const;

    // All publication operations require the owning BlockTreeCache mutex.
    // Only complete, matchable DEVICE values in every required group are advertised.
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    void            refreshPublishedState(const TreeNode* node,
                                          bool            observe_cache_state = true,
                                          const char*     trigger             = "store");

    // Defers interval observation until the end of one locked mutation batch.
    // The caller must already hold the owning BlockTreeCache mutex. Nested scopes
    // are supported; only the outermost scope observes the final tree state.
    class ObservationBatch {
    public:
        explicit ObservationBatch(BlockTree& tree);
        ~ObservationBatch();

        ObservationBatch(const ObservationBatch&)            = delete;
        ObservationBatch& operator=(const ObservationBatch&) = delete;
        ObservationBatch(ObservationBatch&&)                 = delete;
        ObservationBatch& operator=(ObservationBatch&&)      = delete;

    private:
        BlockTree& tree_;
    };

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
                                         bool                                              is_resident,
                                         const std::vector<std::vector<int>>*              token_segments,
                                         const std::vector<size_t>*                         token_ends);

    void      removeNode(TreeNode* node, const char* trigger = "mutation");
    TreeNode* createNode(CacheKeyType key, TreeNode* parent, size_t token_end);
    void      releaseNode(TreeNode* node);

    struct IntervalSet {
        std::multiset<int64_t> starts;
        long double            start_sum{0.0L};
    };

    struct DeferredObservationRoot {
        const TreeNode* node{nullptr};
        const char*     trigger{nullptr};
    };

    static size_t tierIndex(Tier tier);
    void deferObservationRoot(const TreeNode* node, const char* trigger);
    void endObservationBatch();
    void commitDeferredObservations();
    void eraseDeferredObservationRoots(const TreeNode* node);
    static bool isObservationAncestor(const TreeNode* ancestor, const TreeNode* candidate);

    void startInterval(IntervalSet& intervals, int64_t& since_us);
    void endInterval(IntervalSet& intervals,
                     int64_t&     since_us,
                     const TreeNode& node,
                     const char* kind,
                     const char* medium = nullptr,
                     const char* trigger = "mutation",
                     const char* effect = "release");
    void updateContentShape(const TreeNode& node, size_t old_child_count, size_t new_child_count);
    void updateTierIntervals(TreeNode& node, const char* trigger);
    BlockTreeAgeSummary intervalSummary(const IntervalSet& intervals) const;

    std::vector<GroupSetPtr>              group_sets_;
    ReusableGroupLocations                reusable_group_locations_;
    std::unique_ptr<TreeNode>             root_;
    std::deque<std::unique_ptr<TreeNode>> node_pool_;
    std::unordered_map<CacheKeyType, std::unique_ptr<TreeNode>> pending_node_pool_;
    // Indexed together with node_pool_ for contiguous snapshot copies.
    CacheKeysType cache_keys_;
    int64_t       key_version_{-1};
    size_t        reuse_unit_tokens_{1};
    bool          observability_enabled_{false};
    std::string   observability_instance_;
    int64_t       tree_epoch_us_{0};
    size_t        leaf_count_{0};
    size_t        branch_count_{0};
    IntervalSet   content_intervals_;
    IntervalSet   recovery_intervals_;
    std::array<IntervalSet, 3> tier_intervals_;
    size_t       observation_batch_depth_{0};
    std::vector<DeferredObservationRoot> deferred_observation_roots_;
};

}  // namespace rtp_llm
