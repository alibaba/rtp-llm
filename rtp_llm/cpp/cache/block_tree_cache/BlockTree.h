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

// A content-matched interval backed by a real tree node. The source endpoint
// can exceed matched_end; dense groups may reuse the shared part of that node.
struct BlockTreeContentRange {
    TreeNode* node{nullptr};
    size_t    matched_end{0};
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets,
                       size_t                    reuse_unit_tokens = 1);
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;
    // Returns all content- and identity-matched endpoints, including terminal short leaves.
    // The caller supplies keys computed with the request's original hash seed and base-block layout.
    std::vector<TreeNode*> contentMatchCandidates(
        const std::vector<int>& request_tokens, const std::function<CacheKeyType(size_t)>& endpoint_key) const;

    std::vector<BlockTreeContentRange>
    contentMatchRanges(const std::vector<int>&                                                   request_tokens,
                       const std::function<CacheKeyType(size_t)>&                                endpoint_key,
                       const std::function<bool(size_t, const std::vector<int>&, CacheKeyType)>& source_identity) const;

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool                                              collect_path,
                                     bool                                              is_resident,
                                     const std::vector<std::vector<int>>*              token_segments = nullptr,
                                     const std::vector<size_t>*                         token_ends = nullptr);

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

    // All publication operations require the owning BlockTreeCache mutex.
    // Only complete, matchable DEVICE values in every required group are advertised.
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    void            refreshPublishedState(const TreeNode* node);

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

    void      removeNode(TreeNode* node);
    TreeNode* createNode(CacheKeyType key, TreeNode* parent, size_t token_end);
    void      releaseNode(TreeNode* node);

    std::vector<GroupSetPtr>              group_sets_;
    ReusableGroupLocations                reusable_group_locations_;
    std::unique_ptr<TreeNode>             root_;
    std::deque<std::unique_ptr<TreeNode>> node_pool_;
    std::unordered_map<CacheKeyType, std::unique_ptr<TreeNode>> pending_node_pool_;
    // Indexed together with node_pool_ for contiguous snapshot copies.
    size_t reuse_unit_tokens_{1};
    CacheKeysType cache_keys_;
    int64_t       key_version_{-1};
};

}  // namespace rtp_llm
