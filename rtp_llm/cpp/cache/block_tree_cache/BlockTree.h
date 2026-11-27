#pragma once

#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTreeEvictor;

struct ReusableGroupLocation {
    size_t group_set_id{0};
    size_t member_group_id{0};
};

using ReusableGroupLocations = std::unordered_map<std::string, ReusableGroupLocation>;

struct BlockTreeReplacedCandidate {
    TreeNode* node;
    size_t    group_set_id;
    Tier      source_tier;
};

struct BlockTreeInsertResult {
    std::vector<TreeNode*>                                 path;
    std::vector<TreeNode*>                                 inserted_nodes;
    std::vector<std::pair<TreeNode*, std::vector<size_t>>> adopted_nodes;
    // Replaced FULL resources; old-tier heap erasure tolerates absent entries. Consumed under the cache mutex.
    std::vector<BlockTreeReplacedCandidate> replaced_candidates;
    std::vector<TransferDescriptor> revoked_evictions;
    // Borrowed from the tree; consumed under the BlockTreeCache mutex.
    std::vector<TreeNode*> newly_resident_nodes;
    // Number of logical GroupSetResources the tree took BLOCK_CACHE ownership of.
    size_t accepted_resource_count{0};
    size_t resident_prefix_length{0};
    bool hard_stop{false};
};

struct BlockTreeNodeRangeResult {
    size_t visited{0};
    size_t next_cursor{0};
    size_t tree_size{0};
    bool   cycle_complete{false};
};

struct OpaqueDeviceChange {
    std::string key;
    size_t group;
    bool added;
};

class BlockTree {
public:
    explicit BlockTree(std::vector<GroupSetPtr> group_sets, size_t key_tokens = 1);
    ~BlockTree();

    std::vector<TreeNode*> findNode(const CacheKeysType& cache_keys) const;
    std::vector<TreeNode*> findNode(const BlockTreeKeys& cache_keys) const;
    std::vector<TreeNode*> findNode(std::initializer_list<CacheKeyType> cache_keys) const {
        return findNode(CacheKeysType(cache_keys));
    }

    bool isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const;
    TreeNode* previousGroupPage(TreeNode* node, size_t group_set_id) const;
    const TreeNode* previousGroupPage(const TreeNode* node, size_t group_set_id) const;
    size_t keyTokens() const { return key_tokens_; }

    BlockTreeInsertResult insertNode(const CacheKeysType&                              cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool                                              collect_path,
                                     bool                                              is_resident);
    BlockTreeInsertResult insertNode(const BlockTreeKeys& cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool collect_path, bool is_resident);
    BlockTreeInsertResult insertNode(std::initializer_list<CacheKeyType> cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     bool collect_path, bool is_resident) {
        return insertNode(CacheKeysType(cache_keys), resources, collect_path, is_resident);
    }
    bool retainHostReplica(TreeNode* node, size_t group_set_id, BlockIdxType host);

    void      referenceNodeForNonOwnerTransfer(TreeNode* node);
    bool      isRemovable(TreeNode* node) const;
    void      removePendingNodes(const std::unordered_map<TreeNode*, size_t>& nodes);
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
    // Legacy integer event/snapshot identity. Opaque keys are exposed by TreeNode::key().
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

    // All publication operations require the owning BlockTreeCache mutex.
    // Complete, matchable DEVICE components are advertised independently.
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    void            refreshPublishedState(const TreeNode* node);
    // Read under the owning cache mutex. Change tracking stores only unconsumed deltas.
    uint64_t opaqueCatalogVersion() const { return opaque_catalog_version_; }
    size_t opaqueDeviceGroups() const { return opaque_device_groups_; }
    void trackOpaqueDeviceChanges(bool enabled);
    std::vector<OpaqueDeviceChange> takeOpaqueDeviceChanges();

private:
    void                               removePublishedKey(CacheKeyType key);
    void                               refreshOpaqueDeviceState(const TreeNode& node);
    KVCacheEventPublisherPtr           event_publisher_;
    std::vector<ReusableGroupLocation> publication_groups_;
    std::unordered_map<CacheKeyType, std::vector<size_t>> published_groups_;
    int64_t                            publication_version_{-1};

    friend class BlockTreeEvictor;

    template<typename Keys>
    std::vector<TreeNode*> findNodeImpl(const Keys& cache_keys) const;

    template<typename Keys>
    BlockTreeInsertResult validateAndInsert(const Keys& cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            bool collect_path, bool is_resident);

    template<typename Keys>
    BlockTreeInsertResult insertNodeImpl(const Keys&                                       cache_keys,
                                         const std::vector<std::vector<GroupSetResource>>& resources,
                                         bool                                              collect_path,
                                         bool                                              is_resident);

    bool      canReleaseNode(TreeNode* node) const;
    void      removeNode(TreeNode* node);
    TreeNode* createNode(BlockTreeKeyView key, TreeNode* parent);
    void      releaseNode(TreeNode* node);
    bool      hasPendingKey(BlockTreeKeyView key, const TreeNode* parent) const;

    const size_t                          key_tokens_;
    std::vector<GroupSetPtr>              group_sets_;
    ReusableGroupLocations                reusable_group_locations_;
    std::unique_ptr<TreeNode>             root_;
    std::deque<std::unique_ptr<TreeNode>> node_pool_;
    std::unordered_multimap<BlockTreeKeyView, std::unique_ptr<TreeNode>, BlockTreeKeyHash> pending_node_pool_;
    // Indexed together with node_pool_ for contiguous snapshot copies.
    CacheKeysType cache_keys_;
    int64_t       key_version_{-1};
    uint64_t      next_node_identity_{0};
    uint64_t      opaque_catalog_version_{0};
    size_t        opaque_device_groups_{0};
    bool          track_opaque_device_changes_{false};
    std::vector<OpaqueDeviceChange> opaque_device_changes_;
    size_t        opaque_node_count_{0};
};

}  // namespace rtp_llm
