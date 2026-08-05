#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class SharedBlockCache {
public:
    using NamespaceId = uint32_t;

    static constexpr NamespaceId kDefaultNamespace        = 0;
    static constexpr NamespaceId kGpuLogicalNamespace     = 1;
    static constexpr NamespaceId kGpuCpCanonicalNamespace = 2;

    struct NamespacedKey {
        NamespaceId  namespace_id{0};
        CacheKeyType cache_key{0};

        bool operator==(const NamespacedKey& other) const {
            return namespace_id == other.namespace_id && cache_key == other.cache_key;
        }
    };

    struct NamespacedKeyHash {
        size_t operator()(const NamespacedKey& key) const {
            return std::hash<uint64_t>()((static_cast<uint64_t>(key.namespace_id) << 32)
                                         ^ static_cast<uint64_t>(key.cache_key));
        }
    };

    struct UnifiedCacheItem {
        // cache_key is the request-level identity. Tags only route the physical payload for that identity to a pool;
        // they never create a second cache item or dependency timeline.
        CacheKeyType                                  cache_key;
        bool                                          is_resident = false;
        std::unordered_map<std::string, BlockIdxType> group_block_ids;
        std::unordered_map<std::string, bool>         matchable_groups;
        std::unordered_map<std::string, int64_t>      group_block_created_time_us;
        int64_t                                       created_time_us = 0;
        BlockDependency                               dependency;
        NamespaceId                                   dependency_namespace = kDefaultNamespace;
        bool                                          has_dependency       = false;
    };

    struct EvictResult {
        std::vector<CacheKeyType>                                                       evicted_keys;
        std::unordered_map<CacheKeyType, std::unordered_map<std::string, BlockIdxType>> evicted_group_block_ids;
        std::unordered_map<CacheKeyType, BlockDependency>                               evicted_dependencies;
        std::unordered_map<CacheKeyType, NamespaceId>                                   evicted_namespaces;
        std::unordered_map<CacheKeyType, int64_t>                                       evicted_lifetime_ms;
        std::unordered_map<CacheKeyType, std::string>                                   evicted_independent_tag;
    };

    struct MatchResult {
        bool                                          found = false;
        std::unordered_map<std::string, BlockIdxType> group_block_ids;
    };

    using LRUCacheType = LRUCache<CacheKeyType, UnifiedCacheItem>;

public:
    explicit SharedBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    void init(const std::vector<std::pair<std::string, BlockPoolPtr>>& pools);

    void put(CacheKeyType                                             cache_key,
             const std::vector<std::pair<std::string, BlockIdxType>>& group_block_ids,
             bool                                                     is_resident);
    void put(CacheKeyType                                             cache_key,
             const std::vector<std::pair<std::string, BlockIdxType>>& group_block_ids,
             bool                                                     is_resident,
             NamespaceId                                              namespace_id,
             const BlockDependency&                                   dependency,
             const std::unordered_map<std::string, bool>&             matchable_groups = {});

    MatchResult match(CacheKeyType cache_key);

    BlockIdxType matchGroup(CacheKeyType cache_key, std::string_view tag);

    EvictResult selectAndEvict(size_t min_blocks);
    EvictResult selectAndEvictForGroup(std::string_view tag, size_t min_blocks);

    size_t evictAndFree(size_t min_blocks);
    size_t evictAndFreeForGroup(std::string_view tag, size_t min_blocks, EvictResult* evict_result_out = nullptr);

    std::optional<UnifiedCacheItem> remove(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    bool empty() const;

    size_t size() const;

    std::vector<CacheKeyType> allCacheKeys() const;

    int64_t version() const;
    void    setPrefixTreeEnabled(bool enabled);
    bool    prefixTreeEnabled() const;
    void    setIndependentGroupEviction(bool enabled, const std::vector<std::string>& tags);

private:
    static const size_t kCacheMaxCapacity = 10000000;

    struct PrefixTreeNode {
        NamespacedKey                                        key;
        NamespacedKey                                        parent;
        bool                                                 has_parent{false};
        bool                                                 resident{false};
        uint32_t                                             ordinal{0};
        uint64_t                                             last_access_seq{0};
        std::unordered_set<NamespacedKey, NamespacedKeyHash> children;
    };

    struct LeafKey {
        uint64_t     last_access_seq{0};
        NamespaceId  namespace_id{0};
        CacheKeyType cache_key{0};

        bool operator<(const LeafKey& other) const {
            if (last_access_seq != other.last_access_seq) {
                return last_access_seq < other.last_access_seq;
            }
            if (namespace_id != other.namespace_id) {
                return namespace_id < other.namespace_id;
            }
            return cache_key < other.cache_key;
        }
    };

    void        upsertTreeNodeLocked(CacheKeyType           cache_key,
                                     NamespaceId            namespace_id,
                                     const BlockDependency& dependency,
                                     bool                   is_resident);
    void        detachPendingChildLocked(const NamespacedKey& parent, const NamespacedKey& child);
    void        attachPendingChildrenLocked(PrefixTreeNode& node);
    void        touchTreeAliasesLocked(CacheKeyType cache_key);
    void        touchTreeNodeLocked(PrefixTreeNode& node);
    void        eraseLeafLocked(const PrefixTreeNode& node);
    void        insertLeafIfEligibleLocked(const PrefixTreeNode& node);
    void        refreshLeafLocked(const NamespacedKey& key);
    void        removeTreeAliasLocked(const NamespacedKey& key);
    void        removeAllTreeAliasesForCacheKeyLocked(CacheKeyType cache_key);
    void        markAllTreeAliasesResidentLocked(CacheKeyType cache_key);
    void        refreshAllTreeAliasesLocked(CacheKeyType cache_key);
    bool        flatItemHasCanonicalDependencyLocked(CacheKeyType cache_key) const;
    bool        updateItemDependencyLocked(UnifiedCacheItem&      item,
                                           NamespaceId            namespace_id,
                                           const BlockDependency& dependency) const;
    void        validateTagLocked(std::string_view tag) const;
    void        validateInputBlocksLocked(const std::vector<std::pair<std::string, BlockIdxType>>& group_block_ids,
                                          const std::unordered_map<std::string, bool>&             matchable_groups) const;
    static bool hasUsableGroup(const UnifiedCacheItem& item, std::string_view tag);
    std::vector<NamespacedKey> collectEvictChainLocked(const NamespacedKey& leaf_key) const;
    bool chainHasUsableGroupLocked(const std::vector<NamespacedKey>& chain, std::string_view tag) const;
    bool chainHasReachableAncestorGroupLocked(const std::vector<NamespacedKey>& chain, std::string_view tag) const;
    bool subtreeEvictableForAncestorGroupLocked(const NamespacedKey& key) const;
    bool selectIndependentGroupEvictionsLocked(std::string_view tag, size_t min_blocks, EvictResult& result);
    void removeGroupFromItemLocked(CacheKeyType cache_key, std::string_view tag, EvictResult& result);
    bool hasFlatItemLocked(CacheKeyType cache_key) const;
    bool isFlatItemResidentLocked(CacheKeyType cache_key) const;
    bool isIndependentEvictionGroupLocked(std::string_view tag) const;

    LRUCacheType       lru_cache_;
    mutable std::mutex mu_;
    int64_t            version_{-1};
    bool               prefix_tree_enabled_{true};
    bool               independent_group_eviction_enabled_{false};
    uint64_t           tree_access_seq_{0};

    std::unordered_map<std::string, BlockPoolPtr>                                          group_pools_;
    std::unordered_map<NamespacedKey, PrefixTreeNode, NamespacedKeyHash>                   tree_nodes_;
    std::unordered_map<CacheKeyType, std::unordered_set<NamespacedKey, NamespacedKeyHash>> aliases_by_cache_key_;
    std::unordered_map<NamespacedKey, std::unordered_set<NamespacedKey, NamespacedKeyHash>, NamespacedKeyHash>
                                    pending_children_by_parent_;
    std::set<LeafKey>               leaf_lru_;
    std::unordered_set<std::string> independent_eviction_tags_;
};

using SharedBlockCachePtr = std::shared_ptr<SharedBlockCache>;

}  // namespace rtp_llm
