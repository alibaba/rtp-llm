#pragma once

#include <mutex>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

using NamespaceId = uint32_t;

struct SharedGroupBinding {
    BlockIdxType pool_block_id{NULL_BLOCK_IDX};
    bool         matchable{true};
    int64_t      created_time_us{0};
};

struct UnifiedCacheItem {
    bool                                      is_resident{false};
    std::map<std::string, SharedGroupBinding> bindings_by_group;
    int64_t                                   created_time_us{0};
    BlockDependency                           dependency;
    NamespaceId                               dependency_namespace{0};
    bool                                      has_dependency{false};
};

enum class EvictionKind {
    WholeItem,
    IndependentGroup,
};

struct CacheEviction {
    CacheKeyType                        cache_key{0};
    std::map<std::string, BlockIdxType> blocks_by_group;
    BlockDependency                     dependency;
    NamespaceId                         dependency_namespace{0};
    bool                                has_dependency{false};
    int64_t                             lifetime_ms{0};
    EvictionKind                        kind{EvictionKind::WholeItem};
    std::string                         group_tag;
};

struct EvictResult {
    std::vector<CacheEviction> evictions;
};

class SharedBlockCache {
public:
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

    using LRUCacheType = LRUCache<CacheKeyType, UnifiedCacheItem>;

public:
    explicit SharedBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    void init(const CacheConfig& config, const std::map<std::string, BlockPoolPtr>& group_pools);

    void put(CacheKeyType cache_key, const std::map<std::string, BlockIdxType>& group_block_ids, bool is_resident);
    void put(CacheKeyType                               cache_key,
             const std::map<std::string, BlockIdxType>& group_block_ids,
             const std::map<std::string, bool>&         group_matchable,
             bool                                       is_resident,
             NamespaceId                                namespace_id,
             const BlockDependency&                     dependency);

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

protected:
    virtual void blockCacheReferenceByTag(std::string_view tag, BlockIdxType block_id);
    virtual void blockCacheFreeByTag(std::string_view tag, BlockIdxType block_id);

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

    void                       upsertTreeNodeLocked(CacheKeyType           cache_key,
                                                    NamespaceId            namespace_id,
                                                    const BlockDependency& dependency,
                                                    bool                   is_resident);
    void                       detachPendingChildLocked(const NamespacedKey& parent, const NamespacedKey& child);
    void                       attachPendingChildrenLocked(PrefixTreeNode& node);
    void                       touchTreeAliasesLocked(CacheKeyType cache_key);
    void                       touchTreeNodeLocked(PrefixTreeNode& node);
    void                       eraseLeafLocked(const PrefixTreeNode& node);
    void                       insertLeafIfEligibleLocked(const PrefixTreeNode& node);
    void                       refreshLeafLocked(const NamespacedKey& key);
    void                       removeTreeAliasLocked(const NamespacedKey& key);
    void                       removeAllTreeAliasesForCacheKeyLocked(CacheKeyType cache_key);
    void                       markAllTreeAliasesResidentLocked(CacheKeyType cache_key);
    void                       refreshAllTreeAliasesLocked(CacheKeyType cache_key);
    bool                       flatItemHasCanonicalDependencyLocked(CacheKeyType cache_key) const;
    bool                       updateItemDependencyLocked(UnifiedCacheItem&      item,
                                                          NamespaceId            namespace_id,
                                                          const BlockDependency& dependency) const;
    CacheEviction              makeWholeItemEvictionLocked(CacheKeyType            cache_key,
                                                           const UnifiedCacheItem& item,
                                                           NamespaceId             fallback_namespace) const;
    void                       validateTagLocked(std::string_view tag) const;
    static bool                hasUsableGroup(const UnifiedCacheItem& item, std::string_view tag);
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

    std::map<std::string, BlockPoolPtr>                                                    group_pools_;
    std::vector<std::string>                                                               group_tags_in_order_;
    std::unordered_map<NamespacedKey, PrefixTreeNode, NamespacedKeyHash>                   tree_nodes_;
    std::unordered_map<CacheKeyType, std::unordered_set<NamespacedKey, NamespacedKeyHash>> aliases_by_cache_key_;
    std::unordered_map<NamespacedKey, std::unordered_set<NamespacedKey, NamespacedKeyHash>, NamespacedKeyHash>
                                    pending_children_by_parent_;
    std::set<LeafKey>               leaf_lru_;
    std::unordered_set<std::string> independent_eviction_group_tags_;
};

using SharedBlockCachePtr = std::shared_ptr<SharedBlockCache>;

}  // namespace rtp_llm
