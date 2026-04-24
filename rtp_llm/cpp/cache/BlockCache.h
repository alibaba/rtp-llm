#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

using CacheKeyGroupPair = std::pair<CacheKeyType, GroupIdType>;  // <cache_key, group_id>

class BlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key   = 0;
        GroupIdType  group_id    = 0;
        BlockIdxType block_index = NULL_BLOCK_IDX;
        bool         is_resident = false;
        /// 0 = legacy entry (no parent-bucket indexing). >0 means tail/partial metadata is valid.
        int                  valid_token_len  = 0;
        CacheKeyType         parent_block_key = 0;
        std::vector<int32_t> prefix_tokens;
        bool                 is_linear_group = false;
    };

    struct BatchMatchResult {
        std::vector<BlockIdxType> matched_indices;
    };

    struct MatchResult {
        BlockIdxType matched_index;
    };

    /// Best partial-tail reuse under (parent_block_key, group_id) per gpu_partial_kv_cache_reuse.md.
    struct PartialTailMatchResult {
        BlockIdxType matched_index = NULL_BLOCK_IDX;
        size_t       reuse_tokens  = 0;
    };

    using LRUCacheType  = LRUCache<CacheKeyGroupPair,
                                   CacheItem,
                                   PairFirstHash<CacheKeyType, GroupIdType>,
                                   PairBothEqual<CacheKeyType, GroupIdType>>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    bool put(CacheItem& cache_item);

    bool contains(CacheKeyType cache_key, int group_id = 0) const;

    MatchResult match(CacheKeyType cache_key, int group_id = 0);

    /// Parent-bucket partial tail match (phase-2). Caller supplies request tail tokens starting at @p req_token_off.
    PartialTailMatchResult matchPartialTailByParent(CacheKeyType   parent_block_key,
                                                    GroupIdType    group_id,
                                                    int            L,
                                                    int            seq_size_per_block,
                                                    bool           is_linear_attention,
                                                    const int32_t* req_tokens,
                                                    int            req_token_off);

    BlockIndicesType pop(int n);

    std::optional<CacheItem> remove(CacheKeyType cache_key, int group_id = 0);

    // Select and remove LRU cache entries until at least min_blocks are freed.
    // Skips resident entries and keys that have any resident item.
    // Returns evicted items grouped by cache_key (in LRU order).
    struct EvictResult {
        std::vector<CacheKeyType>                                evicted_keys;
        std::unordered_map<CacheKeyType, std::vector<CacheItem>> evicted_items;
    };
    EvictResult selectAndEvict(size_t min_blocks);

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    using ParentGroupKey = CacheKeyGroupPair;

    void onLruEntryRemoved(const CacheKeyGroupPair& lru_key, const CacheItem& item);
    void registerParentIndex(const CacheItem& item, const CacheKeyGroupPair& lru_key);

    size_t       seq_size_per_block_;
    LRUCacheType lru_cache_;
    // NOTE: BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe (unordered_map + list). Guard all accesses here.
    mutable std::mutex mu_;

    /// (parent_block_key, group_id) -> LRU keys of entries indexed for partial-tail lookup.
    std::unordered_map<ParentGroupKey,
                       std::vector<CacheKeyGroupPair>,
                       PairFirstHash<CacheKeyType, GroupIdType>,
                       PairBothEqual<CacheKeyType, GroupIdType>>
        parent_bucket_;

    /// LRU primary key -> parent bucket key (only for entries with valid_token_len > 0).
    std::unordered_map<CacheKeyGroupPair,
                       ParentGroupKey,
                       PairFirstHash<CacheKeyType, GroupIdType>,
                       PairBothEqual<CacheKeyType, GroupIdType>>
        lru_key_to_parent_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
