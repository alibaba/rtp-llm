#pragma once

#include <cstdint>
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
        CacheKeyType cache_key;
        GroupIdType  group_id;
        BlockIdxType block_index;
        bool         is_resident = false;
    };

    struct BatchMatchResult {
        std::vector<BlockIdxType> matched_indices;
    };

    struct MatchResult {
        BlockIdxType matched_index;
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

    // Freeze/resume (M5): drop all entries and bump the cache generation.
    // After a KV physical-memory pause/resume cycle, every cached block content is invalid
    // (constraint C3), so all prefix-cache entries must be discarded.
    // Minimal generation scheme: clearing empties the cache (any old cache key misses) and
    // increments generation_; readers can compare generation() to detect stale snapshots /
    // cache keys captured before the reset.
    void clear();

    // Monotonically increasing generation, bumped on every clear(). Entries put before a
    // clear() belong to an older generation and can never be matched again.
    uint64_t generation() const;

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    size_t       seq_size_per_block_;
    LRUCacheType lru_cache_;
    uint64_t     generation_ = 0;
    // NOTE: BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe (unordered_map + list). Guard all accesses here.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
