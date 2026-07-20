#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/LRUCache.h"

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

    struct LogicalCacheSnapshot {
        int64_t                   version = -1;
        std::vector<CacheKeyType> cache_keys;
    };

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

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

    LogicalCacheSnapshot logicalCacheSnapshot() const;

    // Installed once during engine initialization and removed before publisher shutdown.
    // A logical key is externally reusable only when every required cache group exists.
    void setEventPublisher(KVCacheEventPublisherPtr publisher, int required_group_count);

private:
    size_t                                   seq_size_per_block_;
    LRUCacheType                             lru_cache_;
    KVCacheEventPublisherPtr                 event_publisher_;
    int                                      required_group_count_ = 1;
    std::unordered_map<CacheKeyType, size_t> key_group_counts_;
    // NOTE: BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe (unordered_map + list). Guard all accesses here.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
