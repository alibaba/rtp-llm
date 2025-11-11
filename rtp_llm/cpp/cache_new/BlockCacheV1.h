#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

using CacheKeyGroupPair = std::pair<CacheKeyType, GroupIdType>;  // <cache_key, group_id>

class BlockCacheV1 {
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
    explicit BlockCacheV1(): lru_cache_(kCacheMaxCapacity) {}

    MatchResult match(size_t cache_key, int group_id = 0);

    bool contains(size_t cache_key, int group_id = 0);

    bool put(CacheItem& cache_item);

    // std::vector<BlockIdxType> put(const std::vector<CacheKeyType> cache_keys, const std::vector<BlockIdxType>
    // block_indices);

    std::vector<BlockIdxType> pop(int n);

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    size_t               seq_size_per_block_;
    mutable LRUCacheType lru_cache_;

    // 先不用锁看下，否则再看并发问题。
    mutable std::mutex mutex_;
};

using BlockCacheV1Ptr = std::shared_ptr<BlockCacheV1>;

}  // namespace rtp_llm
