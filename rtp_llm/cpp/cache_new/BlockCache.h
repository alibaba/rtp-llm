#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache_new/types.h"

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

    bool contains(size_t cache_key, int group_id = 0) const;

    MatchResult match(size_t cache_key, int group_id = 0);

    BlockIndicesType pop(int n);

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

    std::vector<BlockIdxType> steal();

private:
    size_t             seq_size_per_block_;
    LRUCacheType       lru_cache_;
    mutable std::mutex mutex_;
};

using BlockCacheV1Ptr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
