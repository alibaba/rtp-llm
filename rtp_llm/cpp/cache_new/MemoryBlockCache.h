#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

class MemoryBlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key;
        BlockIdxType block_index;
        size_t       block_size{0};
        bool         is_resident = false;
    };

    struct MatchResult {
        BlockIdxType matched_index;
        size_t       block_size{0};
    };

    using CacheSnapshot = typename LRUCache<CacheKeyType, CacheItem>::CacheSnapshot;

public:
    static const size_t kCacheMaxCapacity = 10000000;
    explicit MemoryBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    MatchResult match(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    bool put(CacheItem& cache_item);

    std::vector<BlockIdxType> pop(int n);

    bool empty() const;

    size_t size() const;

    std::vector<CacheItem> steal();

private:
    mutable LRUCache<CacheKeyType, CacheItem> lru_cache_;
    mutable std::mutex                        mutex_;
};

using MemoryBlockCachePtr = std::shared_ptr<MemoryBlockCache>;

}  // namespace rtp_llm
