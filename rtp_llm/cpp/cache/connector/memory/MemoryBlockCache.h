#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

#include "rtp_llm/cpp/utils/LRUCache.h"

namespace rtp_llm {

class MemoryBlockCache {
public:
    struct CacheItem {
        int64_t cache_key{0};
        int32_t block_index{-1};
        size_t  block_size{0};
        bool    is_resident{false};
    };

    struct MatchResult {
        int32_t matched_index{-1};
        size_t  block_size{0};
    };

public:
    static const size_t kCacheMaxCapacity = 10000000;
    explicit MemoryBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    MatchResult match(int64_t cache_key);

    bool contains(int64_t cache_key) const;

    std::pair<bool, std::optional<CacheItem>> put(const CacheItem& cache_item);

    std::vector<int32_t> pop(int n);

    bool empty() const;

    size_t size() const;

    std::vector<CacheItem> steal();

private:
    mutable LRUCache<int64_t, CacheItem> lru_cache_;
    mutable std::mutex                   mutex_;
};

using MemoryBlockCachePtr = std::shared_ptr<MemoryBlockCache>;

}  // namespace rtp_llm
