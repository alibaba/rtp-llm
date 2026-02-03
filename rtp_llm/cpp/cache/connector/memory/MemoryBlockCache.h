#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>
#include <optional>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class MemoryBlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key{0};
        BlockIdxType block_index{-1};
        size_t       block_size{0};
        bool         is_resident{false};
        // Hybrid-attn support:
        // - big  : this cache_key has complete KV (e.g. full + linear)
        // - small: this cache_key has partial KV (e.g. full only)
        bool is_big{true};
    };

    struct MatchResult {
        BlockIdxType matched_index{NULL_BLOCK_IDX};
        size_t       block_size{0};
        bool         is_big{false};
    };

public:
    static const size_t kCacheMaxCapacity = 10000000;
    explicit MemoryBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    MatchResult match(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    std::pair<bool, std::optional<CacheItem>> put(const CacheItem& cache_item);

    std::vector<BlockIdxType> pop(int n);

    bool empty() const;

    size_t size() const;

private:
    mutable LRUCache<CacheKeyType, CacheItem> lru_cache_;
    mutable std::mutex                        mutex_;
};

using MemoryBlockCachePtr = std::shared_ptr<MemoryBlockCache>;

}  // namespace rtp_llm
