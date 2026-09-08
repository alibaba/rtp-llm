#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <shared_mutex>
#include <optional>
#include <unordered_map>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class MemoryBlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key{0};
        BlockIdxType block_index{NULL_BLOCK_IDX};
        size_t       block_size{0};
        bool         is_resident{false};
        // 表示是否是完整的 KVCache, 只有当所有层都有 cache 时才为 true
        // 对于全注意力模型: 这个值始终为 true ; 对于混合注意力模型: 所有层都有 cache 时才为 true
        bool is_complete{true};
        // Identity used to reject stale asynchronous eviction callbacks.
        uint64_t generation{0};
    };

    struct MatchResult {
        BlockIdxType matched_index{NULL_BLOCK_IDX};
        size_t       block_size{0};
        bool         is_complete{false};
    };

public:
    static const size_t kCacheMaxCapacity = 10000000;
    explicit MemoryBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    MatchResult match(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    std::pair<bool, std::optional<CacheItem>> put(const CacheItem& cache_item);

    std::optional<CacheItem> remove(CacheKeyType cache_key);

    // Remove only if the stored block_index matches expected_block_index.
    // Returns the removed item on success, nullopt if key is missing or block_index differs.
    std::optional<CacheItem> removeIfMatch(CacheKeyType cache_key, BlockIdxType expected_block_index);

    std::vector<BlockIdxType> pop(int n);

    std::vector<CacheItem> detachForRemoteEviction(int n);

    std::vector<CacheItem> popForImmediateEviction(int n);

    std::optional<CacheItem> finishRemoteEviction(CacheKeyType cache_key, uint64_t generation);

    size_t remoteEvictingSize() const;

    bool empty() const;

    size_t size() const;

    std::vector<CacheKeyType> cacheKeys() const;

private:
    mutable LRUCache<CacheKeyType, CacheItem>   lru_cache_;
    std::unordered_map<CacheKeyType, CacheItem> remote_evicting_items_;
    uint64_t                                    next_generation_{1};
    mutable std::shared_mutex                   mutex_;
};

using MemoryBlockCachePtr = std::shared_ptr<MemoryBlockCache>;

}  // namespace rtp_llm
