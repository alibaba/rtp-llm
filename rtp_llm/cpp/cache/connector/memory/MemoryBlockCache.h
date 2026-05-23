#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <shared_mutex>
#include <optional>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/LRUCache.h"

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
        // Bitmap of which layer-region slots actually have valid GPU data.
        // Length matches layerRegionSlots() of the owning connector. When empty,
        // assume all slots are valid (legacy behaviour for non-DSV4 layouts).
        // Used by disk-spill staging to zero out NULL_BLOCK_IDX ranges so
        // stale memory-block bytes never leak onto disk.
        std::vector<bool> valid_slots;
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

    // Pop up to `n` LRU items.
    // Always skips resident items.
    // When only_complete=true (default), also skips items with is_complete=false. Disk spill
    // callers that need hybrid-attn prefix continuity should pass only_complete=false.
    std::vector<CacheItem> popItems(int n, bool only_complete = true);

    // Preferred name (README §"Memory block 生命周期和反压"). Same semantics as popItems.
    // Calls should migrate away from pop(n) entirely because pop() drops item metadata
    // needed by the disk tier.
    std::vector<CacheItem> takeLRUItems(int n, bool only_complete = true) {
        return popItems(n, only_complete);
    }

    bool empty() const;

    size_t size() const;

    std::vector<CacheKeyType> cacheKeys() const;

private:
    mutable LRUCache<CacheKeyType, CacheItem> lru_cache_;
    mutable std::shared_mutex                 mutex_;
};

using MemoryBlockCachePtr = std::shared_ptr<MemoryBlockCache>;

}  // namespace rtp_llm
