#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"

namespace rtp_llm {

enum class CacheBackingType {
    MEMORY = 0,
    DISK   = 1,
};

class MemoryDiskBlockCache {
public:
    struct CacheItem {
        CacheKeyType     cache_key{0};
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     block_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        bool             is_resident{false};
        bool             is_complete{true};
        uint64_t         last_access_seq{0};
        uint32_t         in_flight_ref{0};
    };

    struct MatchResult {
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     matched_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        bool             is_complete{false};
    };

public:
    MatchResult match(CacheKeyType cache_key);
    MatchResult matchAndMarkInFlight(CacheKeyType cache_key);
    bool        contains(CacheKeyType cache_key) const;

    std::pair<bool, std::optional<CacheItem>>                   putCommitted(const CacheItem& item);
    std::optional<CacheItem>                                    removeIfMatch(CacheKeyType     cache_key,
                                                                              CacheBackingType backing_type,
                                                                              BlockIdxType     expected_block_index,
                                                                              int32_t          expected_disk_slot);
    std::pair<bool, std::optional<MemoryBlockCache::CacheItem>> put(const MemoryBlockCache::CacheItem& item);
    std::optional<MemoryBlockCache::CacheItem>                  remove(CacheKeyType cache_key);
    std::optional<MemoryBlockCache::CacheItem> removeIfMatch(CacheKeyType cache_key, BlockIdxType expected_block_index);
    std::optional<CacheItem>                   popOldestEvictable();

    bool
    markInFlight(CacheKeyType cache_key, CacheBackingType backing_type, BlockIdxType block_index, int32_t disk_slot);
    void
    releaseInFlight(CacheKeyType cache_key, CacheBackingType backing_type, BlockIdxType block_index, int32_t disk_slot);

    bool                      empty() const;
    size_t                    size() const;
    std::vector<CacheKeyType> cacheKeys() const;

private:
    struct EvictKey {
        uint64_t     last_access_seq{0};
        CacheKeyType cache_key{0};

        bool operator<(const EvictKey& other) const {
            if (last_access_seq != other.last_access_seq) {
                return last_access_seq < other.last_access_seq;
            }
            return cache_key < other.cache_key;
        }
    };

    bool                               validItem(const CacheItem& item) const;
    static MemoryBlockCache::CacheItem toMemoryCacheItem(const CacheItem& item);
    void                               insertEvictKeyLocked(const CacheItem& item);
    void                               eraseEvictKeyLocked(const CacheItem& item);
    void                               touchLocked(CacheItem& item);
    std::optional<CacheItem>           oldestFromSetLocked(std::set<EvictKey>& eviction_set);

private:
    mutable std::shared_mutex                   mutex_;
    std::unordered_map<CacheKeyType, CacheItem> items_;
    std::set<EvictKey>                          memory_lru_;
    std::set<EvictKey>                          disk_lru_;
    uint64_t                                    access_seq_{0};
};

using MemoryDiskBlockCachePtr = std::shared_ptr<MemoryDiskBlockCache>;

}  // namespace rtp_llm
