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
#include "rtp_llm/cpp/cache/connector/memory/CacheBlockKind.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"

namespace rtp_llm {

enum class CacheBackingType {
    MEMORY = 0,
    DISK   = 1,
};

enum class CacheBackingRole {
    LEGACY = 0,
    KV     = 1,
    TAIL   = 2,
};

class MemoryDiskBlockCache {
public:
    struct CacheItem {
        CacheKeyType     cache_key{0};
        CacheBackingRole backing_role{CacheBackingRole::LEGACY};
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
        bool             found{false};
        bool             has_kv{false};
        bool             has_tail{false};
        CacheBackingRole backing_role{CacheBackingRole::LEGACY};
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     matched_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        bool             is_complete{false};
        CacheBackingType tail_backing_type{CacheBackingType::MEMORY};
        BlockIdxType     tail_matched_index{NULL_BLOCK_IDX};
        int32_t          tail_disk_slot{-1};
        size_t           tail_block_size{0};
    };

public:
    MatchResult match(CacheKeyType cache_key);
    MatchResult matchAndMarkInFlight(CacheKeyType cache_key);
    MatchResult matchAndMarkInFlight(CacheKeyType cache_key, CacheBackingRole role);
    bool        contains(CacheKeyType cache_key) const;
    bool        contains(CacheKeyType cache_key, CacheBackingRole role) const;

    std::pair<bool, std::optional<CacheItem>>                   putCommitted(const CacheItem& item);
    std::optional<CacheItem>                                    removeIfMatch(CacheKeyType     cache_key,
                                                                              CacheBackingType backing_type,
                                                                              BlockIdxType     expected_block_index,
                                                                              int32_t          expected_disk_slot);
    std::optional<CacheItem>                                    removeIfMatch(CacheKeyType     cache_key,
                                                                              CacheBackingRole role,
                                                                              CacheBackingType backing_type,
                                                                              BlockIdxType     expected_block_index,
                                                                              int32_t          expected_disk_slot);
    std::pair<bool, std::optional<MemoryBlockCache::CacheItem>> put(const MemoryBlockCache::CacheItem& item);
    std::optional<MemoryBlockCache::CacheItem>                  remove(CacheKeyType cache_key);
    std::vector<CacheItem>                                      removeAll(CacheKeyType cache_key);
    std::optional<MemoryBlockCache::CacheItem> removeIfMatch(CacheKeyType cache_key, BlockIdxType expected_block_index);
    std::optional<CacheItem>                   popOldestEvictable();
    std::optional<CacheItem>                   popOldestEvictable(CacheBlockKind kind);
    std::vector<CacheItem>                     popOldestEvictableBackings(CacheBlockKind kind);

    bool
    markInFlight(CacheKeyType cache_key, CacheBackingType backing_type, BlockIdxType block_index, int32_t disk_slot);
    bool markInFlight(CacheKeyType     cache_key,
                      CacheBackingRole role,
                      CacheBackingType backing_type,
                      BlockIdxType     block_index,
                      int32_t          disk_slot);
    void
    releaseInFlight(CacheKeyType cache_key, CacheBackingType backing_type, BlockIdxType block_index, int32_t disk_slot);
    void releaseInFlight(CacheKeyType     cache_key,
                         CacheBackingRole role,
                         CacheBackingType backing_type,
                         BlockIdxType     block_index,
                         int32_t          disk_slot);

    bool                      empty() const;
    size_t                    size() const;
    std::vector<CacheKeyType> cacheKeys() const;

private:
    struct EvictKey {
        uint64_t         last_access_seq{0};
        CacheKeyType     cache_key{0};
        CacheBackingRole backing_role{CacheBackingRole::LEGACY};

        bool operator<(const EvictKey& other) const {
            if (last_access_seq != other.last_access_seq) {
                return last_access_seq < other.last_access_seq;
            }
            if (cache_key != other.cache_key) {
                return cache_key < other.cache_key;
            }
            return static_cast<int>(backing_role) < static_cast<int>(other.backing_role);
        }
    };
    struct StoredItem {
        std::optional<CacheItem> legacy;
        std::optional<CacheItem> kv;
        std::optional<CacheItem> tail;
    };

    bool                               validItem(const CacheItem& item) const;
    static MemoryBlockCache::CacheItem toMemoryCacheItem(const CacheItem& item);
    void                               insertEvictKeyLocked(const CacheItem& item);
    void                               eraseEvictKeyLocked(const CacheItem& item);
    void                               touchLocked(CacheItem& item);
    std::optional<CacheItem>           oldestFromSetLocked(std::set<EvictKey>& eviction_set,
                                                           bool                single_backing_eviction = false);
    std::optional<CacheItem>           popOldestEvictableLocked(CacheBlockKind kind);
    std::vector<CacheItem>             removeBackingLocked(CacheKeyType cache_key, CacheBackingRole role);
    std::optional<CacheItem>           findBackingLocked(CacheKeyType cache_key, CacheBackingRole role) const;
    std::optional<CacheItem*>          mutableBackingLocked(StoredItem& stored, CacheBackingRole role);
    std::optional<const CacheItem*>    backingLocked(const StoredItem& stored, CacheBackingRole role) const;
    bool                               backingMatchesLocked(const CacheItem& item,
                                                            CacheBackingType backing_type,
                                                            BlockIdxType     expected_block_index,
                                                            int32_t          expected_disk_slot) const;
    static bool                        emptyStoredItem(const StoredItem& item);
    static CacheBlockKind              blockKindForItem(const CacheItem& item);
    static CacheBackingRole            normalizeRole(const CacheItem& item);
    std::set<EvictKey>&                lruSetLocked(CacheBackingType backing_type, CacheBlockKind kind);

private:
    mutable std::shared_mutex                     mutex_;
    std::unordered_map<CacheKeyType, StoredItem>  items_;
    std::set<EvictKey>                            memory_complete_lru_;
    std::set<EvictKey>                            memory_incomplete_lru_;
    std::set<EvictKey>                            disk_complete_lru_;
    std::set<EvictKey>                            disk_incomplete_lru_;
    uint64_t                                      access_seq_{0};
};

using MemoryDiskBlockCachePtr = std::shared_ptr<MemoryDiskBlockCache>;

}  // namespace rtp_llm
