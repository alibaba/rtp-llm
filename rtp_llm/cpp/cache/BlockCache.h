#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

struct CacheKeyGroupEpoch {
    CacheKeyType cache_key;
    GroupIdType  group_id;
    int64_t      epoch;
};

struct CacheKeyGroupEpochHash {
    size_t operator()(const CacheKeyGroupEpoch& key) const {
        size_t seed = std::hash<CacheKeyType>{}(key.cache_key);
        seed ^= std::hash<GroupIdType>{}(key.group_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.epoch) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct CacheKeyGroupEpochEqual {
    bool operator()(const CacheKeyGroupEpoch& lhs, const CacheKeyGroupEpoch& rhs) const {
        return lhs.cache_key == rhs.cache_key && lhs.group_id == rhs.group_id && lhs.epoch == rhs.epoch;
    }
};

class BlockCache {
public:
    // Epoch sentinels:
    //   GLOBAL_EPOCH (0)    : applied to CacheItem.epoch — entry is globally visible.
    //                         Also valid as a query epoch meaning "no batch identity,
    //                         only see global entries".
    //   NO_EPOCH_FILTER (-1): query-side ONLY — bypass filter, see everything (diagnostics only).
    // Batch-local epochs are positive integers (>= 1) assigned by the scheduler.
    // CacheItem.epoch must always be >= 0.
    static constexpr int64_t GLOBAL_EPOCH    = 0;
    static constexpr int64_t NO_EPOCH_FILTER = -1;

    struct CacheItem {
        CacheKeyType cache_key;
        GroupIdType  group_id;
        BlockIdxType block_index;
        bool         is_resident = false;
        int64_t      epoch       = GLOBAL_EPOCH;  // 0 = globally visible, >=1 = visible only within the same batch
        std::string  debugString() const {
            std::stringstream debug_string;
            debug_string << "CacheItem cache_key: " << cache_key << ", group_id: " << group_id
                         << ", block_index: " << block_index << ", is_resident: " << is_resident
                         << ", epoch: " << epoch;
            return debug_string.str();
        }
    };

    struct BatchMatchResult {
        std::vector<BlockIdxType> matched_indices;
    };

    struct MatchResult {
        BlockIdxType matched_index;
    };

    struct PutResult {
        enum class Action {
            INSERTED,  // New item inserted
            REPLACED,  // Replaced old item (need to free old block, reference new block)
            SKIPPED    // Skipped (epoch priority, no update needed)
        };

        Action       action;
        BlockIdxType old_block_index;  // Old block index if replaced, otherwise NULL_BLOCK_IDX
    };

    using LRUCacheType  = LRUCache<CacheKeyGroupEpoch, CacheItem, CacheKeyGroupEpochHash, CacheKeyGroupEpochEqual>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    PutResult put(CacheItem& cache_item);

    // Remove ONLY the GLOBAL_EPOCH entry for (cache_key, group_id).
    // Batch-local entries (epoch > 0) are intentionally unaffected.
    // Returns the removed CacheItem if found, std::nullopt otherwise.
    std::optional<CacheItem> remove(CacheKeyType cache_key, int group_id = 0);

    // O(1): contains only checks the globally-visible entry.
    bool contains(CacheKeyType cache_key, int group_id = 0) const;

    // O(n): diagnostic/test helper that checks any epoch for (cache_key, group_id).
    bool containsAnyEpoch(CacheKeyType cache_key, int group_id = 0) const;

    // current_batch_epoch semantics:
    //   NO_EPOCH_FILTER (-1): bypass filter, match every item regardless of epoch
    //   GLOBAL_EPOCH    ( 0): no batch identity, match ONLY items with epoch == 0
    //                         (this is what feature-OFF callers and non-batch-bound paths use)
    //   >= 1                : same-batch caller — match global items OR items with the same epoch
    MatchResult match(CacheKeyType cache_key, int group_id = 0, int64_t current_batch_epoch = GLOBAL_EPOCH);

    BlockIndicesType pop(int n);

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

private:
    CacheKeyGroupEpoch makeKey(CacheKeyType cache_key, GroupIdType group_id, int64_t epoch) const;
    CacheKeyGroupEpoch makeKey(const CacheItem& item) const;

    size_t       seq_size_per_block_;
    LRUCacheType lru_cache_;
    // All public methods acquire mu_ before accessing lru_cache_.
    // This mutex is required because BlockCache is shared across scheduler and rpc threads.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
