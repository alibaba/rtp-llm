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

using CacheKeyGroupPair = std::pair<CacheKeyType, GroupIdType>;  // <cache_key, group_id>

class BlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key;
        GroupIdType  group_id;
        BlockIdxType block_index;
        bool         is_resident = false;
        int64_t      epoch       = 0;  // Epoch ID: 0 = global visible, >0 = batch-specific
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

    using LRUCacheType  = LRUCache<CacheKeyGroupPair,
                                   CacheItem,
                                   PairFirstHash<CacheKeyType, GroupIdType>,
                                   PairBothEqual<CacheKeyType, GroupIdType>>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    PutResult put(CacheItem& cache_item);

    // Remove a cache entry by key. Returns the removed CacheItem if found, std::nullopt otherwise.
    std::optional<CacheItem> remove(CacheKeyType cache_key, int group_id = 0);

    bool contains(CacheKeyType cache_key, int group_id = 0) const;

    static constexpr int64_t NO_EPOCH_FILTER = -1;

    MatchResult match(CacheKeyType cache_key, int group_id = 0, int64_t current_batch_epoch = NO_EPOCH_FILTER);

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
    size_t       seq_size_per_block_;
    LRUCacheType lru_cache_;
    // All public methods acquire mu_ before accessing lru_cache_.
    // This mutex is required because BlockCache is shared across scheduler and stream threads.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
