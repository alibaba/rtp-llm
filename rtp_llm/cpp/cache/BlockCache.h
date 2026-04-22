#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

class BlockPool;
using BlockPoolPtr = std::shared_ptr<BlockPool>;

class BlockCache {
public:
    struct CacheSlot {
        BlockIdxType block_id = NULL_BLOCK_IDX;
        bool         valid() const {
            return block_id != NULL_BLOCK_IDX;
        }
    };

    struct CacheItem {
        CacheKeyType cache_key;
        // slots[model_id][group_id]
        std::vector<std::vector<CacheSlot>> slots;
        bool                                is_resident = false;

        size_t totalValidSlots() const {
            size_t count = 0;
            for (const auto& model_slots : slots) {
                for (const auto& slot : model_slots) {
                    if (slot.valid())
                        ++count;
                }
            }
            return count;
        }

        size_t validSlotsForModel(size_t model_id) const {
            if (model_id >= slots.size())
                return 0;
            size_t count = 0;
            for (const auto& slot : slots[model_id]) {
                if (slot.valid())
                    ++count;
            }
            return count;
        }
    };

    struct MatchResult {
        BlockIdxType matched_index;
    };

    struct ModelRegistryEntry {
        size_t       model_id  = 0;
        size_t       group_num = 0;
        BlockPoolPtr block_pool;
    };

    using LRUCacheType  = LRUCache<CacheKeyType, CacheItem>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

    struct EvictResult {
        std::vector<CacheKeyType>                   evicted_keys;
        std::unordered_map<CacheKeyType, CacheItem> evicted_items;
    };

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    void   registerModel(size_t model_id, size_t group_num, BlockPoolPtr block_pool);
    size_t registeredModelNum() const;

    // Slot-level match: returns matched block_id for the given (model_id, group_id).
    MatchResult matchSlot(CacheKeyType cache_key, size_t model_id = 0, int group_id = 0);

    // Insert/update a single slot. Returns true if a new CacheItem was created.
    // Caller is responsible for blockCacheReference when return is true (or when
    // the slot is newly filled in an existing item).
    bool putSlot(CacheKeyType cache_key, size_t model_id, int group_id, BlockIdxType block_id, bool is_resident);

    bool containsSlot(CacheKeyType cache_key, size_t model_id = 0, int group_id = 0) const;

    // Remove the entire CacheItem for a cache_key. Returns the removed item if found.
    std::optional<CacheItem> removeItem(CacheKeyType cache_key);

    // Pop non-resident items from LRU tail, returning block indices.
    // Each pop removes an entire CacheItem (all slots).
    // Returns block indices for the trigger_model_id's slots (for backward compat with ensureFreeBlocks).
    BlockIndicesType pop(int n);

    // Pop and free blocks through registered BlockPools.
    // Returns the number of blocks freed for trigger_model_id.
    // Requires registerModel() to have been called.
    size_t popAndFree(size_t required_blocks, size_t trigger_model_id = 0);

    // Select and remove LRU cache entries until at least min_blocks are freed.
    // Skips resident entries. Returns evicted items grouped by cache_key (in LRU order).
    EvictResult selectAndEvict(size_t min_blocks);

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    void ensureSlots(CacheItem& item, size_t model_id, int group_id) const;
    void freeItemBlocks(const CacheItem& item);

    std::vector<ModelRegistryEntry> registry_;
    LRUCacheType                    lru_cache_;
    // BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe. Guard all accesses here.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
