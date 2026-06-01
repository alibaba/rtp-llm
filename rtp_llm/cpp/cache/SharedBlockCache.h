#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BlockPool.h"

namespace rtp_llm {

class SharedBlockCache {
public:
    struct UnifiedCacheItem {
        CacheKeyType              cache_key;
        bool                      is_resident = false;
        std::vector<BlockIdxType> slots;
    };

    struct EvictResult {
        std::vector<CacheKeyType>                                   evicted_keys;
        std::unordered_map<CacheKeyType, std::vector<BlockIdxType>> evicted_slots;
    };

    struct MatchResult {
        bool                      found = false;
        std::vector<BlockIdxType> group_blocks;
    };

    using LRUCacheType = LRUCache<CacheKeyType, UnifiedCacheItem>;

public:
    explicit SharedBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    void init(int group_num, const std::vector<BlockPoolPtr>& group_pools);

    void put(CacheKeyType cache_key, const std::vector<BlockIdxType>& group_slots, bool is_resident);

    MatchResult match(CacheKeyType cache_key);

    BlockIdxType matchGroup(CacheKeyType cache_key, int group_id);

    EvictResult selectAndEvict(size_t min_blocks);

    size_t evictAndFree(size_t min_blocks);

    std::optional<UnifiedCacheItem> remove(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    bool empty() const;

    size_t size() const;

    std::vector<CacheKeyType> allCacheKeys() const;

    int64_t version() const;

private:
    static const size_t kCacheMaxCapacity = 10000000;

    LRUCacheType       lru_cache_;
    mutable std::mutex mu_;
    int64_t            version_{0};

    int                       group_num_ = 0;
    std::vector<BlockPoolPtr> group_pools_;
};

using SharedBlockCachePtr = std::shared_ptr<SharedBlockCache>;

}  // namespace rtp_llm
