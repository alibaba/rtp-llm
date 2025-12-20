#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"

namespace rtp_llm {

BlockCache::MatchResult BlockCache::match(size_t cache_key, int group_id) {
    CacheKeyGroupPair key{cache_key, group_id};
    auto [success, item] = lru_cache_.get(key);
    if (success) {
        return {item.block_index};
    } else {
        return {NULL_BLOCK_IDX};
    }
}

bool BlockCache::contains(size_t cache_key, int group_id) const {
    CacheKeyGroupPair key{cache_key, group_id};
    return lru_cache_.contains(key);
}

bool BlockCache::put(CacheItem& item) {
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    CacheKeyGroupPair key{item.cache_key, item.group_id};

    if (lru_cache_.contains(key)) {
        // It already exists; increase its popularity.
        lru_cache_.get(key);
        return false;
    }

    lru_cache_.put(key, item);
    return true;
}

BlockIndicesType BlockCache::pop(int nums) {
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    BlockIndicesType pop_blocks;

    auto cond = [&](const CacheKeyGroupPair& key, const CacheItem& item) { return !item.is_resident; };

    while (nums > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success)
            break;
        pop_blocks.push_back(item.block_index);
        nums--;
    }

    return pop_blocks;
}

bool BlockCache::empty() const {
    return lru_cache_.empty();
}

size_t BlockCache::size() const {
    return lru_cache_.size();
}

BlockCache::CacheSnapshot BlockCache::cacheSnapshot(int64_t latest_version) const {
    return lru_cache_.cacheSnapshot(latest_version);
}

}  // namespace rtp_llm
