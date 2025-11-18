#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

BlockCacheV1::MatchResult BlockCacheV1::match(size_t cache_key, int group_id) {
    CacheKeyGroupPair key{cache_key, group_id};
    auto [success, item] = lru_cache_.get(key);
    if (success) {
        return {item.block_index};
    } else {
        return {NULL_BLOCK_IDX};
    }
}

bool BlockCacheV1::contains(size_t cache_key, int group_id) {
    CacheKeyGroupPair key{cache_key, group_id};
    return lru_cache_.contains(key);
}

bool BlockCacheV1::put(CacheItem& item) {
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    CacheKeyGroupPair key{item.cache_key, item.group_id};

    if (lru_cache_.contains(key)) {
        // 提升热度
        lru_cache_.get(key);
        return false;  // 已存在
    }

    lru_cache_.put(key, item);
    return true;
}

std::vector<BlockIdxType> BlockCacheV1::pop(int nums) {
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    std::vector<BlockIdxType> pop_blocks;

    std::lock_guard<std::mutex> lock(mutex_);

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

bool BlockCacheV1::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.empty();
}

size_t BlockCacheV1::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.size();
}

BlockCacheV1::CacheSnapshot BlockCacheV1::cacheSnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.cacheSnapshot(latest_version);
}

}  // namespace rtp_llm
