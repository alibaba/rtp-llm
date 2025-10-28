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

BlockCacheV1::MatchResult BlockCacheV1::match(CacheKeyType cache_key) {
    const auto& [success, item] = lru_cache_.get(cache_key);
    if (success) {
        return {item.block_index, item.loss};
    } else {
        return {NULL_BLOCK_IDX, {}};
    }
}

bool BlockCacheV1::isExistKey(CacheKeyType cache_key) {
    return lru_cache_.contains(cache_key);
}

// TODO, 如果有重复的，怎么办。
bool BlockCacheV1::put(CacheItem& item) {
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    if (lru_cache_.contains(item.cache_key)) {
        // Increase old matched item's popularity
        lru_cache_.get(item.cache_key);
        return false;
    }

    lru_cache_.put(item.cache_key, item);

    return true;
}

std::vector<BlockIdxType> BlockCacheV1::pop(int nums) {
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    vector<BlockIdxType>   pop_blocks;
    std::vector<CacheItem> resident_list;

    std::lock_guard<std::mutex> lock(mutex_);

    auto cond = [&](const CacheKeyType& key, const CacheItem& item) { return !item.is_resident; };

    while (!lru_cache_.empty()) {
        auto [success, cache_item] = lru_cache_.popWithCond(cond);
        if (!success) {
            break;
        }
        pop_blocks.push_back(cache_item.block_index);
        nums--;
        if (nums == 0) {
            break;
        }
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
