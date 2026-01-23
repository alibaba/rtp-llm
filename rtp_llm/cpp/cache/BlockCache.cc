#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockCache::MatchResult BlockCache::match(CacheKeyType cache_key, int group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    auto [success, item] = lru_cache_.get(key);
    if (success) {
        return {item.block_index};
    } else {
        return {NULL_BLOCK_IDX};
    }
}

bool BlockCache::contains(CacheKeyType cache_key, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    return lru_cache_.contains(key);
}

bool BlockCache::put(CacheItem& item) {
    std::lock_guard<std::mutex> lock(mu_);
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
    std::lock_guard<std::mutex> lock(mu_);
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
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.empty();
}

size_t BlockCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.size();
}

BlockCache::CacheSnapshot BlockCache::cacheSnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.cacheSnapshot(latest_version);
}

std::tuple<int64_t, std::vector<int64_t>> BlockCache::getVersionAndCacheKeys(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t                     current_version = lru_cache_.getVersion();
    std::vector<int64_t>        cachekeys;

    if (latest_version < current_version) {
        std::unordered_set<int64_t> seen_keys;
        lru_cache_.forEachValue([&](const CacheItem& item) {
            for (const auto& key_part : item.cache_key) {
                if (seen_keys.insert(key_part).second) {
                    cachekeys.push_back(key_part);
                }
            }
        });
    }

    return {current_version, std::move(cachekeys)};
}

}  // namespace rtp_llm
