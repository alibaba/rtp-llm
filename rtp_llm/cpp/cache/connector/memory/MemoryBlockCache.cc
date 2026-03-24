#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

MemoryBlockCache::MatchResult MemoryBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto& [success, item] = lru_cache_.get(cache_key);
    if (success) {
        return {item.block_index, item.block_size, item.is_complete};
    } else {
        return {NULL_BLOCK_IDX, 0, false};
    }
}

bool MemoryBlockCache::contains(CacheKeyType cache_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return lru_cache_.contains(cache_key);
}

std::pair<bool, std::optional<MemoryBlockCache::CacheItem>> MemoryBlockCache::put(const CacheItem& item) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null");

    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (lru_cache_.contains(item.cache_key)) {
        // Key exists:
        // - Always increase old matched item's popularity
        // - Allow "partial -> complete" upgrade (same cache_key) by replacing the stored item.
        //   Return the old item as "popped" so the caller can free the old block.
        const auto& [success, old_item] = lru_cache_.get(item.cache_key);
        if (success && !old_item.is_complete && item.is_complete) {
            lru_cache_.put(item.cache_key, item);
            return {true, old_item};
        }
        return {false, std::nullopt};
    }

    std::optional<CacheItem> popped_item;
    if (lru_cache_.full()) {
        auto [success, popped_cache_item] = lru_cache_.pop();
        if (!success) {
            RTP_LLM_LOG_ERROR("put item failed, cache is full but pop item failed, cache key: %ld, cache size: %lu",
                              item.cache_key,
                              lru_cache_.size());
            return {false, std::nullopt};
        }
        popped_item = popped_cache_item;
    }

    lru_cache_.put(item.cache_key, item);
    return {true, popped_item};
}

std::optional<MemoryBlockCache::CacheItem> MemoryBlockCache::remove(CacheKeyType cache_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    CacheItem                           removed_item;
    if (!lru_cache_.remove(cache_key, &removed_item)) {
        return std::nullopt;
    }
    return removed_item;
}

std::optional<MemoryBlockCache::CacheItem> MemoryBlockCache::removeIfMatch(CacheKeyType cache_key,
                                                                           BlockIdxType expected_block_index) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto& [found, item] = lru_cache_.get(cache_key);
    if (!found || item.block_index != expected_block_index) {
        return std::nullopt;
    }
    CacheItem removed_item;
    lru_cache_.remove(cache_key, &removed_item);
    return removed_item;
}

std::vector<BlockIdxType> MemoryBlockCache::pop(int n) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(n > 0, "pop n should > 0, n = " + std::to_string(n));
    std::vector<BlockIdxType> pop_blocks;

    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto cond = [&](const CacheKeyType& /*key*/, const CacheItem& item) { return !item.is_resident; };

    while (!lru_cache_.empty()) {
        auto [success, cache_item] = lru_cache_.popWithCond(cond);
        if (!success) {
            break;
        }
        pop_blocks.push_back(cache_item.block_index);
        --n;
        if (n == 0) {
            break;
        }
    }

    return pop_blocks;
}

bool MemoryBlockCache::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return lru_cache_.empty();
}

size_t MemoryBlockCache::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return lru_cache_.size();
}

std::vector<CacheKeyType> MemoryBlockCache::cacheKeys() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheKeyType>           keys;
    keys.reserve(lru_cache_.size());
    for (const auto& it : lru_cache_.items()) {
        keys.push_back(it.first);
    }
    return keys;
}

}  // namespace rtp_llm
