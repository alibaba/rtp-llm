#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

BlockCache::MatchResult BlockCache::match(CacheKeyType cache_key, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
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
    RTP_LLM_PROFILE_FUNCTION();
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
    RTP_LLM_PROFILE_FUNCTION();
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

std::optional<BlockCache::CacheItem> BlockCache::remove(CacheKeyType cache_key, int group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    CacheItem                   removed_item;
    if (!lru_cache_.remove(key, &removed_item)) {
        return std::nullopt;
    }
    return removed_item;
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

BlockCache::EvictResult BlockCache::selectAndEvict(size_t min_blocks) {
    std::lock_guard<std::mutex> lock(mu_);

    EvictResult result;
    if (lru_cache_.empty()) {
        return result;
    }

    // First pass: collect resident keys
    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }

    // Second pass: group non-resident items by cache_key in LRU order (back = least-recently-used)
    std::unordered_map<CacheKeyType, std::vector<CacheItem>> grouped_items;
    std::vector<CacheKeyType>                                lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident) {
            continue;
        }
        auto [iter, inserted] = grouped_items.try_emplace(item.cache_key);
        if (inserted) {
            lru_keys.push_back(item.cache_key);
        }
        iter->second.push_back(item);
    }

    // Select keys until we have enough blocks
    std::vector<CacheKeyType> selected_keys;
    size_t                    selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        if (resident_keys.find(cache_key) != resident_keys.end()) {
            continue;
        }
        const auto group_it = grouped_items.find(cache_key);
        if (group_it == grouped_items.end() || group_it->second.empty()) {
            continue;
        }
        selected_keys.push_back(cache_key);
        selected_blocks += group_it->second.size();
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    if (selected_keys.empty()) {
        return result;
    }

    // Remove selected items from LRU cache and build result
    for (const auto cache_key : selected_keys) {
        auto&                  items = grouped_items.at(cache_key);
        std::vector<CacheItem> evicted_items;
        for (const auto& item : items) {
            CacheKeyGroupPair key{item.cache_key, item.group_id};
            CacheItem         removed_item;
            if (lru_cache_.remove(key, &removed_item)) {
                evicted_items.push_back(removed_item);
            }
        }
        if (!evicted_items.empty()) {
            result.evicted_keys.push_back(cache_key);
            result.evicted_items[cache_key] = std::move(evicted_items);
        }
    }

    return result;
}

}  // namespace rtp_llm
