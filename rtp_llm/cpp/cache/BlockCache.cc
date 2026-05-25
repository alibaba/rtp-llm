#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

CacheKeyGroupEpoch BlockCache::makeKey(CacheKeyType cache_key, GroupIdType group_id, int64_t epoch) const {
    return CacheKeyGroupEpoch{cache_key, group_id, epoch};
}

CacheKeyGroupEpoch BlockCache::makeKey(const CacheItem& item) const {
    return makeKey(item.cache_key, item.group_id, item.epoch);
}

BlockCache::MatchResult BlockCache::match(CacheKeyType cache_key, int group_id, int64_t current_batch_epoch) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto touch_if_present = [&](const CacheKeyGroupEpoch& key) -> std::optional<CacheItem> {
        auto [found, item] = lru_cache_.peek(key);
        if (!found) {
            return std::nullopt;
        }
        lru_cache_.get(key);
        return item;
    };

    if (current_batch_epoch >= 1) {
        if (auto item = touch_if_present(makeKey(cache_key, group_id, current_batch_epoch))) {
            return {item->block_index};
        }
        if (auto item = touch_if_present(makeKey(cache_key, group_id, GLOBAL_EPOCH))) {
            return {item->block_index};
        }
        return {NULL_BLOCK_IDX};
    }

    if (current_batch_epoch == GLOBAL_EPOCH) {
        if (auto item = touch_if_present(makeKey(cache_key, group_id, GLOBAL_EPOCH))) {
            return {item->block_index};
        }
        return {NULL_BLOCK_IDX};
    }

    if (current_batch_epoch == NO_EPOCH_FILTER) {
        if (auto item = touch_if_present(makeKey(cache_key, group_id, GLOBAL_EPOCH))) {
            return {item->block_index};
        }
        for (const auto& [key, item] : lru_cache_.items()) {
            if (key.cache_key == cache_key && key.group_id == group_id) {
                lru_cache_.get(key);
                return {item.block_index};
            }
        }
    }
    return {NULL_BLOCK_IDX};
}

bool BlockCache::contains(CacheKeyType cache_key, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [key, item] : lru_cache_.items()) {
        (void)item;
        if (key.cache_key == cache_key && key.group_id == group_id) {
            return true;
        }
    }
    return false;
}

BlockCache::PutResult BlockCache::put(CacheItem& item) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");
    RTP_LLM_CHECK_WITH_INFO(item.epoch >= 0,
                            "CacheItem.epoch must be >= 0 (got %ld); negative values are reserved as query-side "
                            "sentinels",
                            item.epoch);

    auto key               = makeKey(item);
    auto [found, old_item] = lru_cache_.peek(key);
    if (found) {
        // Preserve privileged states: a resident entry must not be downgraded
        // by a non-resident put.
        if (old_item.is_resident && !item.is_resident) {
            return {PutResult::Action::SKIPPED, NULL_BLOCK_IDX};
        }

        BlockIdxType old_block_index = old_item.block_index;
        lru_cache_.put(key, item);
        return {PutResult::Action::REPLACED, old_block_index};
    } else {
        if (item.epoch != GLOBAL_EPOCH) {
            auto [global_found, global_item] = lru_cache_.peek(makeKey(item.cache_key, item.group_id, GLOBAL_EPOCH));
            if (global_found) {
                return {PutResult::Action::SKIPPED, NULL_BLOCK_IDX};
            }
        }
        lru_cache_.put(key, item);
        return {PutResult::Action::INSERTED, NULL_BLOCK_IDX};
    }
}

BlockIndicesType BlockCache::pop(int nums) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    BlockIndicesType pop_blocks;

    // Phase 1: prefer evicting stale epoch>0 entries (batch-specific, likely dead)
    auto stale_cond = [](const CacheKeyGroupEpoch&, const CacheItem& item) {
        return !item.is_resident && item.epoch > 0;
    };
    while (nums > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(stale_cond);
        if (!success)
            break;
        pop_blocks.push_back(item.block_index);
        nums--;
    }

    // Phase 2: fall back to normal LRU eviction
    auto normal_cond = [](const CacheKeyGroupEpoch&, const CacheItem& item) { return !item.is_resident; };
    while (nums > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(normal_cond);
        if (!success)
            break;
        pop_blocks.push_back(item.block_index);
        nums--;
    }

    return pop_blocks;
}

std::optional<BlockCache::CacheItem> BlockCache::remove(CacheKeyType cache_key, int group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheItem                   removed_item;
    if (!lru_cache_.remove(makeKey(cache_key, group_id, GLOBAL_EPOCH), &removed_item)) {
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

    // Second pass: group non-resident items by cache_key in LRU order (back = least-recently-used).
    // Tiered eviction promotes selected entries to memory/remote cache via
    // storeCacheAsync, where there is no epoch concept — exposing a batch-local
    // (epoch>0) entry to that path would leak it into the global memory cache and
    // break batch isolation. Skip epoch>0 entries here; they are only reclaimed
    // through BlockCache::pop (Phase 1) which frees blocks locally without export.
    std::unordered_map<CacheKeyType, std::vector<CacheItem>> grouped_items;
    std::vector<CacheKeyType>                                lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident) {
            continue;
        }
        if (item.epoch != GLOBAL_EPOCH) {
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
            CacheItem removed_item;
            if (lru_cache_.remove(makeKey(item), &removed_item)) {
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
