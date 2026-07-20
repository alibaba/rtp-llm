#include "rtp_llm/cpp/cache/BlockCache.h"

#include <algorithm>

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
    if (event_publisher_) {
        auto& group_count = key_group_counts_[item.cache_key];
        ++group_count;
        if (group_count == static_cast<size_t>(required_group_count_)) {
            // tryPublish never waits for network or queue capacity. Keeping it under
            // mu_ preserves the same ordering as cache state transitions.
            (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_ADD, item.cache_key, 0});
        }
    }
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
        if (event_publisher_) {
            auto count_it = key_group_counts_.find(item.cache_key);
            if (count_it != key_group_counts_.end()) {
                const bool was_complete = count_it->second >= static_cast<size_t>(required_group_count_);
                if (count_it->second > 0) {
                    --count_it->second;
                }
                if (was_complete && count_it->second < static_cast<size_t>(required_group_count_)) {
                    (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, item.cache_key, 0});
                }
                if (count_it->second == 0) {
                    key_group_counts_.erase(count_it);
                }
            }
        }
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
    if (event_publisher_) {
        auto count_it = key_group_counts_.find(cache_key);
        if (count_it != key_group_counts_.end()) {
            const bool was_complete = count_it->second >= static_cast<size_t>(required_group_count_);
            if (count_it->second > 0) {
                --count_it->second;
            }
            if (was_complete && count_it->second < static_cast<size_t>(required_group_count_)) {
                (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, cache_key, 0});
            }
            if (count_it->second == 0) {
                key_group_counts_.erase(count_it);
            }
        }
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

BlockCache::LogicalCacheSnapshot BlockCache::logicalCacheSnapshot() const {
    std::lock_guard<std::mutex>              lock(mu_);
    const auto                               snapshot = lru_cache_.cacheSnapshot(-1);
    std::unordered_map<CacheKeyType, size_t> group_counts;
    group_counts.reserve(snapshot.values.size());
    for (const auto& item : snapshot.values) {
        ++group_counts[item.cache_key];
    }

    LogicalCacheSnapshot logical_snapshot;
    logical_snapshot.version = snapshot.version;
    logical_snapshot.cache_keys.reserve(group_counts.size());
    for (const auto& [cache_key, group_count] : group_counts) {
        if (group_count >= static_cast<size_t>(required_group_count_)) {
            logical_snapshot.cache_keys.push_back(cache_key);
        }
    }
    std::sort(logical_snapshot.cache_keys.begin(), logical_snapshot.cache_keys.end());
    return logical_snapshot;
}

void BlockCache::setEventPublisher(KVCacheEventPublisherPtr publisher, int required_group_count) {
    std::lock_guard<std::mutex> lock(mu_);
    event_publisher_      = std::move(publisher);
    required_group_count_ = std::max(required_group_count, 1);
    key_group_counts_.clear();
    if (!event_publisher_) {
        return;
    }
    for (const auto& [key, item] : lru_cache_.items()) {
        (void)key;
        ++key_group_counts_[item.cache_key];
    }
}

BlockCache::EvictResult BlockCache::selectAndEvict(size_t min_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    EvictResult                 result;

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
        const auto             count_it     = key_group_counts_.find(cache_key);
        const bool             was_complete = event_publisher_ && count_it != key_group_counts_.end()
                                  && count_it->second >= static_cast<size_t>(required_group_count_);
        for (const auto& item : items) {
            CacheKeyGroupPair key{item.cache_key, item.group_id};
            CacheItem         removed_item;
            if (lru_cache_.remove(key, &removed_item)) {
                evicted_items.push_back(removed_item);
                if (event_publisher_) {
                    auto current_count = key_group_counts_.find(cache_key);
                    if (current_count != key_group_counts_.end() && current_count->second > 0) {
                        --current_count->second;
                    }
                }
            }
        }
        if (!evicted_items.empty()) {
            result.evicted_keys.push_back(cache_key);
            result.evicted_items[cache_key] = std::move(evicted_items);
        }
        if (event_publisher_) {
            auto         current_count = key_group_counts_.find(cache_key);
            const size_t remaining     = current_count == key_group_counts_.end() ? 0 : current_count->second;
            if (was_complete && remaining < static_cast<size_t>(required_group_count_)) {
                (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, cache_key, 0});
            }
            if (current_count != key_group_counts_.end() && current_count->second == 0) {
                key_group_counts_.erase(current_count);
            }
        }
    }

    return result;
}

}  // namespace rtp_llm
