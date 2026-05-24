#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

#include <algorithm>
#include <mutex>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

MemoryDiskBlockCache::MatchResult MemoryDiskBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return {};
    }
    touchLocked(it->second);
    const auto& item = it->second;
    return {item.backing_type, item.block_index, item.disk_slot, item.block_size, item.is_complete};
}

MemoryDiskBlockCache::MatchResult MemoryDiskBlockCache::matchAndMarkInFlight(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return {};
    }
    touchLocked(it->second);
    it->second.in_flight_ref++;
    const auto& item = it->second;
    return {item.backing_type, item.block_index, item.disk_slot, item.block_size, item.is_complete};
}

bool MemoryDiskBlockCache::contains(CacheKeyType cache_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return items_.find(cache_key) != items_.end();
}

std::pair<bool, std::optional<MemoryDiskBlockCache::CacheItem>>
MemoryDiskBlockCache::putCommitted(const CacheItem& input_item) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(validItem(input_item), "invalid cache item backing fields");

    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                item = input_item;
    item.in_flight_ref                       = 0;

    auto existing = items_.find(item.cache_key);
    if (existing != items_.end()) {
        touchLocked(existing->second);
        if (!existing->second.is_complete && item.is_complete) {
            if (existing->second.in_flight_ref > 0) {
                return {false, std::nullopt};
            }
            auto old_item = existing->second;
            eraseEvictKeyLocked(existing->second);
            item.last_access_seq = ++access_seq_;
            existing->second     = item;
            insertEvictKeyLocked(existing->second);
            return {true, old_item};
        }
        return {false, std::nullopt};
    }

    item.last_access_seq = ++access_seq_;
    auto [it, inserted]  = items_.emplace(item.cache_key, item);
    (void)inserted;
    insertEvictKeyLocked(it->second);
    return {true, std::nullopt};
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::removeIfMatch(CacheKeyType     cache_key,
                                                                                   CacheBackingType backing_type,
                                                                                   BlockIdxType expected_block_index,
                                                                                   int32_t      expected_disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end() || it->second.backing_type != backing_type) {
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::MEMORY && it->second.block_index != expected_block_index) {
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::DISK && it->second.disk_slot != expected_disk_slot) {
        return std::nullopt;
    }
    auto removed_item = it->second;
    eraseEvictKeyLocked(it->second);
    items_.erase(it);
    return removed_item;
}

std::pair<bool, std::optional<MemoryBlockCache::CacheItem>>
MemoryDiskBlockCache::put(const MemoryBlockCache::CacheItem& input_item) {
    CacheItem item;
    item.cache_key    = input_item.cache_key;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = input_item.block_index;
    item.disk_slot    = -1;
    item.block_size   = input_item.block_size;
    item.is_resident  = input_item.is_resident;
    item.is_complete  = input_item.is_complete;
    auto [ok, popped] = putCommitted(item);
    if (!popped.has_value()) {
        return {ok, std::nullopt};
    }
    return {ok, toMemoryCacheItem(*popped)};
}

std::optional<MemoryBlockCache::CacheItem> MemoryDiskBlockCache::remove(CacheKeyType cache_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return std::nullopt;
    }
    auto removed_item = it->second;
    eraseEvictKeyLocked(it->second);
    items_.erase(it);
    return toMemoryCacheItem(removed_item);
}

std::optional<MemoryBlockCache::CacheItem> MemoryDiskBlockCache::removeIfMatch(CacheKeyType cache_key,
                                                                               BlockIdxType expected_block_index) {
    auto removed = removeIfMatch(cache_key, CacheBackingType::MEMORY, expected_block_index, -1);
    if (!removed.has_value()) {
        return std::nullopt;
    }
    return toMemoryCacheItem(*removed);
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::popOldestEvictable() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                memory_item = oldestFromSetLocked(memory_lru_);
    auto                                disk_item   = oldestFromSetLocked(disk_lru_);
    if (!memory_item.has_value()) {
        if (!disk_item.has_value()) {
            return std::nullopt;
        }
        auto it = items_.find(disk_item->cache_key);
        if (it != items_.end()) {
            eraseEvictKeyLocked(it->second);
            items_.erase(it);
        }
        return disk_item;
    }
    if (!disk_item.has_value() || memory_item->last_access_seq <= disk_item->last_access_seq) {
        auto it = items_.find(memory_item->cache_key);
        if (it != items_.end()) {
            eraseEvictKeyLocked(it->second);
            items_.erase(it);
        }
        return memory_item;
    }
    auto it = items_.find(disk_item->cache_key);
    if (it != items_.end()) {
        eraseEvictKeyLocked(it->second);
        items_.erase(it);
    }
    return disk_item;
}

bool MemoryDiskBlockCache::markInFlight(CacheKeyType     cache_key,
                                        CacheBackingType backing_type,
                                        BlockIdxType     block_index,
                                        int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end() || it->second.backing_type != backing_type) {
        return false;
    }
    if (backing_type == CacheBackingType::MEMORY && it->second.block_index != block_index) {
        return false;
    }
    if (backing_type == CacheBackingType::DISK && it->second.disk_slot != disk_slot) {
        return false;
    }
    it->second.in_flight_ref++;
    return true;
}

void MemoryDiskBlockCache::releaseInFlight(CacheKeyType     cache_key,
                                           CacheBackingType backing_type,
                                           BlockIdxType     block_index,
                                           int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end() || it->second.backing_type != backing_type) {
        return;
    }
    if (backing_type == CacheBackingType::MEMORY && it->second.block_index != block_index) {
        return;
    }
    if (backing_type == CacheBackingType::DISK && it->second.disk_slot != disk_slot) {
        return;
    }
    if (it->second.in_flight_ref > 0) {
        it->second.in_flight_ref--;
    }
}

bool MemoryDiskBlockCache::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return items_.empty();
}

size_t MemoryDiskBlockCache::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return items_.size();
}

std::vector<CacheKeyType> MemoryDiskBlockCache::cacheKeys() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheItem>              values;
    values.reserve(items_.size());
    for (const auto& [_, item] : items_) {
        values.push_back(item);
    }
    std::sort(values.begin(), values.end(), [](const CacheItem& lhs, const CacheItem& rhs) {
        return lhs.last_access_seq > rhs.last_access_seq;
    });
    std::vector<CacheKeyType> keys;
    keys.reserve(values.size());
    for (const auto& item : values) {
        keys.push_back(item.cache_key);
    }
    return keys;
}

bool MemoryDiskBlockCache::validItem(const CacheItem& item) const {
    if (item.backing_type == CacheBackingType::MEMORY) {
        return !isNullBlockIdx(item.block_index) && item.disk_slot < 0;
    }
    if (item.backing_type == CacheBackingType::DISK) {
        return isNullBlockIdx(item.block_index) && item.disk_slot >= 0;
    }
    return false;
}

MemoryBlockCache::CacheItem MemoryDiskBlockCache::toMemoryCacheItem(const CacheItem& item) {
    MemoryBlockCache::CacheItem memory_item;
    memory_item.cache_key   = item.cache_key;
    memory_item.block_index = item.block_index;
    memory_item.block_size  = item.block_size;
    memory_item.is_resident = item.is_resident;
    memory_item.is_complete = item.is_complete;
    return memory_item;
}

void MemoryDiskBlockCache::insertEvictKeyLocked(const CacheItem& item) {
    auto& eviction_set = item.backing_type == CacheBackingType::MEMORY ? memory_lru_ : disk_lru_;
    eviction_set.insert(EvictKey{item.last_access_seq, item.cache_key});
}

void MemoryDiskBlockCache::eraseEvictKeyLocked(const CacheItem& item) {
    auto& eviction_set = item.backing_type == CacheBackingType::MEMORY ? memory_lru_ : disk_lru_;
    eviction_set.erase(EvictKey{item.last_access_seq, item.cache_key});
}

void MemoryDiskBlockCache::touchLocked(CacheItem& item) {
    eraseEvictKeyLocked(item);
    item.last_access_seq = ++access_seq_;
    insertEvictKeyLocked(item);
}

std::optional<MemoryDiskBlockCache::CacheItem>
MemoryDiskBlockCache::oldestFromSetLocked(std::set<EvictKey>& eviction_set) {
    for (auto evict_it = eviction_set.begin(); evict_it != eviction_set.end();) {
        const auto key = *evict_it;
        auto       it  = items_.find(key.cache_key);
        if (it == items_.end() || it->second.last_access_seq != key.last_access_seq) {
            evict_it = eviction_set.erase(evict_it);
            continue;
        }
        if (it->second.is_resident || it->second.in_flight_ref > 0) {
            ++evict_it;
            continue;
        }
        return it->second;
    }
    return std::nullopt;
}

}  // namespace rtp_llm
