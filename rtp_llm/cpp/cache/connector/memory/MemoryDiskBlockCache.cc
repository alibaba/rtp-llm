#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

#include <algorithm>
#include <mutex>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

bool isSplitRole(CacheBackingRole role) {
    return role == CacheBackingRole::KV || role == CacheBackingRole::TAIL;
}

}  // namespace

MemoryDiskBlockCache::MatchResult MemoryDiskBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return {};
    }

    auto& stored = it->second;
    if (stored.legacy.has_value()) {
        touchLocked(*stored.legacy);
        const auto& item = *stored.legacy;
        MatchResult result;
        result.found          = true;
        result.has_kv         = true;
        result.has_tail       = item.is_complete;
        result.backing_role   = CacheBackingRole::LEGACY;
        result.backing_type   = item.backing_type;
        result.matched_index  = item.block_index;
        result.disk_slot      = item.disk_slot;
        result.block_size     = item.block_size;
        result.is_complete    = item.is_complete;
        return result;
    }

    if (!stored.kv.has_value()) {
        return {};
    }
    touchLocked(*stored.kv);
    if (stored.tail.has_value()) {
        touchLocked(*stored.tail);
    }

    const auto& kv = *stored.kv;
    MatchResult result;
    result.found          = true;
    result.has_kv         = true;
    result.has_tail       = stored.tail.has_value();
    result.backing_role   = CacheBackingRole::KV;
    result.backing_type   = kv.backing_type;
    result.matched_index  = kv.block_index;
    result.disk_slot      = kv.disk_slot;
    result.block_size     = kv.block_size;
    result.is_complete    = result.has_tail;
    if (stored.tail.has_value()) {
        const auto& tail          = *stored.tail;
        result.tail_backing_type  = tail.backing_type;
        result.tail_matched_index = tail.block_index;
        result.tail_disk_slot     = tail.disk_slot;
        result.tail_block_size    = tail.block_size;
    }
    return result;
}

MemoryDiskBlockCache::MatchResult MemoryDiskBlockCache::matchAndMarkInFlight(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return {};
    }
    auto& stored = it->second;
    if (stored.legacy.has_value()) {
        touchLocked(*stored.legacy);
        stored.legacy->in_flight_ref++;
        const auto& item = *stored.legacy;
        return {true,
                true,
                item.is_complete,
                CacheBackingRole::LEGACY,
                item.backing_type,
                item.block_index,
                item.disk_slot,
                item.block_size,
                item.is_complete};
    }
    if (!stored.kv.has_value()) {
        return {};
    }
    touchLocked(*stored.kv);
    stored.kv->in_flight_ref++;
    const auto& kv = *stored.kv;
    return {true,
            true,
            stored.tail.has_value(),
            CacheBackingRole::KV,
            kv.backing_type,
            kv.block_index,
            kv.disk_slot,
            kv.block_size,
            stored.tail.has_value()};
}

MemoryDiskBlockCache::MatchResult MemoryDiskBlockCache::matchAndMarkInFlight(CacheKeyType     cache_key,
                                                                              CacheBackingRole role) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return {};
    }
    auto backing = mutableBackingLocked(it->second, role);
    if (!backing.has_value()) {
        return {};
    }
    auto* item = backing.value();
    touchLocked(*item);
    item->in_flight_ref++;
    return {true,
            role == CacheBackingRole::KV,
            role == CacheBackingRole::TAIL,
            role,
            item->backing_type,
            item->block_index,
            item->disk_slot,
            item->block_size,
            role == CacheBackingRole::TAIL || (role == CacheBackingRole::LEGACY && item->is_complete)};
}

bool MemoryDiskBlockCache::contains(CacheKeyType cache_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return items_.find(cache_key) != items_.end();
}

bool MemoryDiskBlockCache::contains(CacheKeyType cache_key, CacheBackingRole role) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return false;
    }
    return backingLocked(it->second, role).has_value();
}

std::pair<bool, std::optional<MemoryDiskBlockCache::CacheItem>>
MemoryDiskBlockCache::putCommitted(const CacheItem& input_item) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(validItem(input_item), "invalid cache item backing fields");

    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                item = input_item;
    item.backing_role                        = normalizeRole(item);
    item.in_flight_ref                       = 0;
    if (item.backing_role == CacheBackingRole::KV) {
        item.is_complete = false;
    } else if (item.backing_role == CacheBackingRole::TAIL) {
        item.is_complete = true;
    }

    if (item.backing_role == CacheBackingRole::LEGACY) {
        auto existing = items_.find(item.cache_key);
        if (existing != items_.end()) {
            if (!existing->second.legacy.has_value()) {
                return {false, std::nullopt};
            }
            touchLocked(*existing->second.legacy);
            if (!existing->second.legacy->is_complete && item.is_complete) {
                if (existing->second.legacy->in_flight_ref > 0) {
                    return {false, std::nullopt};
                }
                auto old_item = *existing->second.legacy;
                eraseEvictKeyLocked(*existing->second.legacy);
                item.last_access_seq     = ++access_seq_;
                existing->second.legacy  = item;
                insertEvictKeyLocked(*existing->second.legacy);
                return {true, old_item};
            }
            return {false, std::nullopt};
        }

        item.last_access_seq = ++access_seq_;
        StoredItem stored;
        stored.legacy        = item;
        auto [it, inserted]  = items_.emplace(item.cache_key, std::move(stored));
        (void)inserted;
        insertEvictKeyLocked(*it->second.legacy);
        return {true, std::nullopt};
    }

    auto [it, inserted] = items_.try_emplace(item.cache_key);
    auto& stored        = it->second;
    if (stored.legacy.has_value()) {
        if (inserted && emptyStoredItem(stored)) {
            items_.erase(it);
        }
        return {false, std::nullopt};
    }

    if (item.backing_role == CacheBackingRole::KV) {
        if (stored.kv.has_value()) {
            return {false, std::nullopt};
        }
        item.last_access_seq = ++access_seq_;
        stored.kv            = item;
        insertEvictKeyLocked(*stored.kv);
        return {true, std::nullopt};
    }

    if (!stored.kv.has_value()) {
        if (inserted && emptyStoredItem(stored)) {
            items_.erase(it);
        }
        return {false, std::nullopt};
    }
    if (stored.kv->in_flight_ref > 0) {
        return {false, std::nullopt};
    }
    if (stored.tail.has_value()) {
        touchLocked(*stored.tail);
        if (stored.tail->in_flight_ref > 0) {
            return {false, std::nullopt};
        }
        auto old_item = *stored.tail;
        eraseEvictKeyLocked(*stored.tail);
        item.last_access_seq = ++access_seq_;
        stored.tail          = item;
        insertEvictKeyLocked(*stored.tail);
        return {true, old_item};
    }

    item.last_access_seq = ++access_seq_;
    stored.tail          = item;
    insertEvictKeyLocked(*stored.tail);
    return {true, std::nullopt};
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::removeIfMatch(CacheKeyType     cache_key,
                                                                                   CacheBackingType backing_type,
                                                                                   BlockIdxType expected_block_index,
                                                                                   int32_t      expected_disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return std::nullopt;
    }
    for (auto role : {CacheBackingRole::LEGACY, CacheBackingRole::KV, CacheBackingRole::TAIL}) {
        auto backing = backingLocked(it->second, role);
        if (!backing.has_value() || !backingMatchesLocked(*backing.value(), backing_type, expected_block_index, expected_disk_slot)) {
            continue;
        }
        auto removed = removeBackingLocked(cache_key, role);
        return removed.empty() ? std::nullopt : std::optional<CacheItem>(removed.front());
    }
    return std::nullopt;
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::removeIfMatch(CacheKeyType     cache_key,
                                                                                   CacheBackingRole role,
                                                                                   CacheBackingType backing_type,
                                                                                   BlockIdxType expected_block_index,
                                                                                   int32_t      expected_disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return std::nullopt;
    }
    auto backing = backingLocked(it->second, role);
    if (!backing.has_value() || !backingMatchesLocked(*backing.value(), backing_type, expected_block_index, expected_disk_slot)) {
        return std::nullopt;
    }
    auto removed = removeBackingLocked(cache_key, role);
    return removed.empty() ? std::nullopt : std::optional<CacheItem>(removed.front());
}

std::pair<bool, std::optional<MemoryBlockCache::CacheItem>>
MemoryDiskBlockCache::put(const MemoryBlockCache::CacheItem& input_item) {
    CacheItem item;
    item.cache_key    = input_item.cache_key;
    item.backing_role = CacheBackingRole::LEGACY;
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
    auto removed = removeAll(cache_key);
    if (removed.empty()) {
        return std::nullopt;
    }
    return toMemoryCacheItem(removed.front());
}

std::vector<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::removeAll(CacheKeyType cache_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheItem>              removed;
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return removed;
    }
    if (it->second.legacy.has_value()) {
        eraseEvictKeyLocked(*it->second.legacy);
        removed.push_back(*it->second.legacy);
    }
    if (it->second.kv.has_value()) {
        eraseEvictKeyLocked(*it->second.kv);
        removed.push_back(*it->second.kv);
    }
    if (it->second.tail.has_value()) {
        eraseEvictKeyLocked(*it->second.tail);
        removed.push_back(*it->second.tail);
    }
    items_.erase(it);
    return removed;
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
    std::optional<CacheItem>            selected;
    auto                                consider = [&selected](const std::optional<CacheItem>& candidate) {
        if (!candidate.has_value()) {
            return;
        }
        if (!selected.has_value() || candidate->last_access_seq < selected->last_access_seq) {
            selected = candidate;
        }
    };
    consider(oldestFromSetLocked(memory_complete_lru_, /*single_backing_eviction=*/true));
    consider(oldestFromSetLocked(memory_incomplete_lru_, /*single_backing_eviction=*/true));
    consider(oldestFromSetLocked(disk_complete_lru_, /*single_backing_eviction=*/true));
    consider(oldestFromSetLocked(disk_incomplete_lru_, /*single_backing_eviction=*/true));
    if (!selected.has_value()) {
        return std::nullopt;
    }
    auto removed = removeBackingLocked(selected->cache_key, selected->backing_role);
    return removed.empty() ? std::nullopt : std::optional<CacheItem>(removed.front());
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::popOldestEvictable(CacheBlockKind kind) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return popOldestEvictableLocked(kind);
}

std::vector<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::popOldestEvictableBackings(CacheBlockKind kind) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto memory_item = oldestFromSetLocked(lruSetLocked(CacheBackingType::MEMORY, kind));
    auto disk_item   = oldestFromSetLocked(lruSetLocked(CacheBackingType::DISK, kind));
    std::optional<CacheItem> selected;
    if (!memory_item.has_value()) {
        selected = disk_item;
    } else if (!disk_item.has_value() || memory_item->last_access_seq <= disk_item->last_access_seq) {
        selected = memory_item;
    } else {
        selected = disk_item;
    }
    if (!selected.has_value()) {
        return {};
    }
    if (selected->backing_role == CacheBackingRole::KV) {
        auto all = removeBackingLocked(selected->cache_key, CacheBackingRole::KV);
        auto tail = removeBackingLocked(selected->cache_key, CacheBackingRole::TAIL);
        all.insert(all.end(), tail.begin(), tail.end());
        return all;
    }
    return removeBackingLocked(selected->cache_key, selected->backing_role);
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::popOldestEvictableLocked(CacheBlockKind kind) {
    auto memory_item = oldestFromSetLocked(lruSetLocked(CacheBackingType::MEMORY, kind));
    auto disk_item   = oldestFromSetLocked(lruSetLocked(CacheBackingType::DISK, kind));
    std::optional<CacheItem> selected;
    if (!memory_item.has_value()) {
        selected = disk_item;
    } else if (!disk_item.has_value() || memory_item->last_access_seq <= disk_item->last_access_seq) {
        selected = memory_item;
    } else {
        selected = disk_item;
    }
    if (!selected.has_value()) {
        return std::nullopt;
    }
    auto removed = removeBackingLocked(selected->cache_key, selected->backing_role);
    return removed.empty() ? std::nullopt : std::optional<CacheItem>(removed.front());
}

bool MemoryDiskBlockCache::markInFlight(CacheKeyType     cache_key,
                                        CacheBackingType backing_type,
                                        BlockIdxType     block_index,
                                        int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return false;
    }
    for (auto role : {CacheBackingRole::LEGACY, CacheBackingRole::KV, CacheBackingRole::TAIL}) {
        auto backing = mutableBackingLocked(it->second, role);
        if (backing.has_value() && backingMatchesLocked(*backing.value(), backing_type, block_index, disk_slot)) {
            backing.value()->in_flight_ref++;
            return true;
        }
    }
    return false;
}

bool MemoryDiskBlockCache::markInFlight(CacheKeyType     cache_key,
                                        CacheBackingRole role,
                                        CacheBackingType backing_type,
                                        BlockIdxType     block_index,
                                        int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return false;
    }
    auto backing = mutableBackingLocked(it->second, role);
    if (!backing.has_value() || !backingMatchesLocked(*backing.value(), backing_type, block_index, disk_slot)) {
        return false;
    }
    backing.value()->in_flight_ref++;
    return true;
}

void MemoryDiskBlockCache::releaseInFlight(CacheKeyType     cache_key,
                                           CacheBackingType backing_type,
                                           BlockIdxType     block_index,
                                           int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return;
    }
    for (auto role : {CacheBackingRole::LEGACY, CacheBackingRole::KV, CacheBackingRole::TAIL}) {
        auto backing = mutableBackingLocked(it->second, role);
        if (!backing.has_value() || !backingMatchesLocked(*backing.value(), backing_type, block_index, disk_slot)) {
            continue;
        }
        if (backing.value()->in_flight_ref > 0) {
            backing.value()->in_flight_ref--;
        }
        return;
    }
}

void MemoryDiskBlockCache::releaseInFlight(CacheKeyType     cache_key,
                                           CacheBackingRole role,
                                           CacheBackingType backing_type,
                                           BlockIdxType     block_index,
                                           int32_t          disk_slot) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = items_.find(cache_key);
    if (it == items_.end()) {
        return;
    }
    auto backing = mutableBackingLocked(it->second, role);
    if (!backing.has_value() || !backingMatchesLocked(*backing.value(), backing_type, block_index, disk_slot)) {
        return;
    }
    if (backing.value()->in_flight_ref > 0) {
        backing.value()->in_flight_ref--;
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
    for (const auto& [_, stored] : items_) {
        if (stored.legacy.has_value()) {
            values.push_back(*stored.legacy);
        } else if (stored.kv.has_value()) {
            auto representative = *stored.kv;
            if (stored.tail.has_value() && stored.tail->last_access_seq > representative.last_access_seq) {
                representative.last_access_seq = stored.tail->last_access_seq;
            }
            values.push_back(representative);
        } else if (stored.tail.has_value()) {
            values.push_back(*stored.tail);
        }
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
    auto& eviction_set = lruSetLocked(item.backing_type, blockKindForItem(item));
    eviction_set.insert(EvictKey{item.last_access_seq, item.cache_key, item.backing_role});
}

void MemoryDiskBlockCache::eraseEvictKeyLocked(const CacheItem& item) {
    auto& eviction_set = lruSetLocked(item.backing_type, blockKindForItem(item));
    eviction_set.erase(EvictKey{item.last_access_seq, item.cache_key, item.backing_role});
}

void MemoryDiskBlockCache::touchLocked(CacheItem& item) {
    eraseEvictKeyLocked(item);
    item.last_access_seq = ++access_seq_;
    insertEvictKeyLocked(item);
}

std::optional<MemoryDiskBlockCache::CacheItem>
MemoryDiskBlockCache::oldestFromSetLocked(std::set<EvictKey>& eviction_set, bool single_backing_eviction) {
    for (auto evict_it = eviction_set.begin(); evict_it != eviction_set.end();) {
        const auto key     = *evict_it;
        auto       backing = findBackingLocked(key.cache_key, key.backing_role);
        if (!backing.has_value() || backing->last_access_seq != key.last_access_seq) {
            evict_it = eviction_set.erase(evict_it);
            continue;
        }
        if (backing->is_resident || backing->in_flight_ref > 0) {
            ++evict_it;
            continue;
        }
        if (backing->backing_role == CacheBackingRole::KV) {
            auto item_it = items_.find(backing->cache_key);
            if (item_it != items_.end() && item_it->second.tail.has_value() && item_it->second.tail->in_flight_ref > 0) {
                ++evict_it;
                continue;
            }
            if (single_backing_eviction && item_it != items_.end() && item_it->second.tail.has_value()) {
                ++evict_it;
                continue;
            }
        }
        return backing;
    }
    return std::nullopt;
}

std::vector<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::removeBackingLocked(CacheKeyType     cache_key,
                                                                                       CacheBackingRole role) {
    std::vector<CacheItem> removed;
    auto                   it = items_.find(cache_key);
    if (it == items_.end()) {
        return removed;
    }
    auto erase_one = [this, &removed](std::optional<CacheItem>& item) {
        if (!item.has_value()) {
            return;
        }
        eraseEvictKeyLocked(*item);
        removed.push_back(*item);
        item.reset();
    };
    if (role == CacheBackingRole::LEGACY) {
        erase_one(it->second.legacy);
    } else if (role == CacheBackingRole::KV) {
        erase_one(it->second.kv);
    } else if (role == CacheBackingRole::TAIL) {
        erase_one(it->second.tail);
    }
    if (emptyStoredItem(it->second)) {
        items_.erase(it);
    }
    return removed;
}

std::optional<MemoryDiskBlockCache::CacheItem> MemoryDiskBlockCache::findBackingLocked(CacheKeyType     cache_key,
                                                                                       CacheBackingRole role) const {
    auto it = items_.find(cache_key);
    if (it == items_.end()) {
        return std::nullopt;
    }
    auto backing = backingLocked(it->second, role);
    if (!backing.has_value()) {
        return std::nullopt;
    }
    return *backing.value();
}

std::optional<MemoryDiskBlockCache::CacheItem*>
MemoryDiskBlockCache::mutableBackingLocked(StoredItem& stored, CacheBackingRole role) {
    if (role == CacheBackingRole::LEGACY && stored.legacy.has_value()) {
        return &*stored.legacy;
    }
    if (role == CacheBackingRole::KV && stored.kv.has_value()) {
        return &*stored.kv;
    }
    if (role == CacheBackingRole::TAIL && stored.tail.has_value()) {
        return &*stored.tail;
    }
    return std::nullopt;
}

std::optional<const MemoryDiskBlockCache::CacheItem*>
MemoryDiskBlockCache::backingLocked(const StoredItem& stored, CacheBackingRole role) const {
    if (role == CacheBackingRole::LEGACY && stored.legacy.has_value()) {
        return &*stored.legacy;
    }
    if (role == CacheBackingRole::KV && stored.kv.has_value()) {
        return &*stored.kv;
    }
    if (role == CacheBackingRole::TAIL && stored.tail.has_value()) {
        return &*stored.tail;
    }
    return std::nullopt;
}

bool MemoryDiskBlockCache::backingMatchesLocked(const CacheItem& item,
                                                CacheBackingType backing_type,
                                                BlockIdxType     expected_block_index,
                                                int32_t          expected_disk_slot) const {
    if (item.backing_type != backing_type) {
        return false;
    }
    if (backing_type == CacheBackingType::MEMORY && item.block_index != expected_block_index) {
        return false;
    }
    if (backing_type == CacheBackingType::DISK && item.disk_slot != expected_disk_slot) {
        return false;
    }
    return true;
}

bool MemoryDiskBlockCache::emptyStoredItem(const StoredItem& item) {
    return !item.legacy.has_value() && !item.kv.has_value() && !item.tail.has_value();
}

CacheBlockKind MemoryDiskBlockCache::blockKindForItem(const CacheItem& item) {
    if (item.backing_role == CacheBackingRole::TAIL) {
        return CacheBlockKind::COMPLETE;
    }
    if (item.backing_role == CacheBackingRole::KV) {
        return CacheBlockKind::INCOMPLETE;
    }
    return blockKindFromComplete(item.is_complete);
}

CacheBackingRole MemoryDiskBlockCache::normalizeRole(const CacheItem& item) {
    if (isSplitRole(item.backing_role)) {
        return item.backing_role;
    }
    return CacheBackingRole::LEGACY;
}

std::set<MemoryDiskBlockCache::EvictKey>& MemoryDiskBlockCache::lruSetLocked(CacheBackingType backing_type,
                                                                             CacheBlockKind   kind) {
    if (backing_type == CacheBackingType::MEMORY) {
        return kind == CacheBlockKind::COMPLETE ? memory_complete_lru_ : memory_incomplete_lru_;
    }
    return kind == CacheBlockKind::COMPLETE ? disk_complete_lru_ : disk_incomplete_lru_;
}

}  // namespace rtp_llm
