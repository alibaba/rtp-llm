#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <sstream>

namespace rtp_llm {

namespace {

bool kvCacheDebugLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("KV_CACHE_DEBUG_LOG");
        if (value == nullptr) {
            return false;
        }
        return strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0;
    }();
    return enabled;
}

template<typename T>
std::string previewVector(const std::vector<T>& values, size_t limit = 8) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size() && i < limit; ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << values[i];
    }
    if (values.size() > limit) {
        oss << ",...";
    }
    oss << "]";
    return oss.str();
}

}  // namespace

void SharedBlockCache::init(int group_num, const std::vector<BlockPoolPtr>& group_pools) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int>(group_pools.size()) == group_num,
                            "group_pools size %zu != group_num %d",
                            group_pools.size(),
                            group_num);
    group_num_   = group_num;
    group_pools_ = group_pools;
}

void SharedBlockCache::put(CacheKeyType cache_key, const std::vector<BlockIdxType>& group_slots, bool is_resident) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    if (lru_cache_.contains(cache_key)) {
        auto [success, existing_item] = lru_cache_.get(cache_key);
        if (success) {
            bool updated = false;
            for (size_t gid = 0; gid < group_slots.size(); ++gid) {
                if (isNullBlockIdx(group_slots[gid])) {
                    continue;
                }
                if (gid >= existing_item.slots.size()) {
                    existing_item.slots.resize(gid + 1, NULL_BLOCK_IDX);
                }
                if (isNullBlockIdx(existing_item.slots[gid])) {
                    existing_item.slots[gid] = group_slots[gid];
                    updated                  = true;
                    if (static_cast<int>(gid) < group_num_) {
                        group_pools_[gid]->blockCacheReference(group_slots[gid]);
                    }
                }
            }
            if (updated) {
                lru_cache_.put(cache_key, existing_item);
                ++version_;
                if (kvCacheDebugLogEnabled()) {
                    RTP_LLM_LOG_INFO("device shared-cache put merge: key=%ld slots=%s version=%ld size=%zu",
                                     cache_key,
                                     previewVector(existing_item.slots).c_str(),
                                     version_,
                                     lru_cache_.size());
                }
            }
        }
        return;
    }

    UnifiedCacheItem item;
    item.cache_key   = cache_key;
    item.is_resident = is_resident;
    item.slots       = group_slots;

    lru_cache_.put(cache_key, item);
    ++version_;
    if (kvCacheDebugLogEnabled()) {
        RTP_LLM_LOG_INFO("device shared-cache put new: key=%ld slots=%s resident=%d version=%ld size=%zu",
                         cache_key,
                         previewVector(group_slots).c_str(),
                         is_resident,
                         version_,
                         lru_cache_.size());
    }

    for (int gid = 0; gid < static_cast<int>(group_slots.size()) && gid < group_num_; ++gid) {
        if (!isNullBlockIdx(group_slots[gid])) {
            group_pools_[gid]->blockCacheReference(group_slots[gid]);
        }
    }
}

SharedBlockCache::MatchResult SharedBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        if (kvCacheDebugLogEnabled()) {
            RTP_LLM_LOG_INFO("device shared-cache match miss: key=%ld size=%zu", cache_key, lru_cache_.size());
        }
        return {false, {}};
    }
    if (kvCacheDebugLogEnabled()) {
        RTP_LLM_LOG_INFO("device shared-cache match hit: key=%ld slots=%s size=%zu",
                         cache_key,
                         previewVector(item.slots).c_str(),
                         lru_cache_.size());
    }
    return {true, item.slots};
}

BlockIdxType SharedBlockCache::matchGroup(CacheKeyType cache_key, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        if (kvCacheDebugLogEnabled()) {
            RTP_LLM_LOG_INFO("device shared-cache matchGroup miss: key=%ld group=%d size=%zu",
                             cache_key,
                             group_id,
                             lru_cache_.size());
        }
        return NULL_BLOCK_IDX;
    }
    if (group_id < 0 || static_cast<size_t>(group_id) >= item.slots.size()) {
        return NULL_BLOCK_IDX;
    }
    const auto block = item.slots[group_id];
    if (kvCacheDebugLogEnabled()) {
        RTP_LLM_LOG_INFO("device shared-cache matchGroup %s: key=%ld group=%d block=%d slots=%s size=%zu",
                         isNullBlockIdx(block) ? "null" : "hit",
                         cache_key,
                         group_id,
                         block,
                         previewVector(item.slots).c_str(),
                         lru_cache_.size());
    }
    return block;
}

SharedBlockCache::EvictResult SharedBlockCache::selectAndEvict(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    EvictResult result;
    if (lru_cache_.empty() || min_blocks == 0) {
        return result;
    }

    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }

    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident || resident_keys.count(item.cache_key)) {
            continue;
        }
        lru_keys.push_back(item.cache_key);
    }

    size_t selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        UnifiedCacheItem removed_item;
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }

        result.evicted_keys.push_back(cache_key);
        result.evicted_slots[cache_key] = removed_item.slots;

        for (const auto& slot : removed_item.slots) {
            if (!isNullBlockIdx(slot)) {
                selected_blocks++;
            }
        }
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    if (kvCacheDebugLogEnabled()) {
        std::vector<CacheKeyType> preview_keys;
        preview_keys.reserve(result.evicted_keys.size());
        size_t selected_blocks = 0;
        for (const auto key : result.evicted_keys) {
            preview_keys.push_back(key);
            const auto& slots = result.evicted_slots[key];
            for (const auto slot : slots) {
                if (!isNullBlockIdx(slot)) {
                    ++selected_blocks;
                }
            }
        }
        RTP_LLM_LOG_INFO("device shared-cache select evict: min_blocks=%zu evicted_keys=%zu evicted_blocks=%zu "
                         "key_preview=%s remain_size=%zu",
                         min_blocks,
                         result.evicted_keys.size(),
                         selected_blocks,
                         previewVector(preview_keys).c_str(),
                         lru_cache_.size());
    }
    return result;
}

size_t SharedBlockCache::evictAndFree(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();

    auto evict_result = selectAndEvict(min_blocks);
    if (evict_result.evicted_keys.empty()) {
        return 0;
    }

    size_t freed = 0;
    for (size_t i = 0; i < evict_result.evicted_keys.size(); ++i) {
        const auto  cache_key = evict_result.evicted_keys[i];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);

        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < group_num_; ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                group_pools_[gid]->blockCacheFree(slots[gid]);
                freed++;
            }
        }
    }
    if (kvCacheDebugLogEnabled()) {
        RTP_LLM_LOG_INFO("device shared-cache evict free: min_blocks=%zu evicted_keys=%zu freed_blocks=%zu",
                         min_blocks,
                         evict_result.evicted_keys.size(),
                         freed);
    }
    return freed;
}

std::optional<SharedBlockCache::UnifiedCacheItem> SharedBlockCache::remove(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mu_);

    UnifiedCacheItem removed_item;
    if (!lru_cache_.remove(cache_key, &removed_item)) {
        return std::nullopt;
    }
    return removed_item;
}

bool SharedBlockCache::contains(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.contains(cache_key);
}

bool SharedBlockCache::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.empty();
}

size_t SharedBlockCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.size();
}

std::vector<CacheKeyType> SharedBlockCache::allCacheKeys() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<CacheKeyType>   keys;
    keys.reserve(lru_cache_.size());
    for (const auto& [key, item] : lru_cache_.items()) {
        keys.push_back(key);
    }
    return keys;
}

int64_t SharedBlockCache::version() const {
    std::lock_guard<std::mutex> lock(mu_);
    return version_;
}

}  // namespace rtp_llm
