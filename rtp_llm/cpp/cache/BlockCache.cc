#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

void BlockCache::registerModel(size_t model_id, size_t group_num, BlockPoolPtr block_pool) {
    std::lock_guard<std::mutex> lock(mu_);
    if (model_id >= registry_.size()) {
        registry_.resize(model_id + 1);
    }
    registry_[model_id] = {model_id, group_num, std::move(block_pool)};
}

size_t BlockCache::registeredModelNum() const {
    std::lock_guard<std::mutex> lock(mu_);
    return registry_.size();
}

void BlockCache::ensureSlots(CacheItem& item, size_t model_id, int group_id) const {
    if (model_id >= item.slots.size()) {
        item.slots.resize(model_id + 1);
    }
    if (group_id >= static_cast<int>(item.slots[model_id].size())) {
        item.slots[model_id].resize(static_cast<size_t>(group_id) + 1);
    }
}

BlockCache::MatchResult BlockCache::matchSlot(CacheKeyType cache_key, size_t model_id, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return {NULL_BLOCK_IDX};
    }
    if (model_id >= item.slots.size() || group_id >= static_cast<int>(item.slots[model_id].size())) {
        return {NULL_BLOCK_IDX};
    }
    const auto& slot = item.slots[model_id][group_id];
    return {slot.valid() ? slot.block_id : NULL_BLOCK_IDX};
}

bool BlockCache::putSlot(
    CacheKeyType cache_key, size_t model_id, int group_id, BlockIdxType block_id, bool is_resident) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(block_id), "putSlot block_id should not be null block");

    if (lru_cache_.contains(cache_key)) {
        auto [ok, item] = lru_cache_.get(cache_key);
        if (ok) {
            ensureSlots(item, model_id, group_id);
            auto& slot = item.slots[model_id][group_id];
            if (!slot.valid()) {
                slot.block_id = block_id;
                // Update item in LRU (get already moved it to front; re-put to update value)
                lru_cache_.put(cache_key, item);
                return true;  // slot newly filled
            }
        }
        return false;  // already existed
    }

    CacheItem item;
    item.cache_key   = cache_key;
    item.is_resident = is_resident;
    ensureSlots(item, model_id, group_id);
    item.slots[model_id][group_id].block_id = block_id;
    lru_cache_.put(cache_key, item);
    return true;  // newly created
}

bool BlockCache::upsertCacheItem(CacheKeyType                               cache_key,
                                 const std::vector<std::vector<CacheSlot>>& full_slots,
                                 bool                                       is_resident) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    // Safety: popAndFree calls blockCacheFree while holding mu_.
    // BlockPool::blockCacheFree only acquires BlockPool-internal mutexes (free_mu_, ref_mu_)
    // and never calls back into BlockCache. So there is no lock-order inversion here.

    auto* existing = lru_cache_.peekMutable(cache_key);
    if (existing == nullptr) {
        // New entry: create CacheItem and call blockCacheReference for all valid slots.
        CacheItem item;
        item.cache_key   = cache_key;
        item.is_resident = is_resident;
        item.slots       = full_slots;

        for (size_t mid = 0; mid < full_slots.size(); ++mid) {
            for (const auto& slot : full_slots[mid]) {
                if (slot.valid() && mid < registry_.size() && registry_[mid].block_pool) {
                    registry_[mid].block_pool->blockCacheReference(slot.block_id);
                }
            }
        }
        lru_cache_.put(cache_key, item);
        return true;
    }

    // Existing entry: diff-update slots and manage ref counts.
    CacheItem& item = *existing;

    // Ensure slots matrix is large enough for the new data.
    if (item.slots.size() < full_slots.size()) {
        item.slots.resize(full_slots.size());
    }

    bool any_change = false;
    for (size_t mid = 0; mid < full_slots.size(); ++mid) {
        const auto& new_model_slots = full_slots[mid];
        auto&       old_model_slots = item.slots[mid];
        if (old_model_slots.size() < new_model_slots.size()) {
            old_model_slots.resize(new_model_slots.size());
        }
        BlockPoolPtr pool = (mid < registry_.size()) ? registry_[mid].block_pool : nullptr;

        for (size_t gid = 0; gid < new_model_slots.size(); ++gid) {
            const auto& new_slot = new_model_slots[gid];
            auto&       old_slot = old_model_slots[gid];

            if (!old_slot.valid() && new_slot.valid()) {
                // Newly filled slot: acquire cache reference.
                if (pool) {
                    pool->blockCacheReference(new_slot.block_id);
                }
                old_slot = new_slot;
                any_change = true;
            } else if (old_slot.valid() && new_slot.valid() && old_slot.block_id != new_slot.block_id) {
                // Replaced slot: release old, acquire new.
                if (pool) {
                    pool->blockCacheFree(old_slot.block_id);
                    pool->blockCacheReference(new_slot.block_id);
                }
                old_slot = new_slot;
                any_change = true;
            } else if (old_slot.valid() && !new_slot.valid()) {
                // Slot explicitly cleared: release the cache hold on the old block.
                if (pool) {
                    pool->blockCacheFree(old_slot.block_id);
                }
                old_slot = new_slot;
                any_change = true;
            }
            // old invalid && new invalid: nothing to do.
        }
    }

    // Always touch to refresh LRU position (re-access counts as a use even if no slot changed).
    lru_cache_.put(cache_key, item);
    return any_change;
}

bool BlockCache::containsSlot(CacheKeyType cache_key, size_t model_id, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    // Use O(1) peek to avoid updating LRU order.
    const auto* item = lru_cache_.peek(cache_key);
    if (item == nullptr) {
        return false;
    }
    if (model_id >= item->slots.size() || group_id >= static_cast<int>(item->slots[model_id].size())) {
        return false;
    }
    return item->slots[model_id][group_id].valid();
}

std::optional<BlockCache::CacheItem> BlockCache::removeItem(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheItem                   removed_item;
    if (!lru_cache_.remove(cache_key, &removed_item)) {
        return std::nullopt;
    }
    return removed_item;
}

BlockIndicesType BlockCache::pop(int nums) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    BlockIndicesType pop_blocks;

    auto cond = [&](const CacheKeyType& /*key*/, const CacheItem& item) { return !item.is_resident; };

    while (nums > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success)
            break;
        // Collect all valid block indices from the popped item
        for (const auto& model_slots : item.slots) {
            for (const auto& slot : model_slots) {
                if (slot.valid()) {
                    pop_blocks.push_back(slot.block_id);
                }
            }
        }
        nums--;
    }

    return pop_blocks;
}

size_t BlockCache::popAndFree(size_t required_blocks, size_t trigger_model_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    if (required_blocks == 0 || lru_cache_.empty()) {
        return 0;
    }

    auto cond = [&](const CacheKeyType& /*key*/, const CacheItem& item) { return !item.is_resident; };

    // Holding mu_ while calling BlockPool::blockCacheFree is safe:
    // BlockPool::blockCacheFree only acquires BlockPool-internal mutexes (free_mu_, ref_mu_)
    // and never calls back into BlockCache, so there is no lock-order inversion.

    size_t freed_count = 0;
    while (freed_count < required_blocks && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success)
            break;

        // Free blocks through registered BlockPools — each block back to its own pool.
        for (size_t mid = 0; mid < item.slots.size(); ++mid) {
            BlockIndicesType blocks_to_free;
            for (const auto& slot : item.slots[mid]) {
                if (slot.valid()) {
                    blocks_to_free.push_back(slot.block_id);
                }
            }
            if (!blocks_to_free.empty() && mid < registry_.size() && registry_[mid].block_pool) {
                registry_[mid].block_pool->blockCacheFree(blocks_to_free);
            }
        }

        // Count freed blocks for the trigger model only — caller uses this to decide
        // whether enough space has been reclaimed in its own pool.
        freed_count += item.validSlotsForModel(trigger_model_id);
    }

    return freed_count;
}

void BlockCache::freeItemBlocks(const CacheItem& item) {
    for (size_t mid = 0; mid < item.slots.size(); ++mid) {
        BlockIndicesType blocks_to_free;
        for (const auto& slot : item.slots[mid]) {
            if (slot.valid()) {
                blocks_to_free.push_back(slot.block_id);
            }
        }
        if (!blocks_to_free.empty() && mid < registry_.size() && registry_[mid].block_pool) {
            registry_[mid].block_pool->blockCacheFree(blocks_to_free);
        }
    }
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

    // Collect resident keys
    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(key);
        }
    }

    // Walk LRU order (back = least-recently-used), collect non-resident keys
    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident)
            continue;
        if (resident_keys.count(item.cache_key))
            continue;
        // Avoid duplicates (each cache_key is unique in new model, but be safe)
        if (std::find(lru_keys.begin(), lru_keys.end(), item.cache_key) == lru_keys.end()) {
            lru_keys.push_back(item.cache_key);
        }
    }

    // Select keys until we have enough blocks (O(1) peek per key via cache_items_map_).
    std::vector<CacheKeyType> selected_keys;
    size_t                    selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        const auto* item = lru_cache_.peek(cache_key);
        if (item != nullptr) {
            selected_keys.push_back(cache_key);
            selected_blocks += item->totalValidSlots();
            if (selected_blocks >= min_blocks) {
                break;
            }
        }
    }

    if (selected_keys.empty()) {
        return result;
    }

    // Remove selected items
    for (const auto cache_key : selected_keys) {
        CacheItem removed_item;
        if (lru_cache_.remove(cache_key, &removed_item)) {
            result.evicted_keys.push_back(cache_key);
            result.evicted_items[cache_key] = std::move(removed_item);
        }
    }

    return result;
}

}  // namespace rtp_llm
