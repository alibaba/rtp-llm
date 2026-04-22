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

bool BlockCache::containsSlot(CacheKeyType cache_key, size_t model_id, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!lru_cache_.contains(cache_key)) {
        return false;
    }
    // LRUCache::contains is const, but we need to peek at the value.
    // Walk the items list to find the entry without mutating LRU order.
    for (const auto& [key, item] : lru_cache_.items()) {
        if (key == cache_key) {
            if (model_id >= item.slots.size() || group_id >= static_cast<int>(item.slots[model_id].size())) {
                return false;
            }
            return item.slots[model_id][group_id].valid();
        }
    }
    return false;
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

    size_t freed_count = 0;
    while (freed_count < required_blocks && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success)
            break;

        // Free blocks through registered BlockPools
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

        // Count freed blocks for the trigger model
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

    // Select keys until we have enough blocks
    std::vector<CacheKeyType> selected_keys;
    size_t                    selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        // Peek at the item to count its blocks
        bool found = false;
        for (const auto& [key, item] : lru_cache_.items()) {
            if (key == cache_key) {
                selected_keys.push_back(cache_key);
                selected_blocks += item.totalValidSlots();
                found = true;
                break;
            }
        }
        if (found && selected_blocks >= min_blocks) {
            break;
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
