#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockCache::MatchResult BlockCache::match(CacheKeyType cache_key, int group_id, int64_t current_batch_epoch) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    auto [success, item] = lru_cache_.get(key);
    if (success) {
        // Matching logic for Epoch-based cache isolation:
        // 1. epoch == 0: Legacy blocks (backward compatible), globally visible (completed and committed batches)
        // 2. epoch == current_batch_epoch: Blocks from current batch, visible (allows cross-step visibility within same
        // batch)
        // 3. epoch != current_batch_epoch && epoch > 0: Blocks from other incomplete batches, invisible (prevents dirty
        // data)
        if (item.epoch == 0 || (current_batch_epoch > 0 && item.epoch == current_batch_epoch)) {
            return {item.block_index};
        }
    }
    return {NULL_BLOCK_IDX};
}

bool BlockCache::contains(CacheKeyType cache_key, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    return lru_cache_.contains(key);
}

BlockCache::PutResult BlockCache::put(CacheItem& item) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    CacheKeyGroupPair key{item.cache_key, item.group_id};
    auto [found, old_item] = lru_cache_.get(key);
    if (found) {
        if (old_item.epoch == 0 && item.epoch != 0) {
            return {PutResult::Action::SKIPPED, NULL_BLOCK_IDX};
        }

        // Replace old item
        BlockIdxType old_block_index = old_item.block_index;
        lru_cache_.put(key, item);
        return {PutResult::Action::REPLACED, old_block_index};
    } else {
        // Insert new item
        lru_cache_.put(key, item);
        return {PutResult::Action::INSERTED, NULL_BLOCK_IDX};
    }
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

void BlockCache::debugString() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t                      cache_size = lru_cache_.size();
    RTP_LLM_LOG_INFO("BlockCache state: total cached items = %zu", cache_size);
    if (cache_size > 0) {
        auto snapshot = lru_cache_.cacheSnapshot(-1);
        RTP_LLM_LOG_INFO(
            "BlockCache snapshot: version = %ld, items count = %zu", snapshot.version, snapshot.values.size());
        size_t item_count = 0;
        for (const auto& item : snapshot.values) {
            RTP_LLM_LOG_INFO("BlockCache item[%zu]: %s", item_count++, item.debugString().c_str());
            // Limit output to avoid log flooding
            if (item_count >= 50) {
                RTP_LLM_LOG_INFO("BlockCache: ... (showing first 50 items, total %zu items)", snapshot.values.size());
                break;
            }
        }
    }
}

}  // namespace rtp_llm
