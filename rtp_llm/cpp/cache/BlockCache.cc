#include "rtp_llm/cpp/cache/BlockCache.h"

#include <algorithm>
#include <cstring>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

void BlockCache::registerParentIndex(const CacheItem& item, const CacheKeyGroupPair& lru_key) {
    if (item.valid_token_len <= 0) {
        return;
    }
    ParentGroupKey p{item.parent_block_key, item.group_id};
    auto&          vec = parent_bucket_[p];
    if (std::find(vec.begin(), vec.end(), lru_key) == vec.end()) {
        vec.push_back(lru_key);
    }
    lru_key_to_parent_[lru_key] = p;
}

void BlockCache::onLruEntryRemoved(const CacheKeyGroupPair& lru_key, const CacheItem& item) {
    (void)item;
    auto it = lru_key_to_parent_.find(lru_key);
    if (it == lru_key_to_parent_.end()) {
        return;
    }
    ParentGroupKey p = it->second;
    lru_key_to_parent_.erase(it);
    auto bit = parent_bucket_.find(p);
    if (bit == parent_bucket_.end()) {
        return;
    }
    auto& vec = bit->second;
    vec.erase(std::remove(vec.begin(), vec.end(), lru_key), vec.end());
    if (vec.empty()) {
        parent_bucket_.erase(bit);
    }
}

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

BlockCache::PartialTailMatchResult BlockCache::matchPartialTailByParent(CacheKeyType   parent_block_key,
                                                                        GroupIdType    group_id,
                                                                        int            L,
                                                                        int            seq_size_per_block,
                                                                        bool           is_linear_attention,
                                                                        const int32_t* req_tokens,
                                                                        int            req_token_off) {
    RTP_LLM_PROFILE_FUNCTION();
    PartialTailMatchResult out;
    if (L <= 0 || seq_size_per_block <= 0 || req_tokens == nullptr) {
        return out;
    }

    std::lock_guard<std::mutex> lock(mu_);
    ParentGroupKey              pkey{parent_block_key, group_id};
    auto                        bit = parent_bucket_.find(pkey);
    if (bit == parent_bucket_.end()) {
        return out;
    }

    size_t            best_reuse = 0;
    int               best_v     = -1;
    CacheKeyGroupPair best_lru{};
    bool              found = false;

    for (const auto& cand_lru : bit->second) {
        CacheItem it;
        if (!lru_cache_.peek(cand_lru, &it)) {
            continue;
        }
        if (it.valid_token_len <= 0) {
            continue;
        }
        const int v = it.valid_token_len;
        if (static_cast<int>(it.prefix_tokens.size()) < v) {
            continue;
        }
        const int cmp_n = std::min(v, L);
        if (cmp_n > 0
            && std::memcmp(
                   req_tokens + req_token_off, it.prefix_tokens.data(), sizeof(int32_t) * static_cast<size_t>(cmp_n))
                   != 0) {
            continue;
        }

        size_t reuse = 0;
        if (is_linear_attention) {
            if (L < v) {
                continue;
            }
            reuse = static_cast<size_t>(v);
        } else {
            if (v <= L) {
                reuse = static_cast<size_t>(v);
            } else {
                reuse = static_cast<size_t>(L);
            }
        }

        const bool better = (!found) || (reuse > best_reuse) || (reuse == best_reuse && v > best_v);
        if (better) {
            found      = true;
            best_reuse = reuse;
            best_v     = v;
            best_lru   = cand_lru;
        }
    }

    if (!found) {
        return out;
    }

    auto [touch_ok, touched] = lru_cache_.get(best_lru);
    (void)touched;
    if (!touch_ok) {
        return out;
    }
    out.matched_index = touched.block_index;
    out.reuse_tokens  = best_reuse;
    return out;
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

    CacheKeyGroupPair lru_k{item.cache_key, item.group_id};

    if (lru_cache_.contains(lru_k)) {
        lru_cache_.get(lru_k);
        return false;
    }

    if (lru_cache_.full()) {
        auto [popped_ok, evicted] = lru_cache_.pop();
        if (popped_ok) {
            CacheKeyGroupPair evict_k{evicted.cache_key, evicted.group_id};
            onLruEntryRemoved(evict_k, evicted);
        }
    }

    lru_cache_.put(lru_k, item);
    if (item.valid_token_len > 0) {
        registerParentIndex(item, lru_k);
    }
    return true;
}

BlockIndicesType BlockCache::pop(int n) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(n > 0, "pop n should > 0, n = " + std::to_string(n));
    BlockIndicesType pop_blocks;

    auto cond = [&](const CacheKeyGroupPair& /*key*/, const CacheItem& item) { return !item.is_resident; };

    while (n > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success) {
            break;
        }
        CacheKeyGroupPair k{item.cache_key, item.group_id};
        onLruEntryRemoved(k, item);
        pop_blocks.push_back(item.block_index);
        n--;
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
    onLruEntryRemoved(key, removed_item);
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
                onLruEntryRemoved(key, removed_item);
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
