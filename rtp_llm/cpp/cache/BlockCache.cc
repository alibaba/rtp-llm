#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

std::size_t hashVector(const std::vector<int>& vec) {
    std::hash<std::string> hasher;
    std::string            vecString = vectorToString(vec);
    return hasher(vecString);
}

size_t BlockCache::prefixLength(const std::vector<int64_t>& left, const std::vector<int64_t>& right) {
    size_t max_common_length = std::min(left.size(), right.size());
    for (size_t index = 0; index < max_common_length; ++index) {
        if (left[index] != right[index]) {
            return index;
        }
    }
    return max_common_length;
}

BlockCache::MatchResult BlockCache::match(const std::vector<int64_t>& cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    CacheItem matched_item;
    size_t    matched_blocks = 0;

    for (const auto& item : lru_cache_.items()) {
        size_t common_length = prefixLength(item.second.cache_key, cache_key);
        if (common_length > matched_blocks) {
            matched_item   = item.second;
            matched_blocks = common_length;
        }
    }

    // Increase matched item's popularity
    if (matched_blocks > 0) {
        lru_cache_.get(matched_item.item_key);
    }

    auto matched_block_indices =
        std::vector<int>(matched_item.block_indices.begin(), matched_item.block_indices.begin() + matched_blocks);
    std::vector<float> matched_loss;
    if (!matched_item.loss.empty() && matched_blocks * seq_size_per_block_ <= matched_item.loss.size()) {
        matched_loss = std::vector<float>(matched_item.loss.begin(),
                                          matched_item.loss.begin() + matched_blocks * seq_size_per_block_);
    }
    return {matched_block_indices, matched_loss};
}

std::vector<int> BlockCache::put(CacheItem& item) {
    if (item.token_list.empty() || item.block_indices.empty()) {
        return {};
    }

    item.item_key = hashVector(item.token_list);

    std::lock_guard<std::mutex> lock(mutex_);

    if (lru_cache_.contains(item.item_key)) {
        // Increase matched item's popularity
        lru_cache_.get(item.item_key);
        return item.block_indices;
    }

    lru_cache_.put(item.item_key, item);

    for (auto block : item.block_indices) {
        auto result = ++hold_blocks_[block];
        // is new block
        if (result == 1) {
            total_hold_blocks_++;
        }
    }

    return {};
}

std::vector<int> BlockCache::pop() {
    CacheItem              return_cache_item;
    std::vector<CacheItem> resident_list;

    std::lock_guard<std::mutex> lock(mutex_);

    while (!lru_cache_.empty()) {
        auto [success, cache_item] = lru_cache_.pop();
        RTP_LLM_CHECK(success);
        if (cache_item.is_resident) {
            resident_list.push_back(cache_item);
        } else {
            return_cache_item = cache_item;
            break;
        }
    }

    for (const auto& resident_cache_item : resident_list) {
        lru_cache_.put(resident_cache_item.item_key, resident_cache_item);
    }

    for (auto block : return_cache_item.block_indices) {
        auto result = --hold_blocks_[block];
        // is last reference
        if (result == 0) {
            total_hold_blocks_--;
        }
    }

    return return_cache_item.block_indices;
}

bool BlockCache::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.empty();
}

size_t BlockCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.size();
}

bool BlockCache::hasKey(const std::vector<int>& token_list) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t item_key = hashVector(token_list);
    return hasHashKey(item_key);
}

bool BlockCache::hasHashKey(size_t item_key) const {
    return lru_cache_.contains(item_key);
}

bool BlockCache::isResident(const std::vector<int>& token_list) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t item_key = hashVector(token_list);
    if (!hasHashKey(item_key)) {
        return false;
    }
    const auto& [success, item] = lru_cache_.get(item_key);
    RTP_LLM_CHECK(success);
    return item.is_resident;
}

int BlockCache::holdBlockNums() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_hold_blocks_;
}

BlockCache::CacheSnapshot BlockCache::cacheSnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.cacheSnapshot(latest_version);
}

std::tuple<int64_t, std::vector<int64_t>> BlockCache::getVersionAndCacheKeys(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t                     current_version = lru_cache_.getVersion();
    std::vector<int64_t>        cachekeys;

    if (latest_version < current_version) {
        std::unordered_set<int64_t> seen_keys;
        lru_cache_.forEachValue([&](const CacheItem& item) {
            for (const auto& key_part : item.cache_key) {
                if (seen_keys.insert(key_part).second) {
                    cachekeys.push_back(key_part);
                }
            }
        });
    }

    return {current_version, std::move(cachekeys)};
}

}  // namespace rtp_llm
