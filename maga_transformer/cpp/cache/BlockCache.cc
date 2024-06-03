#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "maga_transformer/cpp/cache/BlockCache.h"
#include "maga_transformer/cpp/utils/LRUCache.h"
#include "maga_transformer/cpp/utils/StringUtil.h"

namespace rtp_llm {

std::size_t hashVector(const std::vector<int>& vec) {
    std::hash<std::string> hasher;
    std::string            vecString = vectorToString(vec);
    return hasher(vecString);
}

size_t BlockCache::prefixLength(const std::vector<int>& left, const std::vector<int>& right) {
    size_t max_common_length = std::min(left.size(), right.size());
    for (size_t index = 0; index < max_common_length; ++index) {
        if (left[index] != right[index]) {
            return index;
        }
    }
    return max_common_length;
}

std::pair<std::vector<int>, size_t> BlockCache::match(const std::vector<int>& token_list) {
    CacheItem matched_item;
    size_t    matched_len = 0;

    for (const auto& item : lru_cache_.items()) {
        size_t common_length = prefixLength(item.second.token_list, token_list);
        if (common_length > matched_len) {
            matched_item = item.second;
            matched_len  = common_length;
        }
    }

    // Increase matched item's popularity
    if (matched_len > 0) {
        lru_cache_.get(matched_item.cache_key);
    }

    return {matched_item.block_indices, matched_len};
}

std::vector<int>
BlockCache::put(const std::vector<int>& token_list, const std::vector<int>& block_indices, bool is_resident) {
    if (token_list.empty() || block_indices.empty()) {
        return {};
    }
    
    size_t    cache_key = hashVector(token_list);
    CacheItem item{token_list, block_indices, cache_key, is_resident};

    if (lru_cache_.contains(cache_key)) {
        return block_indices;
    }

    lru_cache_.put(cache_key, item);  // Assuming LRUCache has a put() method
    return {};
}

std::vector<int> BlockCache::pop() {
    CacheItem              return_cache_item;
    std::vector<CacheItem> resident_list;

    while (!empty()) {
        auto [success, cache_item] = lru_cache_.pop();
        assert(success);
        if (cache_item.is_resident) {
            resident_list.push_back(cache_item);
        } else {
            return_cache_item = cache_item;
            break;
        }
    }

    for (const auto& resident_cache_item : resident_list) {
        lru_cache_.put(resident_cache_item.cache_key, resident_cache_item);
    }

    return return_cache_item.block_indices;
}

bool BlockCache::empty() const {
    return lru_cache_.empty();
}

size_t BlockCache::size() const {
    return lru_cache_.size();
}

bool BlockCache::hasKey(const std::vector<int>& token_list) const {
    size_t cache_key = hashVector(token_list);
    return hasHashKey(cache_key);
}

bool BlockCache::hasHashKey(size_t cache_key) const {
    return lru_cache_.contains(cache_key);
}

bool BlockCache::isResident(const std::vector<int>& token_list) const {
    size_t cache_key = hashVector(token_list);
    if (!hasHashKey(cache_key)) {
        return false;
    }
    const auto& [success, item] = lru_cache_.get(cache_key);
    assert(success);
    return item.is_resident;
}

}  // namespace rtp_llm
