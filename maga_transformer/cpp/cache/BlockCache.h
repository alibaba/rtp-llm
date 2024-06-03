#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "maga_transformer/cpp/utils/LRUCache.h"

namespace rtp_llm {

struct CacheItem {
    std::vector<int> token_list;
    std::vector<int> block_indices;
    size_t           cache_key;
    bool             is_resident = false;
};

const size_t kCacheMaxCapacity = 1000000;

class BlockCache {
public:
    BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    static size_t prefixLength(const std::vector<int>& left, const std::vector<int>& right);

    std::pair<std::vector<int>, size_t> match(const std::vector<int>& token_list);

    std::vector<int> put(const std::vector<int>& token_list, const std::vector<int>& block_indices, bool is_resident);

    std::vector<int> pop();

    bool empty() const;

    size_t size() const;

    bool hasKey(const std::vector<int>& token_list) const;

    bool isResident(const std::vector<int>& token_list) const;

private:
    bool hasHashKey(size_t cache_key) const;

private:
    mutable LRUCache<size_t, CacheItem> lru_cache_;
};

}  // namespace rtp_llm
