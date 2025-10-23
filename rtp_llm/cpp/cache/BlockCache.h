#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct CacheItem {
    std::vector<int32_t> token_list;
    std::vector<int32_t> block_indices;
    std::vector<int64_t> cache_key;
    std::vector<float>   loss;
    bool                 is_resident = false;
    size_t               item_key;
    std::string          debugString() const {
        std::stringstream debug_string;
        debug_string << "FreeInfo item_key: " << item_key << ", token_ids: ";
        debug_string << " cache_keys: ";
        for (auto& v : cache_key) {
            debug_string << v << ", ";
        }
        debug_string << " block_indices: ";
        for (auto& v : block_indices) {
            debug_string << v << ", ";
        }

        return debug_string.str();
    }
};

const size_t kCacheMaxCapacity = 1000000;

class BlockCache {
public:
    struct MatchResult {
        std::vector<int>   block_indices;
        std::vector<float> loss;
    };
    using CacheSnapshot = typename LRUCache<size_t, CacheItem>::CacheSnapshot;

public:
    explicit BlockCache(size_t seq_size_per_block):
        lru_cache_(kCacheMaxCapacity), seq_size_per_block_(seq_size_per_block) {}

    static size_t prefixLength(const std::vector<int64_t>& left, const std::vector<int64_t>& right);

    MatchResult match(const std::vector<int64_t>& cache_key);

    std::vector<int> put(CacheItem& item);

    std::vector<int> pop();

    bool empty() const;

    size_t size() const;

    bool hasKey(const std::vector<int>& token_list) const;

    bool isResident(const std::vector<int>& token_list) const;

    int holdBlockNums() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    bool hasHashKey(size_t item_key) const;

private:
    mutable LRUCache<size_t, CacheItem>  lru_cache_;
    mutable std::mutex                   mutex_;
    mutable std::unordered_map<int, int> hold_blocks_;
    mutable int                          total_hold_blocks_ = 0;
    size_t                               seq_size_per_block_;
};

}  // namespace rtp_llm
