#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

struct CacheItem {
    std::vector<int32_t>    block_indices;
    std::vector<int64_t>    cache_keys;
    std::vector<float>      loss;
    bool                    is_resident = false;
};

const size_t kCacheMaxCapacity = 10000000;

class BlockCache {
public:
    explicit BlockCache(size_t seq_size_per_block):
        lru_cache_(kCacheMaxCapacity), seq_size_per_block_(seq_size_per_block) {}

    ~BlockCache();

    struct MatchResult {
        std::vector<int>    matched_indices;
        std::vector<float>  loss;
    };

    struct CacheBlock {
        int32_t block_index;
        float   loss;
        bool    is_resident = false;
    };

    MatchResult match(const std::vector<int64_t>& cache_keys);
    
    std::vector<int> put(const std::vector<int64_t> cache_keys, const std::vector<int> block_indices);

    std::vector<int> pop();

    bool empty() const;

    size_t size() const;

    // TODO: CacheSnapShoot 

private:
    std::vector<std::vector<int>> cache_keys_vector_;
    std::vector<int> block_indices_;

    size_t                                  seq_size_per_block_;
    mutable std::mutex                      mutex_;
    mutable LRUCache<int64_t, CacheBlock>   lru_cache_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
