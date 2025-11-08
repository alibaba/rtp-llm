#pragma once

#include <vector>
#include <memory>

namespace rtp_llm {

class CacheManager;

class KVCacheResource {
public:
    KVCacheResource(const std::vector<int>& block_id): block_id(block_id) {}
    void            clear();
    KVCacheResource clone(std::shared_ptr<CacheManager>& cache_manager) const;

public:
    // [max_block_per_seq]
    std::vector<int> block_id;
};

}  // namespace rtp_llm