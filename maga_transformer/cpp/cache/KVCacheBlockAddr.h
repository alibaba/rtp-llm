#pragma once

#include <vector>
#include <memory>

namespace rtp_llm {

class CacheManager;

class KVCacheBlockAddr {
public:
    void clear();
    KVCacheBlockAddr clone(std::shared_ptr<CacheManager>& cache_manager);

public:
    // [max_block_per_seq]
    std::vector<int> offset;
};

}  // namespace rtp_llm
