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
    // [layer_num, max_block_per_seq]
    std::vector<std::vector<void*>> k_ptr;
    std::vector<std::vector<void*>> v_ptr;

    std::vector<std::vector<void*>> k_scale_ptr;
    std::vector<std::vector<void*>> v_scale_ptr;
};

}  // namespace rtp_llm
