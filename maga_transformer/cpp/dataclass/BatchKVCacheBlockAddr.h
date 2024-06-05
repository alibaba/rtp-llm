#pragma once

#include "maga_transformer/cpp/cache/KVCacheBlockAddr.h"
#include <memory>

namespace rtp_llm {

class BatchKVCacheBlockAddr {
public:
    BatchKVCacheBlockAddr() {}
    void clear();
    void pushBack(const KVCacheBlockAddr& addr);
    void resize(size_t batch_id, size_t layer_id, int reserver_blocks);
    void append(size_t batch_id, const KVCacheBlockAddr& addr);
    std::string debugString() const;
    
public:
    // [batch_size, layer_num, max_block_per_seq]
    std::vector<std::vector<std::vector<void*>>> k_ptr;
    std::vector<std::vector<std::vector<void*>>> v_ptr;

    std::vector<std::vector<std::vector<void*>>> k_scale_ptr;
    std::vector<std::vector<std::vector<void*>>> v_scale_ptr;
};

}  // namespace rtp_llm
