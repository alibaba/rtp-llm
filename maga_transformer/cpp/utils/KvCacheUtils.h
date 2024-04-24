#pragma once

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace rtp_llm {

void memcpyKvCache(uint64_t*                    kv_cache_blocks,
                   const std::vector<std::vector<void*>>& k_ptr,
                   const std::vector<std::vector<void*>>& v_ptr,
                   int                          layer_nums,
                   int                          max_block_size,
                   int                          total_batch_size,
                   int                          batch_idx)
{
    assert(k_ptr.size() == v_ptr.size() && layer_nums == k_ptr.size());
    const size_t layer_stride = total_batch_size * 2 * max_block_size;
    const size_t batch_begin  = batch_idx * 2 * max_block_size;
    for (size_t layer_id = 0; layer_id < layer_nums; ++layer_id) {
        std::memcpy(kv_cache_blocks + layer_id * layer_stride + batch_begin,
                    k_ptr[layer_id].data(),
                    k_ptr[layer_id].size() * sizeof(int64_t));
        std::memcpy(kv_cache_blocks + layer_id * layer_stride + batch_begin + max_block_size,
                    v_ptr[layer_id].data(),
                    v_ptr[layer_id].size() * sizeof(int64_t));
    }
}

}
