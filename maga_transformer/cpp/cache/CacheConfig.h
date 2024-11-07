#pragma once

#include "src/fastertransformer/core/Types.h"
#include <sstream>
#include <string>

namespace ft = fastertransformer;
namespace rtp_llm {

struct CacheConfig {
    uint32_t     layer_num;
    uint32_t     block_nums;
    uint32_t     local_head_num_kv;
    uint32_t     size_per_head;
    uint32_t     seq_size_per_block;
    ft::DataType dtype;
    size_t       block_size;
    size_t       kv_block_size;
    size_t       kv_scale_block_size;
    size_t       kv_block_stride;
    size_t       kv_scale_block_stride;
    size_t       total_size;
    size_t       reserve_runtime_mem_mb;
    
    CacheConfig() {}

    CacheConfig(uint         layer_num_,
                uint         block_nums_,
                uint         local_head_num_kv_,
                uint         size_per_head_,
                uint         seq_size_per_block_,
                ft::DataType dtype_):
        layer_num(layer_num_),
        block_nums(block_nums_),
        local_head_num_kv(local_head_num_kv_),
        size_per_head(size_per_head_),
        seq_size_per_block(seq_size_per_block_),
        dtype(dtype_) {

        auto dtype_size = ft::getTypeSize(dtype);
        int scale_size = 0;
        if (dtype == ft::TYPE_INT8 || dtype == ft::TYPE_FP8_E4M3) {
            scale_size = 4;
        }

        block_size = layer_num * local_head_num_kv * (size_per_head + scale_size) * seq_size_per_block * dtype_size * 2;
        kv_block_size = layer_num * local_head_num_kv * size_per_head * seq_size_per_block * dtype_size;
        kv_scale_block_size = layer_num * local_head_num_kv * scale_size * seq_size_per_block * dtype_size;

        // kv_block_stride is the size of a single block in a single layer
        kv_block_stride = kv_block_size / layer_num;
        kv_scale_block_stride = kv_scale_block_size / layer_num;

        refresh();
    }

    void refresh() {
        total_size = block_size * block_nums;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "CacheConfig { "
                     << "layer_num: " << layer_num 
                     << ", block_nums: " << block_nums
                     << ", block_size: " << block_size
                     << ", local_head_num_kv: " << local_head_num_kv
                     << ", size_per_head: " << size_per_head
                     << ", seq_size_per_block: " << seq_size_per_block
                     << ", dtype: " << int(dtype)
                     << ", kv_block_stride: " << kv_block_stride
                     << ", kv_scale_block_stride: " << kv_scale_block_stride
                     << ", total_size: " << total_size
                     << ", reserve_runtime_mem_mb: " << reserve_runtime_mem_mb << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
