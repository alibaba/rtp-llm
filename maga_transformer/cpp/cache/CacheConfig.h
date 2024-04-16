#pragma once

#include "src/fastertransformer/core/Types.h"
#include <sstream>
#include <string>

namespace ft = fastertransformer;
namespace rtp_llm {

struct CacheConfig {
    uint         layer_num;
    uint         block_nums;
    size_t       block_size;
    uint         local_head_num_kv;
    uint         size_per_head;
    uint         seq_size_per_block;
    ft::DataType dtype;
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
        block_size =
            layer_num * local_head_num_kv * size_per_head * seq_size_per_block * fastertransformer::getTypeSize(dtype);
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "CacheConfig { "
                     << "layer_num: " << layer_num << ", block_nums: " << block_nums << ", block_size: " << block_size
                     << ", local_head_num_kv: " << local_head_num_kv << ", size_per_head: " << size_per_head
                     << ", seq_size_per_block: " << seq_size_per_block << ", dtype: " << dtype << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
