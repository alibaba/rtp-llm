#pragma once

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace rtp_llm {

enum KVCacheType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

enum MemoryLayout {
    LAYER_FIRST,  // [layer_num, num_blocks, block_size] -> hybrid attention
    KV_FIRST,     // [2, layer_num, num_blocks, kv_block_size] -> full attention
};

struct KVCacheSpec {
    uint32_t layer_num;
    uint32_t block_nums;
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;

    KVCacheType       type;
    rtp_llm::DataType dtype;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
    virtual size_t k_token_size() const = 0;
    virtual size_t v_token_size() const = 0;

    virtual ~KVCacheSpec() = default;
};

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;
    uint32_t scale_size = 0;

    size_t block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return 2 * local_head_num_kv * size_per_head * seq_size_per_block * dtype_size;
    }
    size_t k_block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return local_head_num_kv * size_per_head * seq_size_per_block * dtype_size;
    }
    size_t v_block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return local_head_num_kv * size_per_head * seq_size_per_block * dtype_size;
    }
    size_t k_token_size() const override {
        return size_per_head;
    }
    size_t v_token_size() const override {
        return size_per_head;
    }
};

struct MLAKVCacheSpec: public KVCacheSpec {
    uint32_t kv_lora_rank;
    uint32_t rope_head_dim;

    size_t block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return local_head_num_kv * (kv_lora_rank + rope_head_dim) * seq_size_per_block * dtype_size;
    }
    size_t k_block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return local_head_num_kv * kv_lora_rank * seq_size_per_block * dtype_size;
    }
    size_t v_block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return local_head_num_kv * rope_head_dim * seq_size_per_block * dtype_size;
    }
    size_t k_token_size() const override {
        return kv_lora_rank;
    }
    size_t v_token_size() const override {
        return rope_head_dim;
    }
};

struct LinearKVCacheSpec: public KVCacheSpec {
    uint32_t conv_state_size;
    uint32_t temporal_state_size;

    size_t block_size() const override {
        auto dtype_size = rtp_llm::getTypeSize(dtype);
        return (conv_state_size + temporal_state_size) * seq_size_per_block * dtype_size;
    }
    size_t k_block_size() const override {
        return conv_state_size * seq_size_per_block * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size() const override {
        return temporal_state_size * seq_size_per_block * rtp_llm::getTypeSize(dtype);
    }
    size_t k_token_size() const override {
        return conv_state_size;
    }
    size_t v_token_size() const override {
        return temporal_state_size;
    }
};

struct CacheConfig {
    std::vector<std::shared_ptr<KVCacheSpec>> cache_specs;
    std::vector<std::vector<int>>             layer_ids;

    bool enable_independent_pool = true;

    int    layer_num;
    int    block_num;
    int    block_size;              // including all layers
    size_t seq_size_per_block = 1;  // for cache_keys generation

    // for adpation to MLA
    bool        use_mla        = false;
    std::string mtp_model_type = "default_model";

    // for backward compatibility with old NormalBatchStreamProcessor
    size_t k_block_stride        = 0;  // for one layer
    size_t v_block_stride        = 0;  // for one layer
    size_t kv_block_stride       = 0;  // for one layer
    size_t kv_scale_block_stride = 0;

    CacheConfig() {}
};

struct BlockPoolConfig {
    uint32_t layer_num;
    uint32_t block_num;
    uint32_t block_size;

    MemoryLayout      layout = LAYER_FIRST;
    rtp_llm::DataType dtype  = rtp_llm::TYPE_INVALID;

    size_t total_size;

    // for kv first layout only, delete these fields in future
    size_t k_block_size;
    size_t v_block_size;

    size_t k_block_stride = 0;
    size_t v_block_stride = 0;

    size_t k_token_size = 0;
    size_t v_token_size = 0;

    bool is_mla = false;

    // extra meta for exposing logical shape to kernels
    // valid for KV_FIRST layout
    uint32_t local_head_num_kv  = 0;
    uint32_t seq_size_per_block = 0;
};

}  // namespace rtp_llm
