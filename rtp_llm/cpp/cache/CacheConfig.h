#pragma once

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/types.h"
#include <string>
#include <memory>
#include <vector>

namespace rtp_llm {

enum KVCacheType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

enum MemoryLayout {
    LAYER_FIRST,  // [layer_num, num_blocks, block_size] -> hybrid attention
};

struct KVCacheSpec {
    uint32_t layer_num;
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;

    KVCacheType       type;
    rtp_llm::DataType dtype;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
    virtual size_t k_token_size() const = 0;
    virtual size_t v_token_size() const = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;
};

typedef std::shared_ptr<KVCacheSpec> KVCacheSpecPtr;

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;

    size_t block_size() const override {
        return 2 * local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }

    size_t block_size_bytes() const {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
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

    size_t block_size() const {
        return local_head_num_kv * (kv_lora_rank + rope_head_dim) * seq_size_per_block;
    }
    size_t k_block_size() const {
        return local_head_num_kv * kv_lora_rank * seq_size_per_block;
    }
    size_t v_block_size() const {
        return local_head_num_kv * rope_head_dim * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const override {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const override {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
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
        return (conv_state_size + temporal_state_size) * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return conv_state_size * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return temporal_state_size * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const override {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const override {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
    }

    size_t k_token_size() const override {
        return conv_state_size;
    }
    size_t v_token_size() const override {
        return temporal_state_size;
    }
};

struct CacheConfig {
    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<std::vector<int>> layer_ids;

    uint32_t layer_num;
    uint32_t block_num;

    // ---- Per-block sizes (all layers) ----
    // kv_block_*: kv cache only
    size_t kv_block_size       = 0;
    size_t kv_block_size_bytes = 0;
    // kv_scale_*: kv cache scale only (int8/fp8) (K+V together).
    size_t kv_scale_size       = 0;
    size_t kv_scale_size_bytes = 0;
    // block_*: kv cache + scale, for one logical "block" across all layers. (K+V scales together).
    size_t block_size       = 0;
    size_t block_size_bytes = 0;

    size_t seq_size_per_block = 1;  // for cache_keys generation

    // for adpation to MLA
    bool use_mla = false;

    // mtp
    std::string mtp_model_type = "default_model";

    // ---- Per-block strides (one layer) ----
    // kv_block_stride_*: one-layer kv cache block stride (K+V together).
    size_t kv_block_stride       = 0;
    size_t kv_block_stride_bytes = 0;
    // kv_scale_stride_*: one-layer kv cache scale stride for one logical block (K+V scales together).
    size_t kv_scale_stride       = 0;
    size_t kv_scale_stride_bytes = 0;
    // block_stride_*: one-layer total stride (kv + scale)
    size_t block_stride       = 0;
    size_t block_stride_bytes = 0;

    CacheConfig() {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "cache_specs size: " << cache_specs.size() << "\n"
            << "layer_ids size: " << layer_ids.size() << "\n"
            << "layer_num: " << layer_num << "\n"
            << "block_num: " << block_num << "\n"
            << "kv_block_size: " << kv_block_size << "\n"
            << "kv_block_size_bytes: " << kv_block_size_bytes << "\n"
            << "kv_scale_size: " << kv_scale_size << "\n"
            << "kv_scale_size_bytes: " << kv_scale_size_bytes << "\n"
            << "block_size: " << block_size << "\n"
            << "block_size_bytes: " << block_size_bytes << "\n"
            << "seq_size_per_block: " << seq_size_per_block << "\n"
            << "use_mla: " << use_mla << "\n"
            << "mtp_model_type: " << mtp_model_type << "\n"
            << "kv_block_stride: " << kv_block_stride << "\n"
            << "kv_block_stride_bytes: " << kv_block_stride_bytes << "\n"
            << "kv_scale_stride: " << kv_scale_stride << "\n"
            << "kv_scale_stride_bytes: " << kv_scale_stride_bytes << "\n"
            << "block_stride: " << block_stride << "\n"
            << "block_stride_bytes: " << block_stride_bytes;
        return oss.str();
    }
};

struct BlockPoolConfig {
    uint32_t layer_num;
    uint32_t block_num;

    MemoryLayout      layout = LAYER_FIRST;
    rtp_llm::DataType dtype  = rtp_llm::TYPE_INVALID;

    size_t total_size;
    size_t total_size_bytes;

    // ---- Per-block sizes (all layers) ----
    // kv_block_*: kv cache only
    size_t kv_block_size       = 0;
    size_t kv_block_size_bytes = 0;
    // kv_scale_*: scale only (includes BOTH K and V scales)
    size_t kv_scale_size       = 0;
    size_t kv_scale_size_bytes = 0;
    // block_*: kv cache + scale (logical block across all layers)
    size_t block_size       = 0;
    size_t block_size_bytes = 0;

    // for kv first layout only, keep these meta for partitioning / kernels
    size_t k_block_size = 0;
    size_t v_block_size = 0;

    // ---- Per-block strides (one layer) ----
    // kv_block_stride_*: one-layer kv cache block stride (K+V together)
    size_t kv_block_stride       = 0;
    size_t kv_block_stride_bytes = 0;
    // kv_scale_stride_*: one-layer scale stride for one logical block (K+V scales together)
    size_t kv_scale_stride       = 0;
    size_t kv_scale_stride_bytes = 0;
    // block_stride_*: one-layer total stride (kv + scale)
    size_t block_stride       = 0;
    size_t block_stride_bytes = 0;

    size_t k_block_stride = 0;
    size_t v_block_stride = 0;

    size_t k_block_size_bytes = 0;
    size_t v_block_size_bytes = 0;

    size_t k_block_stride_bytes = 0;
    size_t v_block_stride_bytes = 0;

    size_t k_token_size = 0;
    size_t v_token_size = 0;

    bool is_mla = false;

    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    bool   enable_kv_scale          = false;
    size_t kv_block_pool_size_bytes = 0;
    size_t kv_scale_offset_bytes    = 0;
    size_t kv_scale_block_bytes     = 0;
    size_t kv_scale_pool_size_bytes = 0;
};

}  // namespace rtp_llm
