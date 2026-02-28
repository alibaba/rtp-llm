
#pragma once

#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

struct MemoryLayoutConfig {
    uint32_t layer_num = 0;
    uint32_t block_num = 0;

    rtp_llm::DataType dtype = rtp_llm::TYPE_INVALID;

    // ---- Offsets within BlockPool global buffer ----
    size_t kv_cache_offset_bytes = 0;
    size_t kv_scale_offset_bytes = 0;

    // ---- Pool sizes ----
    size_t kv_block_pool_size_bytes = 0;
    size_t kv_scale_pool_size_bytes = 0;
    size_t total_size_bytes         = 0;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;
    size_t block_stride_bytes    = 0;

    // For partitioning / kernels (KV separation info)
    size_t k_block_stride_bytes = 0;
    size_t v_block_stride_bytes = 0;
    size_t k_scale_stride_bytes = 0;
    size_t v_scale_stride_bytes = 0;

    bool is_mla = false;
    // TODO(xinfei.sxf) rm head info
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    bool enable_kv_scale         = false;
    bool enable_hybrid_attention = false;

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};

}  // namespace rtp_llm