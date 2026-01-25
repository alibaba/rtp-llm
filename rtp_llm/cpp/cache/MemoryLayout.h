#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

enum MemoryLayout {
    LAYER_FIRST,  // [layer_num, num_blocks, block_size]
};

struct MemoryLayoutConfig {
    uint32_t layer_num = 0;
    uint32_t block_num = 0;

    MemoryLayout      layout = LAYER_FIRST;
    rtp_llm::DataType dtype  = rtp_llm::TYPE_INVALID;

    // ---- Offsets within BlockPool global buffer ----
    // kv cache pool base offset
    size_t kv_cache_offset_bytes = 0;
    // kv scale pool base offset (valid only when enable_kv_scale == true)
    size_t kv_scale_offset_bytes = 0;

    // ---- Pool sizes ----
    size_t kv_block_pool_size_bytes = 0;
    size_t kv_scale_pool_size_bytes = 0;
    size_t total_size_bytes         = 0;

    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size       = 0;
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size       = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size          = 0;
    size_t block_size_bytes    = 0;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride       = 0;
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride       = 0;
    size_t kv_scale_stride_bytes = 0;
    size_t block_stride          = 0;
    size_t block_stride_bytes    = 0;

    // For partitioning / kernels (KV separation info)
    size_t k_block_size         = 0;
    size_t v_block_size         = 0;
    size_t k_block_stride       = 0;
    size_t v_block_stride       = 0;
    size_t k_block_size_bytes   = 0;
    size_t v_block_size_bytes   = 0;
    size_t k_block_stride_bytes = 0;
    size_t v_block_stride_bytes = 0;
    size_t k_scale_stride_bytes = 0;
    size_t v_scale_stride_bytes = 0;
    size_t k_dim                = 0;
    size_t v_dim                = 0;

    bool   is_mla             = false;
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    bool enable_kv_scale = false;

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};

}  // namespace rtp_llm