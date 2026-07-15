
#pragma once

#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

struct MemoryLayoutConfig {
    uint32_t layer_num = 0;
    uint32_t block_num = 0;

    rtp_llm::DataType dtype = rtp_llm::TYPE_INVALID;

    // ---- Offsets within DeviceBlockPool global buffer ----
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

    bool is_mla  = false;  // true for scale 3D layout (MLA or indexer)
    bool use_mla = false;  // true for KV 3D layout (concat_and_cache_mla path only)
    // TODO(xinfei.sxf) rm head info
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    // Number of kernel blocks packed inside one DeviceBlockPool block.  When > 1,
    // DeviceBlockPool allocates physical blocks (each = bpk × kernel block bytes), but
    // kernels still address by kernel-block id; MemoryLayoutStrategy reshapes the
    // KV tensor as (layer, block_num × bpk, kv_block_stride_bytes / bpk) so the
    // kernel view sees per-kernel-block strides.
    size_t kernel_blocks_per_kv_block = 1;

    bool enable_kv_scale         = false;
    bool enable_hybrid_attention = false;

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};

}  // namespace rtp_llm