
#pragma once

#include "rtp_llm/cpp/cache/KVCacheSpecType.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

struct MemoryLayoutConfig {
    uint32_t layer_num = 0;
    uint32_t block_num = 0;

    rtp_llm::DataType dtype = rtp_llm::TYPE_INVALID;
    // Copied from this group's KVCacheSpec, independently of model-level attention flags.
    KVCacheSpecType cache_type = KVCacheSpecType::MultiHeadAttention;

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

    // TODO(xinfei.sxf) rm head info
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    // Number of kernel pages in one physical allocation. LayerKVCache owns the
    // kernel-page projection; MemoryLayoutStrategy retains physical block IDs.
    size_t kernel_blocks_per_kv_block = 1;

    bool enable_kv_scale         = false;

    // Optional physical placement. The allocator still owns block_num logical
    // IDs (including reserved ID zero). GPU scratch follows the HBM prefix and
    // has no logical block ID. A zero triple keeps the pool-wide placement.
    uint32_t device_block_count  = 0;
    uint32_t device_spare_blocks = 0;
    uint32_t host_block_count    = 0;

    bool hasExplicitBacking() const {
        return device_block_count || device_spare_blocks || host_block_count;
    }

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};

}  // namespace rtp_llm
