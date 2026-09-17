
#pragma once

#include "rtp_llm/models_py/bindings/core/Types.h"

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

    bool is_mla  = false;  // true for scale 3D layout (MLA or indexer)
    bool use_mla = false;  // true for KV 3D layout (concat_and_cache_mla path only)
    // Linear-cache blocks use [SSM heads][history][Q | K | V heads].
    // Asymmetric-TP PD transfer therefore needs one SSM segment plus three
    // segments per convolution-history entry.
    bool   is_linear_attention           = false;
    bool   enable_linear_cache_partition = false;
    size_t linear_num_k_heads            = 0;
    size_t linear_num_v_heads            = 0;
    size_t linear_conv_history           = 0;
    size_t linear_q_bytes_per_history    = 0;
    size_t linear_k_bytes_per_history    = 0;
    size_t linear_v_bytes_per_history    = 0;
    // TODO(xinfei.sxf) rm head info
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    // Number of kernel blocks packed inside one BlockPool block.  When > 1,
    // BlockPool allocates physical blocks (each = bpk × kernel block bytes), but
    // kernels still address by kernel-block id; MemoryLayoutStrategy reshapes the
    // KV tensor as (layer, block_num × bpk, kv_block_stride_bytes / bpk) so the
    // kernel view sees per-kernel-block strides.
    size_t kernel_blocks_per_kv_block = 1;

    // Tiered MLA only: logical block IDs below this boundary live in HBM;
    // remaining IDs address a registered host arena starting at storage block 0.
    // The HBM allocation ends with a token working set, aligned to kernel pages
    // (not necessarily to the larger allocator blocks).
    uint32_t mla_hbm_blocks      = 0;
    size_t   mla_resident_tokens = 0;

    bool hasMlaHostCache() const {
        return mla_resident_tokens > 0;
    }

    size_t mlaHbmSizeBytes() const {
        // RDMA chunks must not split allocator blocks, including across layers.
        // Padding is allocation-only; kernels see the requested token capacity.
        const size_t resident_blocks =
            mla_resident_tokens / seq_size_per_block + (mla_resident_tokens % seq_size_per_block != 0);
        return static_cast<size_t>(layer_num) * (static_cast<size_t>(mla_hbm_blocks) + resident_blocks)
               * kv_block_stride_bytes;
    }

    bool enable_kv_scale         = false;
    bool enable_hybrid_attention = false;
    // Opaque typed cache regions (DSV4 paged/state pools) are replicated as
    // complete blocks across attention TP ranks. They must not be interpreted
    // as ordinary head-partitioned K/V storage during PD cache transfer.
    bool enable_kv_cache_partition = true;

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};

}  // namespace rtp_llm
