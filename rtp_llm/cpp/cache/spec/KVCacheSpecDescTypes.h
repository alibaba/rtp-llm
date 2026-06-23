#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

enum class CacheType : int8_t {
    MHA           = 0,
    MLA           = 1,
    LINEAR        = 2,
    COMPRESSED_KV = 3,
    FIXED_STATE   = 4,
};

struct KVCacheSpecDescExtra {
    uint32_t explicit_block_num        = 0;
    bool     reserve_from_paged_budget = false;

    bool     derive_entries_from_kernel_block = false;
    uint32_t state_ring_compression_ratio     = 0;
    uint32_t state_ring_overlap               = 0;
    bool     state_ring_add_gen_num_per_cycle = false;
    bool     cp_align_entries                 = false;
    bool     cp_slice_entries                 = false;
    bool     cp_prefill_slice_block_bytes     = false;
    bool     use_fixed_region_cp_tokens       = false;
};

struct KVCacheSpecDesc {
    std::string tag;
    CacheType   cache_type = CacheType::MHA;
    bool        has_group_order = false;
    uint32_t    group_order     = 0;

    uint32_t local_head_num_kv  = 0;
    uint32_t seq_size_per_block = 0;
    DataType dtype              = DataType::TYPE_INVALID;

    uint32_t size_per_head = 0;
    uint32_t kv_lora_rank  = 0;
    uint32_t rope_head_dim = 0;

    uint32_t local_num_k_heads = 0;
    uint32_t local_num_v_heads = 0;
    uint32_t head_k_dim        = 0;
    uint32_t head_v_dim        = 0;
    uint32_t conv_kernel_dim   = 0;
    DataType ssm_state_dtype   = DataType::TYPE_INVALID;
    DataType conv_state_dtype  = DataType::TYPE_INVALID;

    uint32_t entry_elems                      = 0;
    uint32_t entries_per_block                = 0;
    DataType store_dtype                      = DataType::TYPE_INVALID;
    uint32_t compression_ratio                = 1;
    size_t   block_size_bytes_override        = 0;
    size_t   block_size_bytes_alignment       = 0;
    uint32_t block_size_alignment_min_entries = 0;
    bool     is_state_cache                   = true;
    bool     skip_prefix_reuse                = false;

    bool             has_reuse_policy          = false;
    CacheReusePolicy reuse_policy              = CacheReusePolicy::REUSABLE;
    bool             has_evict_policy          = false;
    CacheEvictPolicy evict_policy              = CacheEvictPolicy::CHAIN;
    bool             has_active_tail_blocks    = false;
    int              active_tail_blocks        = 0;
    bool             has_validate_tail_blocks  = false;
    bool             validate_tail_blocks      = true;
    KVCacheSpecDescExtra extra;
    bool             has_prefix_reusable       = false;
    bool             prefix_reusable           = true;
    bool             uses_pinned_cpu_backing   = false;
    bool             has_is_cp_shardable       = false;
    bool             is_cp_shardable           = true;
    bool             has_sparse_slots          = false;
    bool             sparse_slots              = false;
    bool             has_kernel_block_subdiv   = false;
    bool             kernel_block_subdiv       = true;
    bool             has_cp_compact_tail_blocks = false;
    bool             cp_compact_tail_blocks     = false;
    bool             has_is_reservable          = false;
    bool             is_reservable              = true;

};

}  // namespace rtp_llm
