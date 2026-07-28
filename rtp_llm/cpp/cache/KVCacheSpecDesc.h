#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

enum class OpaqueBlockEntryCountMode : int8_t {
    EXPLICIT                = 0,
    KERNEL_BLOCK_COMPRESSED = 1,
    STATE_RING              = 2,
};

enum class CpPrefillSliceLayout : int8_t {
    NONE         = 0,
    PAYLOAD      = 1,
    BLOCK_STRIDE = 2,
};

struct CacheReusePolicyDesc {
    std::optional<bool>             enable_prefix_reuse;
    std::optional<CacheEvictPolicy> evict_policy;
};

struct CacheCapacityPolicyDesc {
    std::optional<bool>     reservable;
    std::optional<uint32_t> explicit_block_num;
    std::optional<bool>     charge_to_paged_budget;
};

struct CacheMemoryPolicyDesc {
    std::optional<CacheMemoryPlacement> placement;
};

struct CacheTailPolicyDesc {
    std::optional<uint32_t> active_tail_blocks;
    std::optional<bool>     validate_tail_blocks;
};

struct CacheCpPolicyDesc {
    std::optional<CpBlockMappingMode>   mapping;
    std::optional<CpBlockSliceMode>     slice;
    std::optional<bool>                 scale_seq_size;
    std::optional<bool>                 align_payload;
    std::optional<CpPrefillSliceLayout> prefill_slice_layout;
};

struct KVCacheSpecDesc {
    std::string     tag;
    KVCacheSpecType cache_type     = KVCacheSpecType::MultiHeadAttention;
    DataType        dtype          = DataType::TYPE_INVALID;
    bool            is_state_cache = false;

    uint32_t entry_elems = 0;
    DataType entry_dtype = DataType::TYPE_INVALID;

    OpaqueBlockEntryCountMode entry_count_mode                     = OpaqueBlockEntryCountMode::EXPLICIT;
    uint32_t                  explicit_entry_count                 = 0;
    uint32_t                  compression_ratio                    = 1;
    uint32_t                  state_ring_overlap                   = 0;
    bool                      state_ring_include_gen_num_per_cycle = false;

    size_t   block_stride_bytes_override        = 0;
    size_t   block_stride_bytes_alignment       = 0;
    uint32_t block_stride_alignment_min_entries = 0;

    std::optional<CacheGroupType>          group_type;
    std::optional<CacheReusePolicyDesc>    reuse;
    std::optional<CacheCapacityPolicyDesc> capacity;
    std::optional<CacheMemoryPolicyDesc>   memory;
    std::optional<CacheTailPolicyDesc>     tail;
    std::optional<CacheCpPolicyDesc>       cp;
};

struct SpecBuildContext {
    DataType                     dtype                   = DataType::TYPE_INVALID;
    uint32_t                     seq_size_per_block      = 0;
    uint32_t                     kernel_tokens_per_block = 0;
    const AttentionConfigs*      attn_config             = nullptr;
    const LinearAttentionConfig* linear_attention_config = nullptr;
    const ParallelismConfig*     parallelism_config      = nullptr;
    uint32_t                     gen_num_per_cycle       = 0;
};

class SpecBuilder {
public:
    static KVCacheSpecPtr   build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx);
    static CacheGroupType   groupType(const KVCacheSpecDesc& desc);
    static CacheGroupPolicy groupPolicy(const KVCacheSpecDesc& desc);
};

}  // namespace rtp_llm
