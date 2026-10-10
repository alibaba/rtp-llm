#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    std::optional<bool> enable_prefix_reuse;
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

    // For KERNEL_BLOCK_COMPRESSED these describe each kernel page, before physical-block aggregation.
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

// Cache-internal physical entry resolution during synchronous spec building.
// The descriptor above remains a model declaration.
struct ResolvedStateEntryLayout {
    uint32_t seq_size_per_block = 0;  // original tokens per physical STATE block
    uint32_t full_entries       = 0;  // aligned entries in the complete ring
    uint32_t local_entry_begin  = 0;  // first complete-ring entry on this prefill rank
    uint32_t local_entries      = 0;  // physical entries in this process's block
    size_t   entry_bytes        = 0;
};
using ResolvedStateEntryLayouts = std::unordered_map<std::string, ResolvedStateEntryLayout>;

// Physical packed-byte projection for BLOCK_STRIDE fixed regions. Unlike a
// STATE entry slice, a rank boundary may fall inside an FP8 data/scale row.
struct ResolvedSwaByteLayout {
    uint32_t seq_size_per_block = 0;  // original tokens per physical SWA block
    uint32_t full_entries       = 0;  // complete logical ring, before byte slicing
    size_t   full_payload_bytes = 0;  // data and scale, excluding trailing padding
    size_t   full_stride_bytes  = 0;  // data, scale and trailing padding
    size_t   local_byte_begin   = 0;  // offset in the complete packed block
    size_t   local_stride_bytes = 0;  // rank-local allocated bytes
};
using ResolvedSwaByteLayouts = std::unordered_map<std::string, ResolvedSwaByteLayout>;

// Physical span and per-kernel-page entries after cache CP resolution.
// Borrowed only while building specs; it does not own any cache resources.
struct ResolvedCompressedLayout {
    uint32_t seq_size_per_block       = 0;  // original tokens per physical OpaqueKV block
    uint32_t entries_per_kernel_block = 0;  // complete entries in each kernel page before physical aggregation
};
using ResolvedCompressedLayouts = std::unordered_map<std::string, ResolvedCompressedLayout>;

// Cache build requirements. A tag here must have one of the physical
// projections above. Borrowed only during synchronous spec construction.
using RequiredSpecProjectionTags = std::unordered_set<std::string>;

// One plain execution-policy overlay per descriptor, indexed by layer and
// descriptor position. Keeping the positions preserves cross-layer policy
// consistency checks in CacheConfigCreator.
struct ResolvedCpGroupPolicy {
    CpBlockMappingMode mapping = CpBlockMappingMode::NONE;
    CpBlockSliceMode   slice   = CpBlockSliceMode::NONE;
    bool uses_physical_projection = false;
};
using ResolvedCpGroupPolicies = std::vector<std::vector<ResolvedCpGroupPolicy>>;

struct SpecBuildContext {
    DataType                     dtype                   = DataType::TYPE_INVALID;
    uint32_t                     seq_size_per_block      = 0;  // tokens/base cache-key block before group projection
    uint32_t                     kernel_tokens_per_block = 0;
    const AttentionConfigs*      attn_config             = nullptr;
    const LinearAttentionConfig* linear_attention_config = nullptr;
    const ParallelismConfig*     parallelism_config      = nullptr;
    uint32_t                     gen_num_per_cycle       = 0;
    const ResolvedStateEntryLayouts* state_entry_layouts = nullptr;
    const ResolvedSwaByteLayouts*    swa_byte_layouts    = nullptr;
    // CacheConfigCreator supplies this map during synchronous spec building.
    // SpecBuilder embeds the values in the constructed spec.
    const ResolvedCompressedLayouts* compressed_layouts = nullptr;
    const RequiredSpecProjectionTags* required_spec_projection_tags = nullptr;
};

class SpecBuilder {
public:
    static KVCacheSpecPtr   build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx);
    static CacheGroupType   groupType(const KVCacheSpecDesc& desc);
    static CacheGroupPolicy groupPolicy(const KVCacheSpecDesc& desc);
    static uint32_t         kernelSeqSizePerBlock(const KVCacheSpecDesc&  desc,
                                                  const SpecBuildContext& ctx,
                                                  uint32_t                physical_seq_size_per_block);
};

}  // namespace rtp_llm
