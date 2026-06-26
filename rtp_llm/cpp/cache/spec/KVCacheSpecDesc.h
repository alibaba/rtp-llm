#pragma once

#include <algorithm>
#include <memory>
#include <numeric>

#include "rtp_llm/cpp/cache/spec/KVCacheSpecDescTypes.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct SpecBuildContext {
    DataType dtype                   = DataType::TYPE_INVALID;
    uint32_t seq_size_per_block      = 0;
    uint32_t attn_tp_size            = 1;  // TP size for computing local head counts from global desc fields
    uint32_t kernel_tokens_per_block = 0;
    uint32_t gen_num_per_cycle       = 0;
    uint32_t cp_size                 = 1;
    bool     cp_prefill_sliced       = false;
};

class SpecBuilder {
public:
    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(!desc.tag.empty(), "KVCacheSpecDesc tag must not be empty");
        auto spec = buildTyped(desc, ctx);
        spec->tag                = desc.tag;
        spec->seq_size_per_block = effectiveSeqSizePerBlock(desc, ctx);
        spec->dtype              = dataTypeOr(desc.dtype, dataTypeOr(ctx.dtype, desc.store_dtype));
        return spec;
    }

    static CacheGroupType groupType(const KVCacheSpecDesc& desc) {
        switch (desc.cache_type) {
            case CacheType::LINEAR:
                return CacheGroupType::LINEAR;
            case CacheType::FIXED_STATE:
                return CacheGroupType::SWA;
            case CacheType::MHA:
            case CacheType::MLA:
            case CacheType::COMPRESSED_KV:
                return CacheGroupType::FULL;
        }
        return CacheGroupType::FULL;
    }

private:
    static uint32_t valueOr(uint32_t value, uint32_t fallback) {
        return value == 0 ? fallback : value;
    }

    static DataType dataTypeOr(DataType value, DataType fallback) {
        return value == DataType::TYPE_INVALID ? fallback : value;
    }

    static uint32_t alignUpToMultiple(uint32_t value, uint32_t multiple) {
        RTP_LLM_CHECK_WITH_INFO(multiple > 0, "align multiple must be > 0");
        return ((value + multiple - 1) / multiple) * multiple;
    }

    static uint32_t effectiveSeqSizePerBlock(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const auto ctx_seq_size = valueOr(ctx.seq_size_per_block, 1);
        if (desc.extra.use_fixed_region_cp_tokens && ctx.cp_size > 1) {
            return ctx_seq_size * ctx.cp_size;
        }
        return valueOr(desc.seq_size_per_block, ctx_seq_size);
    }

    static uint32_t computeStateRingEntries(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(desc.extra.state_ring_compression_ratio > 0,
                                "state ring desc tag=%s requires positive state_ring_compression_ratio",
                                desc.tag.c_str());
        const uint32_t window =
            (1 + desc.extra.state_ring_overlap) * desc.extra.state_ring_compression_ratio;
        const uint32_t raw =
            window + (desc.extra.state_ring_add_gen_num_per_cycle ? ctx.gen_num_per_cycle : 0);
        return (raw + 1) & ~1U;
    }

    static uint32_t effectiveEntriesPerBlock(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        if (desc.extra.derive_entries_from_kernel_block) {
            RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                                    desc.tag.c_str(),
                                    desc.compression_ratio);
            RTP_LLM_CHECK_WITH_INFO(ctx.kernel_tokens_per_block > 0,
                                    "desc tag=%s derives entries from kernel block but kernel_tokens_per_block is 0",
                                    desc.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(ctx.kernel_tokens_per_block % desc.compression_ratio == 0,
                                    "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                    desc.tag.c_str(),
                                    desc.compression_ratio,
                                    ctx.kernel_tokens_per_block);
            return ctx.kernel_tokens_per_block / desc.compression_ratio;
        }

        if (desc.extra.state_ring_compression_ratio > 0) {
            uint32_t entries = computeStateRingEntries(desc, ctx);
            if (ctx.cp_size > 1 && (desc.extra.cp_align_entries || desc.extra.cp_slice_entries)) {
                entries = alignUpToMultiple(entries, ctx.cp_size);
                if (desc.extra.cp_slice_entries && ctx.cp_prefill_sliced) {
                    entries /= ctx.cp_size;
                }
            }
            return entries;
        }

        return desc.entries_per_block;
    }

    static size_t effectiveFixedStateBlockOverride(const KVCacheSpecDesc& desc,
                                                   uint32_t               entries_per_block,
                                                   const SpecBuildContext& ctx) {
        if (ctx.cp_size <= 1 || !ctx.cp_prefill_sliced || !desc.extra.cp_prefill_slice_block_bytes) {
            return desc.block_size_bytes_override;
        }
        const size_t natural_bytes = static_cast<size_t>(entries_per_block) * desc.entry_elems * getTypeSize(desc.store_dtype);
        const size_t align =
            desc.block_size_bytes_alignment > 0 ?
                std::lcm(desc.block_size_bytes_alignment, static_cast<size_t>(ctx.cp_size)) :
                static_cast<size_t>(ctx.cp_size);
        const size_t full_stride_bytes = ((natural_bytes + align - 1) / align) * align;
        RTP_LLM_CHECK_WITH_INFO(full_stride_bytes % ctx.cp_size == 0,
                                "CP prefill byte slicing tag=%s full stride %zu must be divisible by cp_size %u",
                                desc.tag.c_str(),
                                full_stride_bytes,
                                ctx.cp_size);
        return full_stride_bytes / ctx.cp_size;
    }

    // Dispatch to per-type factory methods.
    // Each factory method owns all type-specific field assignments,
    // including local_head_num_kv derived from global desc fields and runtime TP size.
    static KVCacheSpecPtr buildTyped(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        switch (desc.cache_type) {
            case CacheType::MHA:           return buildMHA(desc, ctx);
            case CacheType::MLA:           return buildMLA(desc);
            case CacheType::LINEAR:        return buildLinear(desc, ctx);
            case CacheType::COMPRESSED_KV: return buildCompressedKV(desc, ctx);
            case CacheType::FIXED_STATE:   return buildFixedState(desc, ctx);
        }
        RTP_LLM_CHECK_WITH_INFO(false, "unknown CacheType=%d", static_cast<int>(desc.cache_type));
        return nullptr;
    }

    // MHA/GQA: local_head_num_kv = global_kv_heads / TP, with gcd fallback for non-divisible TP.
    static KVCacheSpecPtr buildMHA(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const uint32_t tp   = std::max(1u, ctx.attn_tp_size);
        auto spec               = std::make_shared<MHAKVCacheSpec>();
        spec->size_per_head     = desc.size_per_head;
        const uint32_t kv       = valueOr(desc.num_kv_heads, 1);
        spec->local_head_num_kv = (kv % tp == 0) ? kv / tp : kv / std::gcd(kv, tp);
        return spec;
    }

    // MLA: local_head_num_kv is always 1 — heads are not split across TP.
    static KVCacheSpecPtr buildMLA(const KVCacheSpecDesc& desc) {
        auto spec               = std::make_shared<MLAKVCacheSpec>();
        spec->kv_lora_rank      = desc.kv_lora_rank;
        spec->rope_head_dim     = desc.rope_head_dim;
        spec->local_head_num_kv = 1;
        return spec;
    }

    // Linear Attention: all three local head fields derived from global counts / TP.
    static KVCacheSpecPtr buildLinear(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const uint32_t tp   = std::max(1u, ctx.attn_tp_size);
        auto spec               = std::make_shared<LinearKVCacheSpec>();
        spec->local_num_k_heads = desc.num_k_heads / tp;
        spec->local_num_v_heads = desc.num_v_heads / tp;
        spec->head_k_dim        = desc.head_k_dim;
        spec->head_v_dim        = desc.head_v_dim;
        spec->conv_kernel_dim   = desc.conv_kernel_dim;
        spec->ssm_state_dtype   = dataTypeOr(desc.ssm_state_dtype, DataType::TYPE_BF16);
        spec->conv_state_dtype  = dataTypeOr(desc.conv_state_dtype, DataType::TYPE_BF16);
        const uint32_t v_heads  = valueOr(desc.num_v_heads, 1);
        spec->local_head_num_kv = std::max(1u, (v_heads > 1u) ? v_heads / tp : v_heads);
        return spec;
    }

    // COMPRESSED_KV / FIXED_STATE: no per-head TP split, local_head_num_kv = global value.
    static KVCacheSpecPtr buildCompressedKV(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        auto spec                        = std::make_shared<CompressedKVCacheSpec>();
        spec->entry_elems                = desc.entry_elems;
        spec->entries_per_block          = effectiveEntriesPerBlock(desc, ctx);
        spec->compression_ratio          = valueOr(desc.compression_ratio, 1);
        spec->store_dtype                = desc.store_dtype;
        spec->block_size_bytes_alignment = desc.block_size_bytes_alignment;
        spec->local_head_num_kv          = valueOr(desc.num_kv_heads, 1);
        return spec;
    }

    static KVCacheSpecPtr buildFixedState(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const auto entries_per_block            = effectiveEntriesPerBlock(desc, ctx);
        auto spec                              = std::make_shared<FixedStateCacheSpec>();
        spec->state_dim                        = desc.entry_elems;
        spec->entries_per_block                = entries_per_block;
        spec->store_dtype                      = desc.store_dtype;
        spec->block_size_bytes_override        = effectiveFixedStateBlockOverride(desc, entries_per_block, ctx);
        spec->block_size_bytes_alignment       = desc.block_size_bytes_alignment;
        spec->block_size_alignment_min_entries = desc.block_size_alignment_min_entries;
        spec->is_state_cache                   = desc.is_state_cache;
        spec->skip_prefix_reuse                = desc.skip_prefix_reuse;
        spec->local_head_num_kv                = valueOr(desc.num_kv_heads, 1);
        return spec;
    }
};

}  // namespace rtp_llm
