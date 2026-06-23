#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/spec/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::test {

inline constexpr uint32_t DSV4_FP8_KV_ENTRY_BYTES            = 584;
inline constexpr uint32_t DSV4_FP8_INDEXER_ENTRY_BYTES       = 132;
inline constexpr size_t   DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;
inline constexpr uint32_t DSV4_SWA_WINDOW_ENTRIES            = 128;

inline size_t alignDsv4Fp8KvBlockBytes(size_t natural, size_t extra_multiple = 1) {
    const size_t align = std::lcm(DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES, std::max<size_t>(extra_multiple, 1));
    return ((natural + align - 1) / align) * align;
}

inline KVCacheSpecPtr makeDsv4Spec(const std::string& tag,
                                    const std::string& kind,
                                    uint32_t           entry_elems,
                                    DataType           dtype,
                                    uint32_t           compression_ratio = 1) {
    KVCacheSpecPtr spec;
    if (kind == "compressed_kv") {
        auto kv_spec               = std::make_shared<CompressedKVCacheSpec>();
        kv_spec->entry_elems       = entry_elems;
        kv_spec->compression_ratio = compression_ratio;
        kv_spec->store_dtype       = dtype;
        spec                       = kv_spec;
    } else {
        auto state_spec        = std::make_shared<FixedStateCacheSpec>();
        state_spec->state_dim  = entry_elems;
        state_spec->store_dtype = dtype;
        spec                   = state_spec;
    }
    spec->tag                = tag;
    spec->dtype              = dtype;
    return spec;
}

inline uint32_t dsv4GroupOrder(const std::string& tag) {
    if (tag == "csa_kv") {
        return 0;
    }
    if (tag == "hca_kv") {
        return 1;
    }
    if (tag == "indexer_kv") {
        return 2;
    }
    if (tag == "indexer_state") {
        return 3;
    }
    if (tag == "csa_state") {
        return 4;
    }
    if (tag == "hca_state") {
        return 5;
    }
    if (tag == "swa_kv") {
        return 6;
    }
    RTP_LLM_FAIL("unknown DSV4 test tag=%s", tag.c_str());
    return 0;
}

inline KVCacheSpecDesc dsv4DescForSpec(const KVCacheSpecPtr& spec) {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "dsv4DescForSpec got null spec");
    KVCacheSpecDesc desc;
    desc.tag             = spec->tag;
    desc.has_group_order = true;
    desc.group_order     = dsv4GroupOrder(spec->tag);
    desc.dtype           = spec->dtype;
    if (auto* compressed = dynamic_cast<CompressedKVCacheSpec*>(spec.get())) {
        desc.cache_type                 = CacheType::COMPRESSED_KV;
        desc.is_state_cache             = false;
        desc.entry_elems                = compressed->entry_elems;
        desc.compression_ratio          = compressed->compression_ratio;
        desc.store_dtype                = compressed->store_dtype;
        desc.block_size_bytes_alignment = compressed->block_size_bytes_alignment;
        desc.extra.derive_entries_from_kernel_block = true;
        if (desc.block_size_bytes_alignment == 0 && desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_size_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
        return desc;
    }

    auto* fixed = dynamic_cast<FixedStateCacheSpec*>(spec.get());
    RTP_LLM_CHECK_WITH_INFO(fixed != nullptr, "DSV4 test spec tag=%s must be opaque", spec->tag.c_str());
    desc.cache_type                       = CacheType::FIXED_STATE;
    desc.entry_elems                      = fixed->entry_elems;
    desc.store_dtype                      = fixed->store_dtype;
    desc.block_size_bytes_override        = fixed->block_size_bytes_override;
    desc.block_size_bytes_alignment       = fixed->block_size_bytes_alignment;
    desc.block_size_alignment_min_entries = fixed->block_size_alignment_min_entries;
    if (desc.tag == "indexer_state" || desc.tag == "csa_state") {
        desc.extra.state_ring_compression_ratio = 4;
        desc.extra.state_ring_overlap           = 1;
        desc.extra.cp_align_entries             = true;
        desc.extra.cp_slice_entries             = true;
    } else if (desc.tag == "hca_state") {
        desc.extra.state_ring_compression_ratio = 128;
        desc.extra.cp_align_entries             = true;
        desc.extra.cp_slice_entries             = true;
        desc.extra.explicit_block_num           = 256;
        desc.skip_prefix_reuse                  = true;
        desc.has_reuse_policy                   = true;
        desc.reuse_policy                       = CacheReusePolicy::NON_REUSABLE;
        desc.has_active_tail_blocks             = true;
        desc.active_tail_blocks                 = 1;
        desc.has_validate_tail_blocks           = true;
        desc.validate_tail_blocks               = false;
    } else if (desc.tag == "swa_kv") {
        desc.extra.state_ring_compression_ratio = DSV4_SWA_WINDOW_ENTRIES;
        desc.extra.cp_align_entries             = true;
        desc.extra.cp_prefill_slice_block_bytes = true;
        if (desc.block_size_bytes_alignment == 0 && desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_size_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
    }
    desc.extra.state_ring_add_gen_num_per_cycle = true;
    desc.extra.use_fixed_region_cp_tokens       = true;
    desc.block_size_alignment_min_entries =
        desc.block_size_alignment_min_entries == 0 ? DSV4_SWA_WINDOW_ENTRIES : desc.block_size_alignment_min_entries;
    desc.is_state_cache     = true;
    desc.has_evict_policy   = true;
    desc.evict_policy       = CacheEvictPolicy::INDEPENDENT;
    return desc;
}

inline void setDefaultKvCacheSpec(ModelConfig& model_config) {
    std::vector<int> layers;
    layers.reserve(static_cast<size_t>(model_config.num_layers));
    for (int i = 0; i < static_cast<int>(model_config.num_layers); ++i) {
        layers.push_back(i);
    }

    KVCacheSpecPtr spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto mla_spec           = std::make_shared<MLAKVCacheSpec>();
        mla_spec->type          = KVCacheSpecType::MultiHeadLatentAttention;
        mla_spec->kv_lora_rank  = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        mla_spec->rope_head_dim = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
        spec                    = mla_spec;
    } else {
        auto mha_spec            = std::make_shared<MHAKVCacheSpec>();
        mha_spec->type           = KVCacheSpecType::MultiHeadAttention;
        mha_spec->size_per_head  = static_cast<uint32_t>(model_config.attn_config.size_per_head);
        spec                     = mha_spec;
    }
    spec->tag                = "default";
    spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    model_config.kv_cache_specs.clear();
    for (int layer_id : layers) {
        model_config.kv_cache_specs[static_cast<int64_t>(layer_id)] = {spec};
    }
}

inline void setHybridAttentionKvCacheSpecs(ModelConfig& model_config) {
    std::vector<int> full_layers;
    std::vector<int> linear_layers;
    const auto&      types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    for (int i = 0; i < static_cast<int>(model_config.num_layers); ++i) {
        if (types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            linear_layers.push_back(i);
        } else {
            full_layers.push_back(i);
        }
    }

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->tag                = "full";
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    full_spec->size_per_head      = static_cast<uint32_t>(model_config.attn_config.size_per_head);

    const auto& linear_config           = model_config.linear_attention_config;
    auto        linear_spec             = std::make_shared<LinearKVCacheSpec>();
    linear_spec->tag                    = "linear";
    linear_spec->type                   = KVCacheSpecType::LinearAttention;
    linear_spec->seq_size_per_block     = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    linear_spec->head_k_dim             = static_cast<uint32_t>(linear_config.linear_key_head_dim);
    linear_spec->head_v_dim             = static_cast<uint32_t>(linear_config.linear_value_head_dim);
    linear_spec->conv_kernel_dim        = static_cast<uint32_t>(linear_config.linear_conv_kernel_dim);
    linear_spec->ssm_state_dtype        = linear_config.ssm_state_dtype;
    linear_spec->conv_state_dtype       = linear_config.conv_state_dtype;
    model_config.kv_cache_specs.clear();
    for (int layer_id : full_layers) {
        model_config.kv_cache_specs[static_cast<int64_t>(layer_id)] = {full_spec};
    }
    for (int layer_id : linear_layers) {
        model_config.kv_cache_specs[static_cast<int64_t>(layer_id)] = {linear_spec};
    }
}

inline void setDsv4KvCacheSpecs(ModelConfig& model_config) {
    const int layer_num = static_cast<int>(model_config.num_layers);

    const bool     fp8_kv              = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8;
    const uint32_t kv_entry_elems      = fp8_kv ? 584 : static_cast<uint32_t>(model_config.attn_config.size_per_head) * 2;
    const uint32_t indexer_entry_elems = fp8_kv ? 132 : static_cast<uint32_t>(model_config.attn_config.indexer_head_dim) * 2;
    const uint32_t head_dim            = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    const uint32_t indexer_head_dim    = static_cast<uint32_t>(model_config.attn_config.indexer_head_dim);

    auto csa_kv = makeDsv4Spec("csa_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 4);
    auto hca_kv = makeDsv4Spec("hca_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 128);
    auto indexer_kv = makeDsv4Spec("indexer_kv", "compressed_kv", indexer_entry_elems, DataType::TYPE_UINT8, 4);
    auto indexer_state = makeDsv4Spec("indexer_state", "fixed_state", 4 * indexer_head_dim, DataType::TYPE_FP32);
    auto csa_state = makeDsv4Spec("csa_state", "fixed_state", 4 * head_dim, DataType::TYPE_FP32);
    auto hca_state = makeDsv4Spec("hca_state", "fixed_state", 2 * head_dim, DataType::TYPE_FP32);
    auto swa_kv = makeDsv4Spec("swa_kv", "sliding_window_kv", kv_entry_elems, DataType::TYPE_UINT8);

    model_config.kv_cache_specs.clear();
    model_config.kv_cache_spec_descs.clear();
    for (int i = 0; i < layer_num; ++i) {
        const int ratio = i < static_cast<int>(model_config.attn_config.layer_compress_ratios.size()) ?
                              model_config.attn_config.layer_compress_ratios[static_cast<size_t>(i)] :
                              0;
        if (ratio == 4) {
            model_config.kv_cache_specs[i] = {csa_kv, indexer_kv, indexer_state, csa_state, swa_kv};
        } else if (ratio == 128) {
            model_config.kv_cache_specs[i] = {hca_kv, hca_state, swa_kv};
        } else {
            model_config.kv_cache_specs[i] = {swa_kv};
        }
        auto& descs = model_config.kv_cache_spec_descs[i];
        descs.reserve(model_config.kv_cache_specs[i].size());
        for (const auto& spec : model_config.kv_cache_specs[i]) {
            descs.push_back(dsv4DescForSpec(spec));
        }
    }
}

inline void refreshDsv4KvCacheSpecDescs(ModelConfig&             model_config,
                                        const ParallelismConfig& parallelism_config,
                                        const KVCacheConfig&     kv_cache_config,
                                        int                      gen_num_per_cycle = 0) {
    (void)parallelism_config;
    (void)kv_cache_config;
    (void)gen_num_per_cycle;
    setDsv4KvCacheSpecs(model_config);
}

inline void setDsv4ExplicitPoolBlocks(ModelConfig& model_config, const std::string& tag, uint32_t block_num) {
    for (auto& [_, descs] : model_config.kv_cache_spec_descs) {
        for (auto& desc : descs) {
            if (desc.tag == tag) {
                desc.extra.explicit_block_num = block_num;
            }
        }
    }
}

// A tiny helper for unit tests to construct a minimal MultiHeadAttention KV cache config.
//
// NOTE:
// - This is test-only code. Keep it small and dependency-light.
// - The returned CacheConfig is fully initialized (no uninitialized fundamental fields).
inline CacheConfig makeSimpleMhaCacheConfig(int               layer_num,
                                            int               block_num,
                                            size_t            tokens_per_block,
                                            rtp_llm::DataType dtype,
                                            uint32_t          local_head_num_kv = 1,
                                            uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype                     = dtype;
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheSpecType::MultiHeadAttention;
    spec->dtype              = dtype;
    spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    spec->local_head_num_kv  = local_head_num_kv;
    spec->size_per_head      = size_per_head;
    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * spec->block_size_bytes();

    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride       = static_cast<size_t>(spec->local_head_num_kv) * tokens_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        config.kv_scale_stride_bytes          = 2 * kv_scale_kv_stride_bytes;
        config.kv_scale_size_bytes            = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

// A tiny helper for unit tests to construct a minimal "hybrid attention" KV cache config where `groupNums() > 1`.
//
// Notes:
// - This is a multi-group config that will trigger
//   HybridTypeKVCacheAllocator path (KVCacheManager selects hybrid allocator when groupNums()>1).
// - Groups are mixed: the first group is LinearAttention, the remaining groups are MultiHeadAttention.
// - For correctness with current hybrid allocator/memory layout, require `layer_num % group_layer_num == 0` and
//   `layer_num / group_layer_num >= 2` (at least 2 groups).
inline CacheConfig makeSimpleHybridMhaCacheConfig(int               layer_num,
                                                  int               block_num,
                                                  size_t            tokens_per_block,
                                                  rtp_llm::DataType dtype,
                                                  int               group_layer_num   = 2,
                                                  uint32_t          local_head_num_kv = 1,
                                                  uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype              = dtype;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = static_cast<uint32_t>(block_num);
    config.seq_size_per_block = tokens_per_block;
    config.group_layer_num    = std::max(group_layer_num, 1);
    config.linear_step        = 2;

    // If the split is not even or cannot form >=2 groups, fall back to a single-group MHA config (keeps config valid).
    // Tests that need `groupNums()>1` should pass `layer_num % group_layer_num == 0` and
    // `layer_num/group_layer_num>=2`.
    if (layer_num <= 0 || (layer_num % config.group_layer_num) != 0 || (layer_num / config.group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt     = layer_num / config.group_layer_num;

    // Specs.
    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = dtype;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = dtype;
    full_spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    full_spec->local_head_num_kv  = local_head_num_kv;
    full_spec->size_per_head      = size_per_head;

    std::vector<KVCacheSpecPtr>    specs;
    std::vector<std::vector<int>>  layers_by_group;
    std::vector<CacheGroupType>    types;
    std::vector<std::string>       tags;
    specs.reserve(static_cast<size_t>(group_cnt));
    layers_by_group.reserve(static_cast<size_t>(group_cnt));
    types.reserve(static_cast<size_t>(group_cnt));
    tags.reserve(static_cast<size_t>(group_cnt));

    // Build groups: gid=0 linear, gid>=1 full.
    for (int gid = 0; gid < group_cnt; ++gid) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(config.group_layer_num));
        for (int local = 0; local < config.group_layer_num; ++local) {
            const int layer_id = gid * config.group_layer_num + local;
            group_layers.push_back(layer_id);
        }
        layers_by_group.push_back(group_layers);

        if (gid == 0) {
            specs.push_back(linear_spec);
            types.push_back(CacheGroupType::LINEAR);
        } else {
            specs.push_back(full_spec);
            types.push_back(CacheGroupType::FULL);
        }
        tags.push_back("default");
    }
    config.fromGroupedSpecs(specs, layers_by_group, types, tags);

    // Physical sizes for hybrid memory layout: one group (group_layer_num) worth of layers.
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

}  // namespace rtp_llm::test
