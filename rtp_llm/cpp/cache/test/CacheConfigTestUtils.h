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

inline KVCacheSpecDesc dsv4DescForSpec(const KVCacheSpecPtr& spec) {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "dsv4DescForSpec got null spec");
    KVCacheSpecDesc desc;
    desc.tag             = spec->tag;
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
    KVCacheSpecDesc desc;
    desc.tag                = "default";
    desc.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        desc.cache_type    = CacheType::MLA;
        desc.kv_lora_rank  = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        desc.rope_head_dim = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
        desc.num_kv_heads  = 1;
    } else {
        desc.cache_type    = CacheType::MHA;
        desc.size_per_head = static_cast<uint32_t>(model_config.attn_config.size_per_head);
        desc.num_kv_heads  = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
    }
    model_config.kv_cache_spec_descs.assign(static_cast<size_t>(model_config.num_layers), {desc});
}

inline void setHybridAttentionKvCacheSpecs(ModelConfig& model_config) {
    std::vector<int> full_layers;
    std::vector<int> swa_layers;
    std::vector<int> linear_layers;
    const auto&      types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    for (int i = 0; i < static_cast<int>(model_config.num_layers); ++i) {
        switch (types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                swa_layers.push_back(i);
                break;
            case HybridAttentionType::NONE:
            default:
                full_layers.push_back(i);
                break;
        }
    }

    KVCacheSpecDesc full_desc;
    full_desc.tag                = "full";
    full_desc.cache_type         = CacheType::MHA;
    full_desc.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    full_desc.size_per_head      = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    full_desc.num_kv_heads       = static_cast<uint32_t>(model_config.attn_config.kv_head_num);

    KVCacheSpecDesc swa_desc = full_desc;
    swa_desc.tag             = "swa";
    swa_desc.cache_type      = CacheType::FIXED_STATE;
    swa_desc.entry_elems     = static_cast<uint32_t>(model_config.attn_config.size_per_head)
                           * static_cast<uint32_t>(model_config.attn_config.kv_head_num) * 2;
    swa_desc.entries_per_block = static_cast<uint32_t>(model_config.attn_config.sliding_window > 0 ?
                                                           model_config.attn_config.sliding_window :
                                                           model_config.attn_config.tokens_per_block);
    swa_desc.store_dtype       = DataType::TYPE_FP16;

    const auto& linear_config           = model_config.linear_attention_config;
    KVCacheSpecDesc linear_desc;
    linear_desc.tag                    = "linear";
    linear_desc.cache_type             = CacheType::LINEAR;
    linear_desc.seq_size_per_block     = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    linear_desc.num_k_heads            = static_cast<uint32_t>(linear_config.linear_num_key_heads);
    linear_desc.num_v_heads            = static_cast<uint32_t>(linear_config.linear_num_value_heads);
    linear_desc.head_k_dim             = static_cast<uint32_t>(linear_config.linear_key_head_dim);
    linear_desc.head_v_dim             = static_cast<uint32_t>(linear_config.linear_value_head_dim);
    linear_desc.conv_kernel_dim        = static_cast<uint32_t>(linear_config.linear_conv_kernel_dim);
    linear_desc.ssm_state_dtype        = linear_config.ssm_state_dtype;
    linear_desc.conv_state_dtype       = linear_config.conv_state_dtype;

    model_config.kv_cache_spec_descs.assign(static_cast<size_t>(model_config.num_layers), {});
    for (int layer_id : full_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {full_desc};
    }
    for (int layer_id : swa_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {swa_desc};
    }
    for (int layer_id : linear_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {linear_desc};
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

    model_config.kv_cache_spec_descs.clear();
    model_config.kv_cache_spec_descs.resize(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        const int ratio = i < static_cast<int>(model_config.attn_config.layer_compress_ratios.size()) ?
                              model_config.attn_config.layer_compress_ratios[static_cast<size_t>(i)] :
                              0;
        std::vector<KVCacheSpecPtr> specs;
        if (ratio == 4) {
            specs = {csa_kv, indexer_kv, indexer_state, csa_state, swa_kv};
        } else if (ratio == 128) {
            specs = {hca_kv, hca_state, swa_kv};
        } else {
            specs = {swa_kv};
        }
        auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(i)];
        descs.reserve(specs.size());
        for (const auto& spec : specs) {
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
    for (auto& descs : model_config.kv_cache_spec_descs) {
        for (auto& desc : descs) {
            if (desc.tag == tag) {
                desc.extra.explicit_block_num = block_num;
            }
        }
    }
}

inline void setGroupedSpecs(CacheConfig&                         config,
                            const std::vector<KVCacheSpecPtr>&    specs,
                            const std::vector<std::vector<int>>& layers_by_group,
                            const std::vector<CacheGroupType>&   types,
                            const std::vector<std::string>&      tags = {}) {
    const size_t group_num = specs.size();
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "setGroupedSpecs requires at least one cache spec");
    RTP_LLM_CHECK_WITH_INFO(layers_by_group.size() == group_num,
                            "setGroupedSpecs layer group count %zu != spec count %zu",
                            layers_by_group.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(types.size() == group_num,
                            "setGroupedSpecs group type count %zu != spec count %zu",
                            types.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(tags.empty() || tags.size() == group_num,
                            "setGroupedSpecs tag count %zu != spec count %zu",
                            tags.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(config.layer_num > 0, "setGroupedSpecs requires positive layer_num");

    std::vector<GroupInfo> new_groups;
    std::vector<LayerInfo> new_layers(static_cast<size_t>(config.layer_num));
    new_groups.reserve(group_num);

    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "setGroupedSpecs got null spec at group %zu", gid);
        std::string tag = tags.empty() ? spec->tag : tags[gid];
        if (tag.empty() && group_num == 1) {
            tag = "default";
        }
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "setGroupedSpecs requires non-empty tag for cache spec %zu", gid);
        auto stored_spec = spec->clone();
        stored_spec->tag = tag;

        GroupInfo group;
        group.spec               = stored_spec;
        group.policy             = CacheConfig::cacheGroupPolicyForSpec(stored_spec, types[gid]);
        group.layer_ids          = layers_by_group[gid];
        group.seq_size_per_block = stored_spec->seq_size_per_block;
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "setGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                    tag.c_str(),
                                    layer_id,
                                    config.layer_num);
            auto& layer = new_layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(static_cast<int>(gid));
            const auto [it, inserted] = layer.tag_to_gid.emplace(tag, static_cast<int>(gid));
            RTP_LLM_CHECK_WITH_INFO(inserted || it->second == static_cast<int>(gid),
                                    "setGroupedSpecs layer %d tag %s maps to both group %d and %zu",
                                    layer_id,
                                    tag.c_str(),
                                    inserted ? static_cast<int>(gid) : it->second,
                                    gid);
        }
    }

    config.setTopology(std::move(new_groups), std::move(new_layers));
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
    setGroupedSpecs(config, {spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});

    size_t kv_scale_stride_bytes = 0;
    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride       = static_cast<size_t>(spec->local_head_num_kv) * tokens_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        kv_scale_stride_bytes                 = 2 * kv_scale_kv_stride_bytes;
    }

    config.setGroupBlockLayout({static_cast<uint32_t>(block_num)}, {spec->block_size_bytes()}, {kv_scale_stride_bytes});

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
    config.seq_size_per_block = tokens_per_block;
    config.linear_step        = 2;
    const int physical_group_layer_num = std::max(group_layer_num, 1);

    // If the split is not even or cannot form >=2 groups, fall back to a single-group MHA config (keeps config valid).
    // Tests that need `groupNums()>1` should pass `layer_num % group_layer_num == 0` and
    // `layer_num/group_layer_num>=2`.
    if (layer_num <= 0 || (layer_num % physical_group_layer_num) != 0
        || (layer_num / physical_group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt     = layer_num / physical_group_layer_num;

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
        group_layers.reserve(static_cast<size_t>(physical_group_layer_num));
        for (int local = 0; local < physical_group_layer_num; ++local) {
            const int layer_id = gid * physical_group_layer_num + local;
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
    setGroupedSpecs(config, specs, layers_by_group, types, tags);

    // Physical sizes for hybrid memory layout: one group worth of layers.
    const size_t kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    const size_t kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.setGroupBlockLayout(std::vector<uint32_t>(static_cast<size_t>(group_cnt), static_cast<uint32_t>(block_num)),
                               std::vector<size_t>(static_cast<size_t>(group_cnt), kv_block_stride_bytes),
                               std::vector<size_t>(static_cast<size_t>(group_cnt), kv_scale_stride_bytes));
    return config;
}

}  // namespace rtp_llm::test
