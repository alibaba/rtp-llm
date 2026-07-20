#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
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

inline std::shared_ptr<MHAKVCacheSpec> makeResolvedMhaSpec(rtp_llm::DataType  dtype,
                                                           uint32_t           local_head_num_kv,
                                                           uint32_t           size_per_head,
                                                           uint32_t           seq_size_per_block,
                                                           const std::string& tag = "") {
    RTP_LLM_CHECK_WITH_INFO(local_head_num_kv > 0, "local_head_num_kv must be > 0");
    RTP_LLM_CHECK_WITH_INFO(size_per_head > 0, "size_per_head must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");

    AttentionConfigs attn{};
    attn.kv_head_num      = static_cast<int>(local_head_num_kv);
    attn.size_per_head    = static_cast<int>(size_per_head);
    attn.tokens_per_block = seq_size_per_block;
    ParallelismConfig parallelism;
    parallelism.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "default" : tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.attn_config             = &attn;
    ctx.parallelism_config      = &parallelism;
    ctx.kernel_tokens_per_block = seq_size_per_block;
    return std::dynamic_pointer_cast<MHAKVCacheSpec>(SpecBuilder::build(desc, ctx));
}

inline std::shared_ptr<MLAKVCacheSpec> makeResolvedMlaSpec(rtp_llm::DataType  dtype,
                                                           uint32_t           kv_lora_rank,
                                                           uint32_t           rope_head_dim,
                                                           uint32_t           seq_size_per_block,
                                                           const std::string& tag = "") {
    RTP_LLM_CHECK_WITH_INFO(kv_lora_rank > 0, "kv_lora_rank must be > 0");
    RTP_LLM_CHECK_WITH_INFO(rope_head_dim > 0, "rope_head_dim must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");

    AttentionConfigs attn{};
    attn.kv_lora_rank  = static_cast<int>(kv_lora_rank);
    attn.rope_head_dim = static_cast<int>(rope_head_dim);

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "mla" : tag;
    desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn;
    return std::dynamic_pointer_cast<MLAKVCacheSpec>(SpecBuilder::build(desc, ctx));
}

inline std::shared_ptr<LinearKVCacheSpec>
makeResolvedLinearSpec(rtp_llm::DataType  dtype,
                       uint32_t           local_num_k_heads,
                       uint32_t           local_num_v_heads,
                       uint32_t           head_k_dim,
                       uint32_t           head_v_dim,
                       uint32_t           conv_kernel_dim,
                       uint32_t           seq_size_per_block,
                       rtp_llm::DataType  ssm_state_dtype  = rtp_llm::DataType::TYPE_INVALID,
                       rtp_llm::DataType  conv_state_dtype = rtp_llm::DataType::TYPE_INVALID,
                       const std::string& tag              = "") {
    RTP_LLM_CHECK_WITH_INFO(local_num_k_heads > 0 && local_num_v_heads > 0, "linear head counts must be > 0");
    RTP_LLM_CHECK_WITH_INFO(head_k_dim > 0 && head_v_dim > 0, "linear head dims must be > 0");
    RTP_LLM_CHECK_WITH_INFO(conv_kernel_dim > 1, "conv_kernel_dim must be > 1");

    LinearAttentionConfig linear{};
    linear.linear_num_key_heads   = static_cast<int>(local_num_k_heads);
    linear.linear_num_value_heads = static_cast<int>(local_num_v_heads);
    linear.linear_key_head_dim    = static_cast<int>(head_k_dim);
    linear.linear_value_head_dim  = static_cast<int>(head_v_dim);
    linear.linear_conv_kernel_dim = static_cast<int>(conv_kernel_dim);
    linear.ssm_state_dtype        = ssm_state_dtype == rtp_llm::DataType::TYPE_INVALID ? dtype : ssm_state_dtype;
    linear.conv_state_dtype       = conv_state_dtype == rtp_llm::DataType::TYPE_INVALID ? dtype : conv_state_dtype;
    ParallelismConfig parallelism;
    parallelism.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "linear" : tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.linear_attention_config = &linear;
    ctx.parallelism_config      = &parallelism;
    ctx.kernel_tokens_per_block = seq_size_per_block;
    return std::dynamic_pointer_cast<LinearKVCacheSpec>(SpecBuilder::build(desc, ctx));
}

inline KVCacheSpecPtr makeResolvedOpaqueSpec(bool               state_cache,
                                             const std::string& tag,
                                             rtp_llm::DataType  dtype,
                                             size_t             block_bytes,
                                             uint32_t           seq_size_per_block) {
    const size_t dtype_size = getTypeSize(dtype);
    RTP_LLM_CHECK_WITH_INFO(dtype_size > 0, "invalid dtype=%d", static_cast<int>(dtype));
    RTP_LLM_CHECK_WITH_INFO(block_bytes % dtype_size == 0,
                            "opaque block_bytes=%zu must be divisible by dtype size=%zu",
                            block_bytes,
                            dtype_size);
    const auto block_elems = static_cast<uint32_t>(block_bytes / dtype_size);

    KVCacheSpecDesc desc;
    desc.tag                         = tag.empty() ? "opaque" : tag;
    desc.cache_type                  = state_cache ? KVCacheSpecType::OpaqueState : KVCacheSpecType::OpaqueKV;
    desc.dtype                       = dtype;
    desc.entry_dtype                 = dtype;
    desc.entry_elems                 = 1;
    desc.explicit_entry_count        = block_elems;
    desc.block_stride_bytes_override = block_bytes;
    desc.is_state_cache              = state_cache;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = seq_size_per_block;
    return SpecBuilder::build(desc, ctx);
}

inline KVCacheSpecDesc makeDsv4Desc(const std::string& tag,
                                    const std::string& kind,
                                    uint32_t           entry_elems,
                                    DataType           dtype,
                                    uint32_t           compression_ratio = 1) {
    KVCacheSpecDesc desc;
    desc.tag         = tag;
    desc.dtype       = dtype;
    desc.entry_elems = entry_elems;
    desc.entry_dtype = dtype;
    if (kind == "compressed_kv") {
        desc.cache_type        = KVCacheSpecType::OpaqueKV;
        desc.is_state_cache    = false;
        desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
        desc.compression_ratio = compression_ratio;
        if (desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
        return desc;
    }

    desc.cache_type          = KVCacheSpecType::OpaqueState;
    desc.is_state_cache      = true;
    desc.entry_count_mode    = OpaqueBlockEntryCountMode::STATE_RING;
    desc.reuse               = CacheReusePolicyDesc{};
    desc.reuse->evict_policy = CacheEvictPolicy::INDEPENDENT;
    desc.cp                  = CacheCpPolicyDesc{};
    if (desc.tag == "indexer_state" || desc.tag == "csa_state") {
        desc.compression_ratio        = 4;
        desc.state_ring_overlap       = 1;
        desc.cp->align_payload        = true;
        desc.cp->prefill_slice_layout = CpPrefillSliceLayout::PAYLOAD;
        desc.cp->slice                = CpBlockSliceMode::PAYLOAD_BYTES;
    } else if (desc.tag == "hca_state") {
        desc.compression_ratio                = 128;
        desc.cp->align_payload                = true;
        desc.cp->prefill_slice_layout         = CpPrefillSliceLayout::PAYLOAD;
        desc.cp->slice                        = CpBlockSliceMode::PAYLOAD_BYTES;
        desc.capacity                         = CacheCapacityPolicyDesc{};
        desc.capacity->explicit_block_num     = 256;
        desc.capacity->charge_to_paged_budget = true;
        desc.reuse->enable_prefix_reuse       = false;
        desc.tail                             = CacheTailPolicyDesc{};
        desc.tail->active_tail_blocks         = 1;
        desc.tail->validate_tail_blocks       = false;
    } else if (desc.tag == "swa_kv") {
        desc.compression_ratio        = DSV4_SWA_WINDOW_ENTRIES;
        desc.cp->align_payload        = true;
        desc.cp->prefill_slice_layout = CpPrefillSliceLayout::BLOCK_STRIDE;
        desc.cp->slice                = CpBlockSliceMode::EQUAL_BYTES;
        if (desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
    }
    desc.state_ring_include_gen_num_per_cycle = true;
    desc.cp->scale_seq_size                   = true;
    desc.block_stride_alignment_min_entries   = DSV4_SWA_WINDOW_ENTRIES;
    return desc;
}

inline void setDefaultKvCacheSpec(ModelConfig& model_config) {
    KVCacheSpecDesc desc;
    desc.tag = "default";
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    } else {
        desc.cache_type = KVCacheSpecType::MultiHeadAttention;
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
    full_desc.tag        = "full";
    full_desc.cache_type = KVCacheSpecType::MultiHeadAttention;

    KVCacheSpecDesc swa_desc = full_desc;
    swa_desc.tag             = "swa";
    swa_desc.cache_type      = KVCacheSpecType::OpaqueState;
    swa_desc.entry_elems     = static_cast<uint32_t>(model_config.attn_config.size_per_head)
                           * static_cast<uint32_t>(model_config.attn_config.kv_head_num) * 2;
    swa_desc.explicit_entry_count = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    swa_desc.entry_dtype = DataType::TYPE_FP16;

    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

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

inline void setDsv4KvCacheSpecs(ModelConfig& model_config, const std::vector<int>& layer_compress_ratios) {
    const int layer_num = static_cast<int>(model_config.num_layers);
    model_config.hybrid_attention_config.hybrid_attention_types.assign(static_cast<size_t>(layer_num),
                                                                       HybridAttentionType::NONE);

    const bool     fp8_kv = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8;
    const uint32_t kv_entry_elems =
        fp8_kv ? DSV4_FP8_KV_ENTRY_BYTES : static_cast<uint32_t>(model_config.attn_config.size_per_head) * 2;
    const uint32_t indexer_entry_elems =
        fp8_kv ? DSV4_FP8_INDEXER_ENTRY_BYTES : static_cast<uint32_t>(model_config.attn_config.indexer_head_dim) * 2;
    const uint32_t head_dim         = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    const uint32_t indexer_head_dim = static_cast<uint32_t>(model_config.attn_config.indexer_head_dim);

    auto csa_kv        = makeDsv4Desc("csa_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 4);
    auto hca_kv        = makeDsv4Desc("hca_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 128);
    auto indexer_kv    = makeDsv4Desc("indexer_kv", "compressed_kv", indexer_entry_elems, DataType::TYPE_UINT8, 4);
    auto indexer_state = makeDsv4Desc("indexer_state", "fixed_state", 4 * indexer_head_dim, DataType::TYPE_FP32);
    auto csa_state     = makeDsv4Desc("csa_state", "fixed_state", 4 * head_dim, DataType::TYPE_FP32);
    auto hca_state     = makeDsv4Desc("hca_state", "fixed_state", 2 * head_dim, DataType::TYPE_FP32);
    auto swa_kv        = makeDsv4Desc("swa_kv", "sliding_window_kv", kv_entry_elems, DataType::TYPE_UINT8);

    model_config.kv_cache_spec_descs.clear();
    model_config.kv_cache_spec_descs.resize(static_cast<size_t>(layer_num));
    for (int i = 0; i < layer_num; ++i) {
        const int ratio =
            i < static_cast<int>(layer_compress_ratios.size()) ? layer_compress_ratios[static_cast<size_t>(i)] : 0;
        if (ratio == 4) {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {
                csa_kv, indexer_kv, indexer_state, csa_state, swa_kv};
        } else if (ratio == 128) {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {hca_kv, hca_state, swa_kv};
        } else {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {swa_kv};
        }
    }
}

inline void setDsv4ExplicitPoolBlocks(ModelConfig& model_config, const std::string& tag, uint32_t block_num) {
    for (auto& descs : model_config.kv_cache_spec_descs) {
        for (auto& desc : descs) {
            if (desc.tag == tag) {
                if (!desc.capacity.has_value()) {
                    desc.capacity = CacheCapacityPolicyDesc{};
                }
                desc.capacity->explicit_block_num     = block_num;
                desc.capacity->charge_to_paged_budget = block_num > 0;
            }
        }
    }
}

inline KVCacheSpecPtr makeMhaSpec(const std::string& tag,
                                  size_t             tokens_per_block,
                                  rtp_llm::DataType  dtype,
                                  uint32_t           local_head_num_kv,
                                  uint32_t           size_per_head) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = local_head_num_kv;
    attn_config.size_per_head    = size_per_head;
    attn_config.tokens_per_block = static_cast<uint32_t>(tokens_per_block);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

inline KVCacheSpecPtr makeLinearSpec(const std::string& tag,
                                     size_t             tokens_per_block,
                                     rtp_llm::DataType  dtype,
                                     uint32_t           local_head_num_kv,
                                     uint32_t           size_per_head) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = static_cast<int>(size_per_head);
    linear_config.linear_value_head_dim  = static_cast<int>(size_per_head);
    linear_config.linear_num_key_heads   = static_cast<int>(local_head_num_kv);
    linear_config.linear_num_value_heads = static_cast<int>(local_head_num_kv);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = static_cast<uint32_t>(tokens_per_block);
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

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

    auto spec = makeMhaSpec("default", tokens_per_block, dtype, local_head_num_kv, size_per_head);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

inline CacheConfig makeSimpleLinearCacheConfig(int               layer_num,
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

    auto spec = makeLinearSpec("linear", tokens_per_block, dtype, local_head_num_kv, size_per_head);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::LINEAR}, {"linear"});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

inline CacheConfig makeSimpleHybridMhaCacheConfig(int               layer_num,
                                                  int               block_num,
                                                  size_t            tokens_per_block,
                                                  rtp_llm::DataType dtype,
                                                  int               group_layer_num   = 2,
                                                  uint32_t          local_head_num_kv = 1,
                                                  uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype                     = dtype;
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;
    config.group_layer_num           = std::max(group_layer_num, 1);
    config.linear_step               = 2;

    if (layer_num <= 0 || (layer_num % config.group_layer_num) != 0 || (layer_num / config.group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt = layer_num / config.group_layer_num;

    auto linear_spec = makeLinearSpec("linear", tokens_per_block, dtype, local_head_num_kv, size_per_head);
    auto full_spec   = makeMhaSpec("full", tokens_per_block, dtype, local_head_num_kv, size_per_head);

    std::vector<KVCacheSpecPtr>   specs;
    std::vector<std::vector<int>> layers_by_group;
    std::vector<CacheGroupType>   types;
    std::vector<std::string>      tags;
    specs.reserve(static_cast<size_t>(group_cnt));
    layers_by_group.reserve(static_cast<size_t>(group_cnt));
    types.reserve(static_cast<size_t>(group_cnt));
    tags.reserve(static_cast<size_t>(group_cnt));

    for (int gid = 0; gid < group_cnt; ++gid) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(config.group_layer_num));
        for (int local = 0; local < config.group_layer_num; ++local) {
            group_layers.push_back(gid * config.group_layer_num + local);
        }
        if (gid == 0) {
            specs.push_back(linear_spec);
            types.push_back(CacheGroupType::LINEAR);
            tags.push_back("linear");
        } else {
            specs.push_back(full_spec);
            types.push_back(CacheGroupType::FULL);
            tags.push_back("full" + std::to_string(gid));
        }
        layers_by_group.push_back(std::move(group_layers));
    }
    config.fromGroupedSpecs(specs, layers_by_group, types, tags);

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

}  // namespace rtp_llm::test
