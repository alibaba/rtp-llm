#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/config_creator/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

struct HybridPoolLayers {
    std::vector<int> full_layers;
    std::vector<int> linear_layers;
    std::vector<int> swa_layers;
};

HybridPoolLayers splitHybridPoolLayers(const ModelConfig& model_config) {
    const auto layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);
    RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.hybrid_attention_types.size()
                                == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            model_config.hybrid_attention_config.hybrid_attention_types.size(),
                            layer_num);

    HybridPoolLayers layers;
    layers.full_layers.reserve(static_cast<size_t>(layer_num));
    layers.linear_layers.reserve(static_cast<size_t>(layer_num));
    layers.swa_layers.reserve(static_cast<size_t>(layer_num));
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        switch (model_config.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                layers.linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                layers.swa_layers.push_back(i);
                break;
            case HybridAttentionType::NONE:
            default:
                layers.full_layers.push_back(i);
                break;
        }
    }
    return layers;
}

KVCacheSpecPtr getHybridSpecByTag(const ModelConfig& model_config, const std::string& tag) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (const auto& layer_specs : model_config.kv_cache_specs) {
        for (const auto& spec : layer_specs.second) {
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool kv_cache_specs must not contain null specs");
            RTP_LLM_CHECK_WITH_INFO(!spec->tag.empty(), "hybrid-pool kv_cache_specs must not contain empty tags");
            if (spec->tag == tag) {
                const auto current_fingerprint = spec->fingerprint();
                if (result == nullptr) {
                    result      = spec;
                    fingerprint = current_fingerprint;
                } else {
                    RTP_LLM_CHECK_WITH_INFO(fingerprint == current_fingerprint,
                                            "duplicate hybrid-pool kv_cache spec tag=%s has different prototype",
                                            tag.c_str());
                }
            }
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "missing hybrid-pool kv_cache spec tag=%s", tag.c_str());
    return result->clone();
}

LayerKVCacheSpecs layerSpecsFromDescs(const ModelConfig& model_config, rtp_llm::DataType dtype) {
    SpecBuildContext ctx;
    ctx.dtype = dtype;
    ctx.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

    LayerKVCacheSpecs layer_specs;
    for (const auto& [layer_id, layer_descs] : model_config.kv_cache_spec_descs) {
        auto& specs = layer_specs[layer_id];
        specs.reserve(layer_descs.size());
        for (auto desc : layer_descs) {
            desc.dtype = desc.dtype == DataType::TYPE_INVALID ? dtype : desc.dtype;
            desc.seq_size_per_block = desc.seq_size_per_block == 0 ? ctx.seq_size_per_block : desc.seq_size_per_block;
            specs.push_back(SpecBuilder::build(desc, ctx));
        }
    }
    return layer_specs;
}

ModelConfig modelConfigWithDescSpecs(const ModelConfig& model_config, rtp_llm::DataType dtype) {
    if (model_config.kv_cache_spec_descs.empty()) {
        return model_config;
    }
    auto model_copy = model_config;
    model_copy.kv_cache_specs = layerSpecsFromDescs(model_config, dtype);
    return model_copy;
}

uint32_t alignUpToMultiple(uint32_t value, uint32_t multiple) {
    RTP_LLM_CHECK_WITH_INFO(multiple > 0, "align multiple must be > 0");
    return ((value + multiple - 1) / multiple) * multiple;
}

uint32_t fixedRegionCpSize(const ParallelismConfig& parallelism_config) {
    if (!parallelism_config.prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    if (parallelism_config.role_type == RoleType::PREFILL && parallelism_config.tp_size > 1) {
        return static_cast<uint32_t>(parallelism_config.tp_size);
    }
    if (parallelism_config.role_type == RoleType::DECODE && parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        RTP_LLM_CHECK_WITH_INFO(
            parallelism_config.prefill_cp_config.prefill_cp_size > 1,
            "fixed/SWA CP sharding decode requires explicit prefill_cp_size when PREFILL_CP and kv_cache_sharded are enabled");
        return static_cast<uint32_t>(parallelism_config.prefill_cp_config.prefill_cp_size);
    }
    return 1;
}

bool isPrefillCpSliced(const ParallelismConfig& parallelism_config) {
    return parallelism_config.role_type == RoleType::PREFILL && fixedRegionCpSize(parallelism_config) > 1;
}

uint32_t computeStateRingEntries(const KVCacheSpecDesc& desc, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(desc.extra.state_ring_compression_ratio > 0,
                            "state ring desc tag=%s requires positive state_ring_compression_ratio",
                            desc.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "state ring desc tag=%s requires non-negative gen_num_per_cycle, got %d",
                            desc.tag.c_str(),
                            gen_num_per_cycle);
    const uint32_t window =
        (1 + desc.extra.state_ring_overlap) * desc.extra.state_ring_compression_ratio;
    const uint32_t raw =
        window + (desc.extra.state_ring_add_gen_num_per_cycle ? static_cast<uint32_t>(gen_num_per_cycle) : 0);
    return (raw + 1) & ~1U;
}

size_t cpPrefillSliceBlockBytes(const KVCacheSpecDesc&  desc,
                                uint32_t                entries_per_block,
                                const ParallelismConfig& parallelism_config) {
    const auto cp_size = fixedRegionCpSize(parallelism_config);
    if (cp_size <= 1 || !isPrefillCpSliced(parallelism_config) || !desc.extra.cp_prefill_slice_block_bytes) {
        return desc.block_size_bytes_override;
    }
    const size_t natural_bytes = static_cast<size_t>(entries_per_block) * desc.entry_elems * getTypeSize(desc.store_dtype);
    const size_t align =
        desc.block_size_bytes_alignment > 0 ?
            std::lcm(desc.block_size_bytes_alignment, static_cast<size_t>(cp_size)) :
            static_cast<size_t>(cp_size);
    const size_t full_stride_bytes = ((natural_bytes + align - 1) / align) * align;
    RTP_LLM_CHECK_WITH_INFO(full_stride_bytes % cp_size == 0,
                            "CP prefill byte slicing tag=%s full stride %zu must be divisible by cp_size %u",
                            desc.tag.c_str(),
                            full_stride_bytes,
                            cp_size);
    return full_stride_bytes / cp_size;
}

uint32_t localKvHeads(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    return static_cast<uint32_t>(
        (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
            model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
            model_config.attn_config.kv_head_num
                / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
}

LayerKVCacheSpecDescs prepareHybridPoolDescs(const ModelConfig&       model_config,
                                             const ParallelismConfig& parallelism_config,
                                             rtp_llm::DataType        dtype,
                                             uint32_t                 physical_tokens_per_block,
                                             uint32_t                 kernel_tokens_per_block,
                                             int                      gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    const auto cp_size        = fixedRegionCpSize(parallelism_config);
    const bool prefill_sliced = isPrefillCpSliced(parallelism_config);

    auto descs = model_config.kv_cache_spec_descs;
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        auto it = descs.find(layer_id);
        RTP_LLM_CHECK_WITH_INFO(it != descs.end(),
                                "hybrid-pool desc config missing kv_cache_spec_descs for layer %ld",
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(!it->second.empty(),
                                "hybrid-pool desc config layer %ld has no descs",
                                layer_id);
        for (auto& desc : it->second) {
            desc.dtype = desc.dtype == DataType::TYPE_INVALID ? dtype : desc.dtype;
            if (desc.cache_type == CacheType::MHA && desc.local_head_num_kv == 0) {
                desc.local_head_num_kv = localKvHeads(model_config, parallelism_config);
            } else if (desc.cache_type == CacheType::MLA && desc.local_head_num_kv == 0) {
                desc.local_head_num_kv = 1;
            } else if (desc.cache_type == CacheType::LINEAR) {
                const auto& linear_config = model_config.linear_attention_config;
                const int tp = std::max(1, static_cast<int>(parallelism_config.get_attn_tp_size()));
                if (desc.local_num_k_heads == 0) {
                    desc.local_num_k_heads = static_cast<uint32_t>(linear_config.linear_num_key_heads / tp);
                }
                if (desc.local_num_v_heads == 0) {
                    desc.local_num_v_heads = static_cast<uint32_t>(linear_config.linear_num_value_heads / tp);
                }
                if (desc.local_head_num_kv == 0) {
                    desc.local_head_num_kv = static_cast<uint32_t>(std::max(
                        1,
                        (linear_config.linear_num_value_heads > 1) ?
                            static_cast<int>(linear_config.linear_num_value_heads / parallelism_config.get_attn_tp_size()) :
                            static_cast<int>(linear_config.linear_num_value_heads)));
                }
            }

            if (desc.extra.derive_entries_from_kernel_block) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_tokens_per_block);
                desc.entries_per_block = kernel_tokens_per_block / desc.compression_ratio;
            }

            if (desc.extra.state_ring_compression_ratio > 0) {
                uint32_t entries = computeStateRingEntries(desc, gen_num_per_cycle);
                if (cp_size > 1 && (desc.extra.cp_align_entries || desc.extra.cp_slice_entries)) {
                    entries = alignUpToMultiple(entries, cp_size);
                    if (desc.extra.cp_slice_entries && prefill_sliced) {
                        entries /= cp_size;
                    }
                }
                desc.entries_per_block = entries;
                desc.block_size_bytes_override = cpPrefillSliceBlockBytes(desc, entries, parallelism_config);
            }

            if (desc.extra.use_fixed_region_cp_tokens && cp_size > 1) {
                desc.seq_size_per_block = physical_tokens_per_block * cp_size;
            } else {
                desc.seq_size_per_block =
                    desc.seq_size_per_block == 0 ? physical_tokens_per_block : desc.seq_size_per_block;
            }

            if (desc.cache_type == CacheType::COMPRESSED_KV) {
                desc.has_sparse_slots   = true;
            }
        }
    }
    return descs;
}

void prepareFullAttentionSpec(KVCacheSpecPtr            spec,
                              const ModelConfig&       model_config,
                              const ParallelismConfig& parallelism_config,
                              rtp_llm::DataType        dtype) {
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto* mla_spec = dynamic_cast<MLAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mla_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadLatentAttention,
                                "full kv_cache spec must be MLAKVCacheSpec for MLA model");
        // local_head_num_kv is already set to 1 by Python-side MLAKVCacheSpec default.
        // kv_lora_rank, rope_head_dim, seq_size_per_block are already populated by Python.
    } else {
        auto* mha_spec = dynamic_cast<MHAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mha_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadAttention,
                                "full kv_cache spec must be MHAKVCacheSpec for MHA/GQA model");
        // local_head_num_kv depends on TP and cannot be provided by Python-side spec.
        spec->local_head_num_kv = static_cast<uint32_t>(
            (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                model_config.attn_config.kv_head_num
                    / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        // size_per_head, seq_size_per_block are already populated by Python.
    }
    // dtype depends on runtime quantization config and cannot be provided by Python-side spec.
    spec->dtype = dtype;
}

void prepareLinearAttentionSpec(KVCacheSpecPtr            spec,
                                const ModelConfig&       model_config,
                                const ParallelismConfig& parallelism_config,
                                rtp_llm::DataType        dtype) {
    auto* linear_spec = dynamic_cast<LinearKVCacheSpec*>(spec.get());
    RTP_LLM_CHECK_WITH_INFO(linear_spec != nullptr && spec->type == KVCacheSpecType::LinearAttention,
                            "linear kv_cache spec must be LinearKVCacheSpec");
    const auto& linear_config = model_config.linear_attention_config;
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim > 0 && linear_config.linear_value_head_dim > 0,
                            "invalid linear head dim");
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_conv_kernel_dim > 1,
                            "invalid linear_conv_kernel_dim=%d",
                            linear_config.linear_conv_kernel_dim);
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_num_key_heads > 0 && linear_config.linear_num_value_heads > 0,
                            "invalid linear heads");
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim == linear_config.linear_value_head_dim,
                            "linear head dims must match (current impl): k=%d v=%d",
                            linear_config.linear_key_head_dim,
                            linear_config.linear_value_head_dim);
    // local_num_k_heads, local_num_v_heads, and local_head_num_kv depend on TP
    // and cannot be provided by Python-side spec.
    const int tp = std::max(1, static_cast<int>(parallelism_config.get_attn_tp_size()));
    linear_spec->local_num_k_heads = static_cast<uint32_t>(linear_config.linear_num_key_heads / tp);
    linear_spec->local_num_v_heads = static_cast<uint32_t>(linear_config.linear_num_value_heads / tp);
    RTP_LLM_CHECK_WITH_INFO(linear_spec->local_num_k_heads > 0 && linear_spec->local_num_v_heads > 0,
                            "invalid local heads for linear attention: k=%d v=%d tp=%d",
                            linear_spec->local_num_k_heads,
                            linear_spec->local_num_v_heads,
                            tp);
    spec->local_head_num_kv = static_cast<uint32_t>(std::max(
        1,
        (linear_config.linear_num_value_heads > 1) ?
            static_cast<int>(linear_config.linear_num_value_heads / parallelism_config.get_attn_tp_size()) :
            static_cast<int>(linear_config.linear_num_value_heads)));
    // dtype depends on runtime quantization config and cannot be provided by Python-side spec.
    spec->dtype = dtype;
    // seq_size_per_block, head_k_dim, head_v_dim, conv_kernel_dim,
    // ssm_state_dtype, conv_state_dtype are already populated by Python.
}

void appendGroup(std::vector<KVCacheSpecPtr>&    specs,
                 std::vector<std::vector<int>>&  layers_by_group,
                 std::vector<CacheGroupType>&    types,
                 std::vector<CacheGroupPolicy>&  policies,
                 std::vector<std::string>&       tags,
                 const std::vector<int>&         layer_ids,
                 CacheGroupType                  group_type,
                 KVCacheSpecPtr                  spec,
                 std::string                     tag = "") {
    if (layer_ids.empty()) {
        return;
    }
    if (tag.empty() && spec != nullptr) {
        tag = spec->tag;
    }
    specs.push_back(spec);
    layers_by_group.push_back(layer_ids);
    types.push_back(group_type);
    policies.push_back(CacheConfig::cacheGroupPolicyForSpec(spec, group_type));
    tags.push_back(std::move(tag));
}

size_t kernelBlocksPerKvBlockForGroup(const CacheConfig& config, size_t group_id) {
    const bool is_full = config.typeForGroup(group_id) == CacheGroupType::FULL;
    return is_full ? config.kernelBlocksPerKvBlock() : 1;
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto group_num               = static_cast<size_t>(config.groupNums());
    std::vector<uint32_t> group_block_nums(group_num, 0);
    config.group_seq_size_per_block.resize(group_num, config.seq_size_per_block);
    std::vector<size_t> group_kv_block_stride_bytes(group_num, 0);
    std::vector<size_t> group_kv_scale_stride_bytes(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = config.specForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count                = static_cast<uint32_t>(config.layerIdsForGroup(gid).size());
        const size_t kernel_kv_stride           = spec->block_size_bytes();
        const auto   kernel_scale               = spec->scale_block_size_bytes();
        const size_t group_bpk                  = kernelBlocksPerKvBlockForGroup(config, gid);
        const size_t kv_stride                  = kernel_kv_stride * group_bpk;
        const size_t scale_stride               = kernel_scale * group_bpk;
        group_kv_block_stride_bytes[gid]        = kv_stride;
        group_kv_scale_stride_bytes[gid]        = scale_stride;
        const auto type     = config.typeForGroup(gid);
        const bool is_state = spec->is_state_cache;
        if (!is_state && type == CacheGroupType::FULL) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);

        for (int layer_id : config.layerIdsForGroup(gid)) {
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] =
                static_cast<int>(kv_stride + scale_stride);
        }
    }

    config.group_layer_num         = static_cast<int>(std::max<uint32_t>(1, max_group_layers));
    config.kv_block_stride_bytes   = max_kv_stride;
    config.kv_scale_stride_bytes   = max_scale_stride;
    config.kv_block_size_bytes     = total_kv_block_bytes;
    config.kv_scale_size_bytes     = total_scale_block_bytes;
    const size_t paged_block_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    if (paged_block_bytes == 0) {
        RTP_LLM_CHECK_WITH_INFO(is_mtp && config.use_typed_cache_regions,
                                "hybrid-pool paged groups produced zero block bytes");
        config.kv_block_size_bytes = 1;
        config.kv_scale_size_bytes = 0;
        config.block_size_bytes    = 1;
    } else {
        config.block_size_bytes = paged_block_bytes;
    }
    config.explicitly_sized_pool_reserve_bytes = 0;
    config.setGroupBlockLayout(group_block_nums, group_kv_block_stride_bytes, group_kv_scale_stride_bytes);
}

void populateHybridAttentionGroups(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config) {
    const auto dtype  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto layers = splitHybridPoolLayers(model_config);

    auto full_spec   = getHybridSpecByTag(model_config, "full");
    auto swa_spec    = full_spec->clone();
    auto linear_spec = getHybridSpecByTag(model_config, "linear");
    swa_spec->tag    = "swa";
    prepareFullAttentionSpec(full_spec, model_config, parallelism_config, dtype);
    prepareFullAttentionSpec(swa_spec, model_config, parallelism_config, dtype);
    prepareLinearAttentionSpec(linear_spec, model_config, parallelism_config, dtype);

    std::vector<KVCacheSpecPtr>    specs;
    std::vector<std::vector<int>>  layers_by_group;
    std::vector<CacheGroupType>    types;
    std::vector<CacheGroupPolicy>  policies;
    std::vector<std::string>       tags;

    appendGroup(specs, layers_by_group, types, policies, tags, layers.full_layers, CacheGroupType::FULL, full_spec);
    appendGroup(specs, layers_by_group, types, policies, tags, layers.swa_layers, CacheGroupType::SWA, swa_spec);
    appendGroup(
        specs, layers_by_group, types, policies, tags, layers.linear_layers, CacheGroupType::LINEAR, linear_spec);

    config.fromGroupedSpecs(specs, layers_by_group, types, tags);
    config.setGroupPolicies(policies);
}

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            const KVCacheConfig&     kv_cache_config,
                                            bool                     is_mtp,
                                            int                      gen_num_per_cycle) {
    const auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto physical_tokens_per_block =
        kv_cache_config.seq_size_per_block > 0 ? static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                                 static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block =
        kv_cache_config.kernel_seq_size_per_block > 0 ? static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                                        physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "hybrid-pool seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "hybrid-pool kernel_seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block >= kernel_tokens_per_block
                                && physical_tokens_per_block % kernel_tokens_per_block == 0,
                            "hybrid-pool seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = physical_tokens_per_block;
    config.kernel_seq_size_per_block = kernel_tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    if (!model_config.kv_cache_spec_descs.empty()) {
        auto descs = prepareHybridPoolDescs(
            model_config, parallelism_config, dtype, physical_tokens_per_block, kernel_tokens_per_block, gen_num_per_cycle);
        SpecBuildContext ctx;
        ctx.dtype              = dtype;
        ctx.seq_size_per_block = physical_tokens_per_block;
        config.fromLayerDescs(descs, ctx);
        config.group_seq_size_per_block.resize(static_cast<size_t>(config.groupNums()), config.seq_size_per_block);
        for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
            const auto& spec = config.specForGroup(gid);
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool desc config produced null spec gid=%zu", gid);
            config.group_seq_size_per_block[gid] = spec->seq_size_per_block;
            config.use_typed_cache_regions =
                config.use_typed_cache_regions || spec->type == KVCacheSpecType::OpaqueKV
                || spec->type == KVCacheSpecType::OpaqueState;
            config.use_opaque_kv_cache_store =
                config.use_opaque_kv_cache_store || spec->type == KVCacheSpecType::OpaqueKV
                || spec->type == KVCacheSpecType::OpaqueState;
        }
        for (const auto& [_, layer_descs] : descs) {
            for (const auto& desc : layer_descs) {
                config.is_sparse = config.is_sparse || desc.cache_type == CacheType::COMPRESSED_KV;
            }
        }
        config.disable_decode_first_malloc_device_reuse =
            config.disable_decode_first_malloc_device_reuse || config.use_opaque_kv_cache_store;
    } else {
        RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.enable_hybrid_attention,
                                "HybridPoolConfigCreator requires kv_cache_spec_descs or hybrid attention");
        const auto spec_model_config = modelConfigWithDescSpecs(model_config, dtype);
        populateHybridAttentionGroups(config, spec_model_config, parallelism_config);
    }

    RTP_LLM_CHECK_WITH_INFO(config.groupNums() > 0, "hybrid-pool config produced no cache specs");
    setupIndependentPoolSizes(config, is_mtp);
    return config;
}

}  // namespace

CacheConfig HybridPoolConfigCreator::createConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    return createHybridAttentionPoolConfig(
        model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
}

}  // namespace rtp_llm
