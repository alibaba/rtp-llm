#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

namespace rtp_llm {

namespace {

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

}  // namespace

std::vector<std::vector<int>> HybridConfigCreator::splitIntoGroups(const std::vector<int>& ids, int group_layer_num) {
    std::vector<std::vector<int>> groups;
    if (ids.empty()) {
        return groups;
    }
    const int n = static_cast<int>(ids.size());
    const int s = std::max(group_layer_num, 1);
    groups.reserve((n + s - 1) / s);
    for (int i = 0; i < n; i += s) {
        const int end = std::min(i + s, n);
        groups.emplace_back(ids.begin() + i, ids.begin() + end);
    }
    return groups;
}

int HybridConfigCreator::calculateGroupLayerNum(int linear_layer_count, int full_layer_count) {
    int group_layer_num = 0;
    if (linear_layer_count > 0 && full_layer_count > 0) {
        group_layer_num = std::gcd(linear_layer_count, full_layer_count);
    } else {
        group_layer_num = std::max(linear_layer_count, full_layer_count);
    }
    group_layer_num = std::max(group_layer_num, 1);
    return group_layer_num;
}

std::pair<std::vector<int>, std::vector<int>>
HybridConfigCreator::splitLayersByAttentionType(const ModelConfig& model_config) {
    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    std::vector<int> linear_layers;
    std::vector<int> full_layers;
    linear_layers.reserve(layer_num);
    full_layers.reserve(layer_num);

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            layer_num);
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        if (types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            linear_layers.push_back(i);
        } else {
            full_layers.push_back(i);
        }
    }

    return std::make_pair(std::move(linear_layers), std::move(full_layers));
}

CacheConfig HybridConfigCreator::initializeConfig(const ModelConfig&      model_config,
                                                  const std::vector<int>& linear_layers,
                                                  const std::vector<int>& full_layers,
                                                  rtp_llm::DataType       dtype) {
    int64_t layer_num = model_config.num_layers;

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;

    return config;
}

KVCacheSpecPtr HybridConfigCreator::getSpecFromLayers(const ModelConfig&      model_config,
                                                      const std::vector<int>& layer_ids,
                                                      const char*              spec_role) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (int layer_id : layer_ids) {
        const auto it = model_config.kv_cache_specs.find(layer_id);
        RTP_LLM_CHECK_WITH_INFO(it != model_config.kv_cache_specs.end() && !it->second.empty(),
                                "missing kv_cache_specs for %s layer %d",
                                spec_role,
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(it->second.size() == 1,
                                "%s layer %d must have exactly one kv_cache spec, got %zu",
                                spec_role,
                                layer_id,
                                it->second.size());
        const auto& spec = it->second[0];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "%s layer %d has null kv_cache spec", spec_role, layer_id);
        if (result == nullptr) {
            result      = spec;
            fingerprint = spec->fingerprint();
        } else {
            RTP_LLM_CHECK_WITH_INFO(fingerprint == spec->fingerprint(),
                                    "%s layers have different kv_cache spec fingerprints",
                                    spec_role);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "no %s layers found", spec_role);
    return result->clone();
}

void HybridConfigCreator::prepareFullAttentionSpec(KVCacheSpecPtr            spec,
                                                   const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   rtp_llm::DataType        dtype,
                                                   uint32_t                 layer_num) {
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

void HybridConfigCreator::prepareLinearAttentionSpec(KVCacheSpecPtr            spec,
                                                     const ModelConfig&       model_config,
                                                     const ParallelismConfig& parallelism_config,
                                                     rtp_llm::DataType        dtype,
                                                     uint32_t                 layer_num) {
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

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> HybridConfigCreator::createLayerGroups(
    const std::vector<int>& linear_layers, const std::vector<int>& full_layers, int& group_layer_num) {
    const int linear_cnt = static_cast<int>(linear_layers.size());
    const int full_cnt   = static_cast<int>(full_layers.size());
    group_layer_num      = HybridConfigCreator::calculateGroupLayerNum(linear_cnt, full_cnt);

    const auto linear_groups = HybridConfigCreator::splitIntoGroups(linear_layers, group_layer_num);
    const auto full_groups   = HybridConfigCreator::splitIntoGroups(full_layers, group_layer_num);

    return std::make_pair(std::move(linear_groups), std::move(full_groups));
}

void HybridConfigCreator::setupCacheConfigSpecs(CacheConfig&                         config,
                                                const std::vector<std::vector<int>>& linear_groups,
                                                const std::vector<std::vector<int>>& full_groups,
                                                const KVCacheSpecPtr&                linear_spec,
                                                const KVCacheSpecPtr&                full_spec) {
    std::vector<KVCacheSpecPtr>    specs;
    std::vector<std::vector<int>> layers_by_group;
    std::vector<CacheGroupType>   types;

    // Keep order: all full groups first, then linear groups.
    for (const auto& g : full_groups) {
        specs.push_back(full_spec);
        layers_by_group.push_back(g);
        types.push_back(CacheGroupType::FULL);
    }
    for (const auto& g : linear_groups) {
        specs.push_back(linear_spec);
        layers_by_group.push_back(g);
        types.push_back(CacheGroupType::LINEAR);
    }
    config.fromGroupedSpecs(specs, layers_by_group, types);
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig&          config,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec) {
    // Decide the physical KV block/scale sizes by taking max between full and linear specs.
    const size_t full_kv_block_stride_bytes   = full_spec->block_size_bytes();
    const size_t linear_kv_block_stride_bytes = linear_spec->block_size_bytes();

    // now we only support that linear attention block have padding
    RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                            "not support full attention with padding now");

    config.kv_block_stride_bytes = full_kv_block_stride_bytes;
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    auto spec_model_config = modelConfigWithDescSpecs(model_config, dtype);

    // Split layers by attention type
    auto [linear_layers, full_layers] = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config = HybridConfigCreator::initializeConfig(model_config, linear_layers, full_layers, dtype);

    auto full_spec   = HybridConfigCreator::getSpecFromLayers(spec_model_config, full_layers, "full attention");
    auto linear_spec = HybridConfigCreator::getSpecFromLayers(spec_model_config, linear_layers, "linear attention");

    // Create layer groups and calculate group layer number
    int group_layer_num = 0;
    auto [linear_groups, full_groups] =
        HybridConfigCreator::createLayerGroups(linear_layers, full_layers, group_layer_num);
    config.group_layer_num = group_layer_num;

    HybridConfigCreator::prepareFullAttentionSpec(
        full_spec, model_config, parallelism_config, dtype, static_cast<uint32_t>(full_layers.size()));
    HybridConfigCreator::prepareLinearAttentionSpec(
        linear_spec, model_config, parallelism_config, dtype, static_cast<uint32_t>(linear_layers.size()));

    // Setup cache config specs
    HybridConfigCreator::setupCacheConfigSpecs(config, linear_groups, full_groups, linear_spec, full_spec);

    // Setup physical sizes
    HybridConfigCreator::setupPhysicalSizes(config, full_spec, linear_spec);

    // fromGroupedSpecs populated layer/group mappings from the existing hybrid grouping policy.

    // Per-layer block stride (kv + scale).
    // For hybrid attention, the physical per-layer stride follows the selected physical layout stride.
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

}  // namespace rtp_llm
