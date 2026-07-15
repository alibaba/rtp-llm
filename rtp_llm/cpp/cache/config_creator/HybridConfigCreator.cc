#include "rtp_llm/cpp/cache/config_creator/HybridConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/MemoryEvaluationHelper.h"

namespace rtp_llm {

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

KVCacheSpecPtr HybridConfigCreator::getSpecFromLayers(const LayerKVCacheSpecs& runtime_specs,
                                                      const std::vector<int>&  layer_ids,
                                                      const char*              spec_role) {
    if (layer_ids.empty()) {
        return nullptr;
    }
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (int layer_id : layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(layer_id) < runtime_specs.size()
                                    && !runtime_specs[static_cast<size_t>(layer_id)].empty(),
                                "missing runtime kv_cache specs for %s layer %d",
                                spec_role,
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(runtime_specs[static_cast<size_t>(layer_id)].size() == 1,
                                "%s layer %d must have exactly one runtime kv_cache spec, got %zu",
                                spec_role,
                                layer_id,
                                runtime_specs[static_cast<size_t>(layer_id)].size());
        const auto& spec = runtime_specs[static_cast<size_t>(layer_id)][0];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "%s layer %d has null kv_cache spec", spec_role, layer_id);
        if (result == nullptr) {
            result      = spec;
            fingerprint = spec->fingerprint();
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                fingerprint == spec->fingerprint(), "%s layers have different kv_cache spec fingerprints", spec_role);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "no %s layers found", spec_role);
    return result->clone();
}

void HybridConfigCreator::prepareFullAttentionSpec(KVCacheSpecPtr           spec,
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

void HybridConfigCreator::prepareLinearAttentionSpec(KVCacheSpecPtr           spec,
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
    const int tp                   = std::max(1, static_cast<int>(parallelism_config.get_attn_tp_size()));
    linear_spec->local_num_k_heads = static_cast<uint32_t>(linear_config.linear_num_key_heads / tp);
    linear_spec->local_num_v_heads = static_cast<uint32_t>(linear_config.linear_num_value_heads / tp);
    RTP_LLM_CHECK_WITH_INFO(linear_spec->local_num_k_heads > 0 && linear_spec->local_num_v_heads > 0,
                            "invalid local heads for linear attention: k=%d v=%d tp=%d",
                            linear_spec->local_num_k_heads,
                            linear_spec->local_num_v_heads,
                            tp);
    spec->local_head_num_kv = static_cast<uint32_t>(
        std::max(1,
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
    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));

    auto append_group = [&](const KVCacheSpecPtr& spec, CacheGroupType type, const std::vector<int>& layer_ids) {
        GroupBase group;
        group.spec      = spec;
        group.policy    = defaultCacheGroupPolicy(type);
        group.layer_ids = layer_ids;
        const int gid   = static_cast<int>(groups.size());
        groups.push_back(group);
        for (int layer_id : layer_ids) {
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(gid);
            layer.tag_to_gid[spec->tag] = gid;
        }
    };

    // Keep order: all full groups first, then linear groups.
    for (const auto& g : full_groups) {
        append_group(full_spec, CacheGroupType::FULL, g);
    }
    for (const auto& g : linear_groups) {
        append_group(linear_spec, CacheGroupType::LINEAR, g);
    }
    config.setTopology(std::move(groups), std::move(layers));
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig&          config,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec) {
    // Decide the physical KV block/scale sizes by taking max between full and linear specs.
    // Either spec may be nullptr when the model has no layers of that type.
    RTP_LLM_CHECK_WITH_INFO(full_spec || linear_spec, "both full_spec and linear_spec are null");
    const size_t full_kv_block_stride_bytes   = full_spec ? full_spec->block_size_bytes() : 0;
    const size_t linear_kv_block_stride_bytes = linear_spec ? linear_spec->block_size_bytes() : 0;

    // now we only support that linear attention block have padding
    if (full_spec && linear_spec) {
        RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                                "not support full attention with padding now");
    }

    config.kv_block_stride_bytes = std::max(full_kv_block_stride_bytes, linear_kv_block_stride_bytes);
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes =
        full_spec ? full_spec->scale_block_size_bytes() : (linear_spec ? linear_spec->scale_block_size_bytes() : 0);
    config.kv_scale_size_bytes = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes    = config.kv_block_size_bytes + config.kv_scale_size_bytes;
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    const KVCacheConfig&     kv_cache_config,
                                                    bool                     is_mtp,
                                                    int                      gen_num_per_cycle) {
    (void)is_mtp;
    auto       dtype                     = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto physical_tokens_per_block = kv_cache_config.seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                               static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block   = kv_cache_config.kernel_seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                               physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "hybrid seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "hybrid kernel_seq_size_per_block must be > 0");
    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = physical_tokens_per_block;
    ctx.attn_tp_size            = static_cast<uint32_t>(parallelism_config.get_attn_tp_size());
    ctx.kernel_tokens_per_block = kernel_tokens_per_block;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    // Split layers by attention type
    auto [linear_layers, full_layers] = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config        = HybridConfigCreator::initializeConfig(model_config, linear_layers, full_layers, dtype);
    config.seq_size_per_block = physical_tokens_per_block;
    config.kernel_seq_size_per_block = kernel_tokens_per_block;

    auto full_spec   = HybridConfigCreator::getSpecFromLayers(runtime_specs, full_layers, "full attention");
    auto linear_spec = HybridConfigCreator::getSpecFromLayers(runtime_specs, linear_layers, "linear attention");

    // Create layer groups and calculate group layer number
    int group_layer_num = 0;
    auto [linear_groups, full_groups] =
        HybridConfigCreator::createLayerGroups(linear_layers, full_layers, group_layer_num);
    config.group_layer_num = group_layer_num;

    if (full_spec) {
        HybridConfigCreator::prepareFullAttentionSpec(
            full_spec, model_config, parallelism_config, dtype, static_cast<uint32_t>(full_layers.size()));
    }
    if (linear_spec) {
        HybridConfigCreator::prepareLinearAttentionSpec(
            linear_spec, model_config, parallelism_config, dtype, static_cast<uint32_t>(linear_layers.size()));
    }

    // Setup cache config specs
    HybridConfigCreator::setupCacheConfigSpecs(config, linear_groups, full_groups, linear_spec, full_spec);

    // Setup physical sizes
    HybridConfigCreator::setupPhysicalSizes(config, full_spec, linear_spec);

    // Per-layer block stride (kv + scale).
    // For hybrid attention, the physical per-layer stride follows the selected physical layout stride.
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

}  // namespace rtp_llm
