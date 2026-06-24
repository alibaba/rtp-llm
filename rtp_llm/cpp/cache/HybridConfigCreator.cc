#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

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
    // All full attention layers must reside in one cache group (full_group_num <= 1).
    // prepare_fmha_impl binds the block table of group 0 once; it is not re-bound per layer.
    // group_layer_num must be >= full_layer_count to satisfy this.
    // When gcd is already sufficient it works directly; the fallback handles all other cases
    // (coprime gcd==1, or gcd>1 but still smaller than full_layer_count).
    int group_layer_num = 0;
    if (linear_layer_count > 0 && full_layer_count > 0) {
        group_layer_num = std::gcd(linear_layer_count, full_layer_count);
        // Fallback: when gcd < full_layer_count, force group_layer_num = full_layer_count
        // to guarantee all full layers fit in one group.
        // e.g. Kimi Linear 20:7 -> gcd=1 < 7 -> group_layer_num=7, linear groups=[7,7,6],
        // last group wastes 1 layer slot per block, negligible.
        if (group_layer_num < full_layer_count) {
            group_layer_num = full_layer_count;
        }
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

    config.global_layer_ids.push_back(linear_layers);
    config.global_layer_ids.push_back(full_layers);
    config.layer_ids.push_back(linear_layers);
    config.layer_ids.push_back(full_layers);

    return config;
}

KVCacheSpecPtr HybridConfigCreator::getSpecByTag(const ModelConfig& model_config, const std::string& tag) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (const auto& layer_specs : model_config.kv_cache_specs) {
        for (const auto& spec : layer_specs.second) {
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid kv_cache_specs must not contain null specs");
            RTP_LLM_CHECK_WITH_INFO(!spec->tag.empty(), "hybrid kv_cache_specs must not contain empty tags");
            if (spec->tag == tag) {
                const auto current_fingerprint = CacheConfig::specFingerprint(spec);
                if (result == nullptr) {
                    result      = spec;
                    fingerprint = current_fingerprint;
                } else {
                    RTP_LLM_CHECK_WITH_INFO(fingerprint == current_fingerprint,
                                            "duplicate hybrid kv_cache spec tag=%s has different prototype",
                                            tag.c_str());
                }
            }
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "missing hybrid kv_cache spec tag=%s", tag.c_str());
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
        spec->local_head_num_kv = 1;
        mla_spec->kv_lora_rank  = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        mla_spec->rope_head_dim = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
    } else {
        auto* mha_spec = dynamic_cast<MHAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mha_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadAttention,
                                "full kv_cache spec must be MHAKVCacheSpec for MHA/GQA model");
        spec->local_head_num_kv = static_cast<uint32_t>(
            (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                model_config.attn_config.kv_head_num
                    / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        mha_spec->size_per_head = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    }
    spec->dtype              = dtype;
    spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
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
    spec->dtype                   = dtype;
    spec->seq_size_per_block      = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    linear_spec->head_k_dim       = static_cast<uint32_t>(linear_config.linear_key_head_dim);
    linear_spec->head_v_dim       = static_cast<uint32_t>(linear_config.linear_value_head_dim);
    linear_spec->conv_kernel_dim  = static_cast<uint32_t>(linear_config.linear_conv_kernel_dim);
    linear_spec->ssm_state_dtype  = linear_config.ssm_state_dtype;
    linear_spec->conv_state_dtype = linear_config.conv_state_dtype;
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
    config.linear_group_num = static_cast<int>(linear_groups.size());
    config.full_group_num   = static_cast<int>(full_groups.size());
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

void HybridConfigCreator::setupLayerToGroupMapping(CacheConfig& config) {
    config.layer_to_group_id.assign(config.layer_num, 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            if (layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num) {
                config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int32_t>(gid);
            }
        }
    }
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    // Split layers by attention type
    auto [linear_layers, full_layers] = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config = HybridConfigCreator::initializeConfig(model_config, linear_layers, full_layers, dtype);

    auto full_spec   = HybridConfigCreator::getSpecByTag(model_config, "full");
    auto linear_spec = HybridConfigCreator::getSpecByTag(model_config, "linear");

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

    // Hard check: current only supports a single full attention group.
    RTP_LLM_CHECK_WITH_INFO(
        config.full_group_num <= 1,
        "Multiple full attention groups (%d) are not supported in hybrid mode. "
        "prepare_fmha_impl is called once before the layer loop, binding the block table from group 0. "
        "To support multiple full groups, implement per-group fmha preparation.",
        config.full_group_num);

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