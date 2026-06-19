#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/models/dsv4/Dsv4CachePlanBuilder.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

bool hasDsv4KvCacheSpecs(const ModelConfig& model_config) {
    constexpr const char* kExpectedTags[] = {
        "csa_kv", "hca_kv", "indexer_kv", "indexer_state", "csa_state", "hca_state", "swa_kv"};
    for (const auto& layer_specs : model_config.kv_cache_specs) {
        for (const auto& spec : layer_specs.second) {
            if (spec == nullptr) {
                continue;
            }
            for (const char* expected_tag : kExpectedTags) {
                if (spec->tag == expected_tag) {
                    return true;
                }
            }
        }
    }
    return false;
}

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

void appendGroup(CacheConfig&            config,
                 const std::vector<int>& layer_ids,
                 CacheGroupType          group_type,
                 KVCacheSpecPtr          spec,
                 KVCacheRegionName       region_name = KVCacheRegionName::DEFAULT,
                 std::string             tag = "") {
    if (layer_ids.empty()) {
        return;
    }
    if (tag.empty() && spec != nullptr) {
        tag = spec->tag;
    }
    config.global_layer_ids.push_back(layer_ids);
    config.layer_ids.push_back(layer_ids);
    config.cache_specs.push_back(spec);
    config.group_types.push_back(group_type);
    config.group_region_names.push_back(region_name);
    config.group_policies.push_back(cacheGroupPolicyForLegacyRegion(group_type, region_name));
    config.group_tags.push_back(std::move(tag));
}

size_t kernelBlocksPerKvBlockForGroup(const CacheConfig& config, size_t group_id) {
    RTP_LLM_CHECK_WITH_INFO(group_id < config.group_types.size(),
                            "missing cache group type for group %zu (group_types.size=%zu)",
                            group_id,
                            config.group_types.size());
    const bool is_full = config.group_types[group_id] == CacheGroupType::FULL;
    return is_full ? config.kernelBlocksPerKvBlock() : 1;
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto group_num               = static_cast<size_t>(config.groupNums());
    config.group_block_nums.resize(group_num, 0);
    config.group_seq_size_per_block.resize(group_num, config.seq_size_per_block);
    config.group_kv_block_stride_bytes.resize(group_num, 0);
    config.group_kv_scale_stride_bytes.resize(group_num, 0);
    config.group_block_size_bytes.resize(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        const auto& spec = config.cache_specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count                = static_cast<uint32_t>(config.global_layer_ids[gid].size());
        const size_t kernel_kv_stride           = spec->block_size_bytes();
        const auto   kernel_scale               = spec->scale_block_size_bytes();
        const size_t group_bpk                  = kernelBlocksPerKvBlockForGroup(config, gid);
        const size_t kv_stride                  = kernel_kv_stride * group_bpk;
        const size_t scale_stride               = kernel_scale * group_bpk;
        config.group_kv_block_stride_bytes[gid] = kv_stride;
        config.group_kv_scale_stride_bytes[gid] = scale_stride;
        config.group_block_size_bytes[gid]      = static_cast<size_t>(layer_count) * (kv_stride + scale_stride);
        const auto type     = gid < config.group_types.size() ? config.group_types[gid] : spec->lifecycle;
        const bool is_state = spec->is_state_cache;
        if (!is_state && type == CacheGroupType::FULL) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);

        for (int layer_id : config.global_layer_ids[gid]) {
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
}

void populateHybridAttentionGroups(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config) {
    const auto dtype  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto layers = splitHybridPoolLayers(model_config);

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_policies.clear();
    config.group_region_names.clear();
    config.group_tags.clear();

    auto full_spec   = getHybridSpecByTag(model_config, "full");
    auto swa_spec    = full_spec->clone();
    auto linear_spec = getHybridSpecByTag(model_config, "linear");
    swa_spec->tag    = "swa";
    prepareFullAttentionSpec(full_spec, model_config, parallelism_config, dtype);
    prepareFullAttentionSpec(swa_spec, model_config, parallelism_config, dtype);
    prepareLinearAttentionSpec(linear_spec, model_config, parallelism_config, dtype);

    appendGroup(config, layers.full_layers, CacheGroupType::FULL, full_spec);
    appendGroup(config, layers.swa_layers, CacheGroupType::SWA, swa_spec);
    appendGroup(config, layers.linear_layers, CacheGroupType::LINEAR, linear_spec);
}

void setupGroupCounts(CacheConfig& config) {
    config.full_group_num   = 0;
    config.swa_group_num    = 0;
    config.linear_group_num = 0;
    for (auto group_type : config.group_types) {
        if (group_type == CacheGroupType::FULL) {
            ++config.full_group_num;
        } else if (group_type == CacheGroupType::SWA) {
            ++config.swa_group_num;
        } else {
            ++config.linear_group_num;
        }
    }
}

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            const KVCacheConfig&     kv_cache_config,
                                            bool                     is_mtp,
                                            int                      gen_num_per_cycle) {
    const auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    RTP_LLM_CHECK_WITH_INFO(model_config.attn_config.layer_compress_ratios.empty() || hasDsv4KvCacheSpecs(model_config),
                            "DSV4 cache config requires model_config.kv_cache_specs; "
                            "layer_compress_ratios fallback is disabled");

    if (hasDsv4KvCacheSpecs(model_config)) {
        Dsv4CachePlanBuilder::applyConfig(
            config, model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    } else {
        RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.enable_hybrid_attention,
                                "HybridPoolConfigCreator requires DSV4 kv_cache_specs or hybrid attention");
        populateHybridAttentionGroups(config, model_config, parallelism_config);
    }

    RTP_LLM_CHECK_WITH_INFO(!config.cache_specs.empty(), "hybrid-pool config produced no cache specs");
    setupGroupCounts(config);
    auto specs           = config.cache_specs;
    auto layers_by_group = config.layer_ids;
    auto types           = config.group_types;
    auto regions         = config.group_region_names;
    auto tags            = config.group_tags;
    auto policies        = config.group_policies;
    config.fromGroupedSpecs(specs, layers_by_group, types, regions, tags);
    if (policies.size() == config.group_policies.size()) {
        config.group_policies = std::move(policies);
    }
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
