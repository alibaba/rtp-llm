#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

const KVCacheSpecBuildResult& singleRuntimeSpecForLayer(const LayerKVCacheSpecBuildResults& runtime_specs,
                                                        int                                 layer_id) {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < runtime_specs.size(),
                            "missing runtime kv_cache specs for layer %d",
                            layer_id);
    const auto& layer_specs = runtime_specs[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == 1,
                            "hybrid layer %d must have exactly one runtime kv_cache spec, got %zu",
                            layer_id,
                            layer_specs.size());
    RTP_LLM_CHECK_WITH_INFO(layer_specs[0].first != nullptr, "hybrid layer %d has null kv_cache spec", layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        !layer_specs[0].first->tag.empty(), "hybrid layer %d has empty kv_cache spec tag", layer_id);
    return layer_specs[0];
}

std::vector<GroupBase> buildGroups(const LayerKVCacheSpecBuildResults& runtime_specs,
                                   const ModelConfig&                  model_config,
                                   const ParallelismConfig&            parallelism_config) {
    const int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(layer_num),
                            "runtime kv_cache specs size %zu != num_layers %ld",
                            runtime_specs.size(),
                            layer_num);
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            layer_num);

    std::vector<GroupBase> groups;
    for (int layer_id = 0; layer_id < static_cast<int>(layer_num); ++layer_id) {
        const auto& build_result  = singleRuntimeSpecForLayer(runtime_specs, layer_id);
        const auto& spec          = build_result.first;
        const auto& policy        = build_result.second;
        const auto  group_type    = policy.group_type;
        const auto  expected_type = types[static_cast<size_t>(layer_id)] == HybridAttentionType::LINEAR ?
                                        CacheGroupType::LINEAR :
                                        CacheGroupType::FULL;
        RTP_LLM_CHECK_WITH_INFO(group_type == expected_type,
                                "hybrid layer %d desc tag=%s cache type %d does not match attention type %d",
                                layer_id,
                                spec->tag.c_str(),
                                static_cast<int>(group_type),
                                static_cast<int>(expected_type));

        auto it = std::find_if(
            groups.begin(), groups.end(), [&](const GroupBase& group) { return group.spec->tag == spec->tag; });
        if (it == groups.end()) {
            GroupBase group;
            group.tag               = spec->tag;
            group.spec              = spec;
            group.policy            = policy;
            group.local_kv_head_num = resolveLocalKVHeadNum(
                spec->type, model_config.attn_config, model_config.linear_attention_config, parallelism_config);
            group.layer_ids.push_back(layer_id);
            groups.push_back(std::move(group));
            continue;
        }

        RTP_LLM_CHECK_WITH_INFO(
            it->policy.group_type == group_type, "hybrid tag=%s maps to multiple cache group types", spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(it->spec->layoutFingerprint() == spec->layoutFingerprint(),
                                "hybrid tag=%s maps to different kv cache spec layouts",
                                spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(it->policy, policy),
                                "hybrid tag=%s maps to different cache group policies",
                                spec->tag.c_str());
        it->layer_ids.push_back(layer_id);
    }

    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "hybrid config requires at least one cache group");
    const auto full_group_num = std::count_if(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::FULL;
    });
    RTP_LLM_CHECK_WITH_INFO(
        full_group_num <= 1,
        "multiple full attention cache groups (%zu) are not supported: FMHA parameters bind one block table before "
        "the layer loop",
        static_cast<size_t>(full_group_num));
    return groups;
}

KVCacheSpecPtr representativeSpec(const std::vector<GroupBase>& groups, CacheGroupType group_type) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (const auto& group : groups) {
        if (group.policy.group_type != group_type) {
            continue;
        }
        if (result == nullptr) {
            result      = group.spec->clone();
            fingerprint = group.spec->layoutFingerprint();
        } else {
            RTP_LLM_CHECK_WITH_INFO(fingerprint == group.spec->layoutFingerprint(),
                                    "hybrid %d cache groups have different kv cache spec layouts",
                                    static_cast<int>(group_type));
        }
    }
    return result;
}

void setupTopologyFromGroups(CacheConfig& config, std::vector<GroupBase> groups) {
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
    }

    for (const auto& group : groups) {
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "hybrid tag=%s has invalid layer id %d",
                                    group.spec->tag.c_str(),
                                    layer_id);
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(group.tag);
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
}

}  // namespace

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp,
                                                    int                      gen_num_per_cycle) {
    (void)is_mtp;

    auto       dtype            = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto tokens_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(tokens_per_block > 0, "hybrid seq_size_per_block must be > 0");
    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.seq_size_per_block = tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.linear_step        = 1;

    auto cache_groups = buildGroups(runtime_specs, model_config, parallelism_config);
    auto full_spec    = representativeSpec(cache_groups, CacheGroupType::FULL);
    auto linear_spec  = representativeSpec(cache_groups, CacheGroupType::LINEAR);

    if (full_spec != nullptr && linear_spec != nullptr) {
        RTP_LLM_CHECK_WITH_INFO(full_spec->block_size_bytes() >= linear_spec->block_size_bytes(),
                                "not support full attention with padding now");
    }

    for (auto& group : cache_groups) {
        group.kv_block_stride_bytes = group.spec->block_size_bytes();
        group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
    }

    setupTopologyFromGroups(config, std::move(cache_groups));

    return config;
}

}  // namespace rtp_llm
