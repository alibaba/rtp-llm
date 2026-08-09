#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

void validateHybridPoolDescs(const ModelConfig& model_config, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
        "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
        model_config.kv_cache_spec_descs.size(),
        model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "hybrid-pool desc config requires non-negative gen_num_per_cycle, got %d",
                            gen_num_per_cycle);

    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& layer_descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!layer_descs.empty(), "hybrid-pool desc config layer %ld has no descs", layer_id);
        for (const auto& desc : layer_descs) {
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                RTP_LLM_CHECK_WITH_INFO(
                    desc.compression_ratio > 0,
                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                    desc.tag.c_str(),
                    desc.compression_ratio);
            }
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "state ring desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

std::pair<std::vector<GroupBase>, std::vector<LayerBase>>
buildGroupsFromLayerSpecs(const LayerKVCacheSpecBuildResults& layer_specs,
                          const ModelConfig&                  model_config,
                          const ParallelismConfig&            parallelism_config) {
    const auto layer_num = static_cast<uint32_t>(model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(layer_num),
                            "hybrid-pool layer spec count %zu != layer_num %u",
                            layer_specs.size(),
                            layer_num);

    struct GroupBuildState {
        KVCacheSpecPtr   spec;
        std::string      fingerprint;
        CacheGroupPolicy policy;
        uint32_t         local_kv_head_num = 1;
        std::vector<int> layer_ids;
    };

    std::map<std::string, GroupBuildState> group_by_tag;
    std::vector<std::string>               ordered_tags;

    for (uint32_t layer = 0; layer < layer_num; ++layer) {
        const auto& specs = layer_specs[layer];
        RTP_LLM_CHECK_WITH_INFO(!specs.empty(), "hybrid-pool layer %u has no specs", layer);
        std::set<std::string> layer_tags;
        for (const auto& [spec, policy] : specs) {
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool layer %u has null spec", layer);
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(spec->tag).second,
                                    "hybrid-pool layer %u has duplicate tag=%s",
                                    layer,
                                    spec->tag.c_str());
            const auto local_kv_head_num = resolveLocalKVHeadNum(
                spec->type, model_config.attn_config, model_config.linear_attention_config, parallelism_config);
            auto group_it = group_by_tag.find(spec->tag);
            if (group_it == group_by_tag.end()) {
                GroupBuildState state;
                state.spec              = spec;
                state.fingerprint       = spec->layoutFingerprint();
                state.policy            = policy;
                state.local_kv_head_num = local_kv_head_num;
                group_it                = group_by_tag.emplace(spec->tag, std::move(state)).first;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(group_it->second.fingerprint == spec->layoutFingerprint(),
                                        "hybrid-pool tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group_it->second.policy, policy),
                                        "hybrid-pool tag=%s has inconsistent policy",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group_it->second.local_kv_head_num == local_kv_head_num,
                                        "hybrid-pool tag=%s has inconsistent local_kv_head_num",
                                        spec->tag.c_str());
            }
            group_it->second.layer_ids.push_back(static_cast<int>(layer));
        }
    }

    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers(static_cast<size_t>(layer_num));
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto& state = group_by_tag.at(tag);
        GroupBase   group;
        group.tag               = tag;
        group.spec              = state.spec;
        group.policy            = state.policy;
        group.layer_ids         = state.layer_ids;
        group.local_kv_head_num = state.local_kv_head_num;
        groups.push_back(group);
        for (int layer_id : state.layer_ids) {
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(tag);
        }
    }
    return {std::move(groups), std::move(layers)};
}

void setupIndependentPoolSizes(CacheConfig& config, std::vector<GroupBase> groups, std::vector<LayerBase> layers) {
    for (auto& group : groups) {
        const auto& spec = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec tag=%s is null", group.tag.c_str());
        const size_t kv_stride      = spec->block_size_bytes();
        const size_t scale_stride   = spec->scale_block_size_bytes();
        group.block_num             = 0;
        group.kv_block_stride_bytes = kv_stride;
        group.kv_scale_stride_bytes = scale_stride;
    }
    config.setTopology(std::move(groups), std::move(layers));
}

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            int                      gen_num_per_cycle) {
    const auto dtype                     = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto physical_tokens_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "hybrid-pool seq_size_per_block must be > 0");

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.seq_size_per_block = physical_tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    if (!model_config.kv_cache_spec_descs.empty()) {
        validateHybridPoolDescs(model_config, gen_num_per_cycle);
        SpecBuildContext ctx;
        ctx.dtype                   = dtype;
        ctx.seq_size_per_block      = physical_tokens_per_block;
        ctx.attn_config             = &model_config.attn_config;
        ctx.linear_attention_config = &model_config.linear_attention_config;
        ctx.parallelism_config      = &parallelism_config;
        ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
        auto refreshed_specs        = CacheConfigCreator::buildLayerSpecsFromDescs(
            model_config.kv_cache_spec_descs, ctx, model_config.num_layers);
        auto [groups, layers] = buildGroupsFromLayerSpecs(refreshed_specs, model_config, parallelism_config);
        setupIndependentPoolSizes(config, std::move(groups), std::move(layers));
        for (const auto& layer_descs : model_config.kv_cache_spec_descs) {
            for (const auto& desc : layer_descs) {
                config.is_sparse = config.is_sparse || desc.cache_type == KVCacheSpecType::OpaqueKV;
            }
        }
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "HybridPoolConfigCreator requires kv_cache_spec_descs");
    }

    RTP_LLM_CHECK_WITH_INFO(config.groupNums() > 0, "hybrid-pool config produced no cache specs");
    return config;
}

}  // namespace

CacheConfig HybridPoolConfigCreator::createConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool /*is_mtp*/,
                                                  int gen_num_per_cycle) {
    return createHybridAttentionPoolConfig(model_config, parallelism_config, gen_num_per_cycle);
}

}  // namespace rtp_llm
