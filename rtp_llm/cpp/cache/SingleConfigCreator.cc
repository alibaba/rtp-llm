#include "rtp_llm/cpp/cache/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <numeric>

namespace rtp_llm {

namespace {

KVCacheSpecBuildResult getDefaultSpecFromRuntimeSpecs(const ModelConfig&                  model_config,
                                                      const LayerKVCacheSpecBuildResults& runtime_specs) {
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "single cache config requires layer-wise runtime specs for every layer, got %zu/%ld",
                            runtime_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(!runtime_specs.empty(), "single cache config requires at least one runtime spec");
    RTP_LLM_CHECK_WITH_INFO(runtime_specs[0].size() == 1,
                            "single cache config requires exactly one spec for layer 0, got %zu",
                            runtime_specs[0].size());
    const auto& [spec, policy] = runtime_specs[0][0];
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "single cache config got null runtime spec for layer 0");
    const auto& expected_tag = spec->tag;
    const auto  fingerprint  = spec->layoutFingerprint();
    for (int64_t layer_id = 1; layer_id < model_config.num_layers; ++layer_id) {
        const auto layer = static_cast<size_t>(layer_id);
        RTP_LLM_CHECK_WITH_INFO(runtime_specs[layer].size() == 1,
                                "single cache config requires exactly one spec for layer %ld, got %zu",
                                layer_id,
                                runtime_specs[layer].size());
        const auto& [layer_spec, layer_policy] = runtime_specs[layer][0];
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec != nullptr, "single cache config got null runtime spec for layer %ld", layer_id);
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec->tag == expected_tag,
            "single cache config requires consistent tag across layers, layer %ld has tag=%s but layer 0 has tag=%s",
            layer_id,
            layer_spec->tag.c_str(),
            expected_tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec->layoutFingerprint() == fingerprint, "single cache config spec differs at layer %ld", layer_id);
        RTP_LLM_CHECK_WITH_INFO(
            CacheConfig::samePolicy(layer_policy, policy), "single cache config policy differs at layer %ld", layer_id);
    }
    return {spec->clone(), policy};
}

}  // namespace

CacheConfig SingleConfigCreator::createSingleConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp,
                                                    int                      gen_num_per_cycle) {
    (void)is_mtp;

    auto       dtype            = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto tokens_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(tokens_per_block > 0, "single seq_size_per_block must be > 0");

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    auto layer_num = model_config.num_layers;

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.seq_size_per_block = tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.is_sparse          = model_config.attn_config.is_sparse;

    auto [spec, policy] = getDefaultSpecFromRuntimeSpecs(model_config, runtime_specs);

    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    GroupBase group;
    group.tag               = spec->tag;
    group.spec              = spec;
    group.policy            = policy;
    group.layer_ids         = layer_ids;
    group.local_kv_head_num = resolveLocalKVHeadNum(
        spec->type, model_config.attn_config, model_config.linear_attention_config, parallelism_config);

    // Using spec interface for block size and scale
    group.kv_block_stride_bytes = spec->block_size_bytes();

    // scale_block_size_bytes() returns 0 when scales are not used.
    group.kv_scale_stride_bytes = spec->scale_block_size_bytes();

    std::vector<LayerBase> layers(static_cast<size_t>(layer_num));
    for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        auto& layer      = layers[static_cast<size_t>(layer_id)];
        layer.layer_id   = static_cast<int>(layer_id);
        layer.group_tags = {spec->tag};
    }
    config.setTopology({group}, std::move(layers));
    RTP_LLM_CHECK_WITH_INFO(config.groupNums() == 1, "single config expected one cache group");

    return config;
}

}  // namespace rtp_llm
