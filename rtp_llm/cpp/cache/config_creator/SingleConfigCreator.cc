#include "rtp_llm/cpp/cache/config_creator/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <numeric>

namespace rtp_llm {

namespace {

KVCacheSpecPtr getDefaultSpecFromRuntimeSpecs(const ModelConfig&        model_config,
                                              const LayerKVCacheSpecs& runtime_specs) {
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "single cache config requires layer-wise runtime specs for every layer, got %zu/%ld",
                            runtime_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(!runtime_specs.empty(), "single cache config requires at least one runtime spec");
    RTP_LLM_CHECK_WITH_INFO(runtime_specs[0].size() == 1,
                            "single cache config requires exactly one spec for layer 0, got %zu",
                            runtime_specs[0].size());
    auto spec = runtime_specs[0][0];
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "single cache config got null runtime spec for layer 0");
    RTP_LLM_CHECK_WITH_INFO(spec->tag == "default",
                            "single cache config requires tag=default for layer 0, got=%s",
                            spec->tag.c_str());
    const auto fingerprint = spec->fingerprint();
    for (int64_t layer_id = 1; layer_id < model_config.num_layers; ++layer_id) {
        const auto layer = static_cast<size_t>(layer_id);
        RTP_LLM_CHECK_WITH_INFO(runtime_specs[layer].size() == 1,
                                "single cache config requires exactly one spec for layer %ld, got %zu",
                                layer_id,
                                runtime_specs[layer].size());
        const auto& layer_spec = runtime_specs[layer][0];
        RTP_LLM_CHECK_WITH_INFO(layer_spec != nullptr, "single cache config got null runtime spec for layer %ld", layer_id);
        RTP_LLM_CHECK_WITH_INFO(layer_spec->tag == "default",
                                "single cache config requires tag=default for layer %ld, got=%s",
                                layer_id,
                                layer_spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(layer_spec->fingerprint() == fingerprint,
                                "single cache config default spec differs at layer %ld",
                                layer_id);
    }
    return spec->clone();
}

}  // namespace

CacheConfig SingleConfigCreator::createSingleConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    (void)is_mtp;
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    ctx.attn_tp_size       = static_cast<uint32_t>(parallelism_config.get_attn_tp_size());
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    auto layer_num = model_config.num_layers;

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

    config.use_mla   = model_config.attn_config.use_mla;
    config.dtype     = dtype;
    config.is_sparse = model_config.attn_config.is_sparse;

    auto spec = getDefaultSpecFromRuntimeSpecs(model_config, runtime_specs);

    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    GroupBase group;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids = layer_ids;

    std::vector<LayerBase> layers(static_cast<size_t>(layer_num));
    for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        auto& layer                  = layers[static_cast<size_t>(layer_id)];
        layer.group_ids              = {0};
        layer.tag_to_gid[spec->tag]  = 0;
        layer.legacy_single_group_id = 0;
    }
    config.setTopology({group}, std::move(layers));
    RTP_LLM_CHECK_WITH_INFO(config.groupNums() == 1, "single config expected one cache group");

    // Using spec interface for block size and scale
    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;

    // Scale handling - no need to check dtype as scale_block_size_bytes() returns 0 if no scale support
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;

    if (config.is_sparse) {
        auto indexer_dim             = model_config.attn_config.indexer_head_dim;
        config.kv_scale_stride_bytes = (indexer_dim + indexer_dim / 128 * 4) * spec->seq_size_per_block;
        config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.group_layer_num  = layer_num;  // only 1 group for SingleConfig

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

}  // namespace rtp_llm
