#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct MTPModuleConfigPlan {
    std::vector<size_t>      source_layer_indices;
    std::vector<ModelConfig> module_configs;
};

inline ModelConfig makeSingleLayerMTPModelConfig(const ModelConfig& model_config, size_t source_layer) {
    RTP_LLM_CHECK_WITH_INFO(model_config.num_layers > 0,
                            "MTP model config must have positive num_layers, got %ld",
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(source_layer < static_cast<size_t>(model_config.num_layers),
                            "MTP source layer %zu is out of range [0, %ld)",
                            source_layer,
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(source_layer < model_config.kv_cache_spec_descs.size(),
                            "MTP source layer %zu has no kv cache descriptor row (row count %zu)",
                            source_layer,
                            model_config.kv_cache_spec_descs.size());
    RTP_LLM_CHECK_WITH_INFO(!model_config.kv_cache_spec_descs[source_layer].empty(),
                            "MTP source layer %zu has no kv cache descriptors",
                            source_layer);

    ModelConfig single_layer_config         = model_config;
    single_layer_config.num_layers          = 1;
    single_layer_config.kv_cache_spec_descs = {model_config.kv_cache_spec_descs[source_layer]};

    const auto& attention_types = model_config.hybrid_attention_config.hybrid_attention_types;
    if (model_config.hybrid_attention_config.enable_hybrid_attention || !attention_types.empty()) {
        RTP_LLM_CHECK_WITH_INFO(source_layer < attention_types.size(),
                                "MTP source layer %zu has no hybrid attention type (type count %zu)",
                                source_layer,
                                attention_types.size());
        single_layer_config.hybrid_attention_config.hybrid_attention_types = {attention_types[source_layer]};
    }
    return single_layer_config;
}

inline void validateActiveMTPCacheLayout(const ModelConfig& active_module_config) {
    const auto& expected = active_module_config;
    RTP_LLM_CHECK_WITH_INFO(
        expected.num_layers == 1, "MTP module 0 must be a one-layer config, got %ld layers", expected.num_layers);
    RTP_LLM_CHECK_WITH_INFO(expected.kv_cache_spec_descs.size() == 1,
                            "MTP module 0 must have one descriptor row, got %zu",
                            expected.kv_cache_spec_descs.size());
    RTP_LLM_CHECK_WITH_INFO(!expected.kv_cache_spec_descs[0].empty(),
                            "MTP module 0 must have at least one cache descriptor");
}

inline MTPModuleConfigPlan buildMTPModuleConfigPlan(const ModelConfig& model_config,
                                                    size_t             weight_count,
                                                    size_t             gen_num_per_cycle,
                                                    SpeculativeType    sp_type) {
    RTP_LLM_CHECK_WITH_INFO(weight_count > 0, "MTP module config plan requires at least one layer weight");

    size_t model_num = weight_count;
    if (gen_num_per_cycle > 1 && weight_count == 1) {
        model_num = gen_num_per_cycle;
    } else if (gen_num_per_cycle != weight_count) {
        model_num = std::min(weight_count, gen_num_per_cycle);
    }
    if (sp_type == SP_TYPE_EAGLE || sp_type == SP_TYPE_EAGLE3) {
        model_num = 1;
    }
    RTP_LLM_CHECK_WITH_INFO(model_num > 0,
                            "MTP module config plan produced no modules: weights=%zu gen_num_per_cycle=%zu",
                            weight_count,
                            gen_num_per_cycle);

    MTPModuleConfigPlan plan;
    plan.source_layer_indices.reserve(model_num);
    plan.module_configs.reserve(model_num);
    // Runtime execution and cache loading currently use module 0 only. Keep a
    // physical slot and its own weight for every configured module, but make
    // inactive slots share module 0's cache layout so heterogeneous checkpoint
    // metadata cannot change allocation or trigger an unused-module fail-fast.
    const auto active_module_config = makeSingleLayerMTPModelConfig(model_config, /*source_layer=*/0);
    validateActiveMTPCacheLayout(active_module_config);
    for (size_t module_index = 0; module_index < model_num; ++module_index) {
        const size_t source_layer = weight_count == 1 ? 0 : module_index;
        plan.source_layer_indices.push_back(source_layer);
        plan.module_configs.push_back(active_module_config);
    }
    return plan;
}

}  // namespace rtp_llm
