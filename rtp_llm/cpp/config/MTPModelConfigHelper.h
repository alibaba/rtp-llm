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
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "MTP kv_cache_spec_descs size %zu != num_layers %ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(!model_config.kv_cache_spec_descs[source_layer].empty(),
                            "MTP source layer %zu has no kv cache descriptors",
                            source_layer);

    ModelConfig single_layer_config         = model_config;
    single_layer_config.num_layers          = 1;
    single_layer_config.kv_cache_spec_descs = {model_config.kv_cache_spec_descs[source_layer]};

    const auto& attention_types = model_config.hybrid_attention_config.hybrid_attention_types;
    if (model_config.hybrid_attention_config.enable_hybrid_attention || !attention_types.empty()) {
        RTP_LLM_CHECK_WITH_INFO(attention_types.size() == static_cast<size_t>(model_config.num_layers),
                                "MTP hybrid_attention_types size %zu != num_layers %ld",
                                attention_types.size(),
                                model_config.num_layers);
        single_layer_config.hybrid_attention_config.hybrid_attention_types = {attention_types[source_layer]};
    }
    return single_layer_config;
}

inline void validateHomogeneousMTPCacheLayouts(const std::vector<ModelConfig>& module_configs) {
    if (module_configs.empty()) {
        return;
    }

    const auto& expected = module_configs.front();
    RTP_LLM_CHECK_WITH_INFO(
        expected.num_layers == 1, "MTP module 0 must be a one-layer config, got %ld layers", expected.num_layers);
    RTP_LLM_CHECK_WITH_INFO(expected.kv_cache_spec_descs.size() == 1,
                            "MTP module 0 must have one descriptor row, got %zu",
                            expected.kv_cache_spec_descs.size());
    RTP_LLM_CHECK_WITH_INFO(!expected.kv_cache_spec_descs[0].empty(),
                            "MTP module 0 must have at least one cache descriptor");

    for (size_t module_index = 1; module_index < module_configs.size(); ++module_index) {
        const auto& current = module_configs[module_index];
        RTP_LLM_CHECK_WITH_INFO(current.num_layers == 1 && current.kv_cache_spec_descs.size() == 1,
                                "MTP module %zu must have one layer and one descriptor row",
                                module_index);

        const auto& expected_descs = expected.kv_cache_spec_descs[0];
        const auto& current_descs  = current.kv_cache_spec_descs[0];
        RTP_LLM_CHECK_WITH_INFO(current_descs.size() == expected_descs.size(),
                                "heterogeneous MTP cache layout: module %zu descriptor count %zu != module 0 count %zu",
                                module_index,
                                current_descs.size(),
                                expected_descs.size());
        for (size_t desc_index = 0; desc_index < expected_descs.size(); ++desc_index) {
            const auto& expected_desc = expected_descs[desc_index];
            const auto& current_desc  = current_descs[desc_index];
            RTP_LLM_CHECK_WITH_INFO(
                current_desc.cache_type == expected_desc.cache_type && current_desc.tag == expected_desc.tag
                    && current_desc.dtype == expected_desc.dtype,
                "heterogeneous MTP cache layout: module %zu descriptor %zu differs in type, tag, or dtype",
                module_index,
                desc_index);
        }

        const auto& expected_types = expected.hybrid_attention_config.hybrid_attention_types;
        const auto& current_types  = current.hybrid_attention_config.hybrid_attention_types;
        RTP_LLM_CHECK_WITH_INFO(current_types == expected_types,
                                "heterogeneous MTP cache layout: module %zu attention type differs from module 0",
                                module_index);
    }
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
    for (size_t module_index = 0; module_index < model_num; ++module_index) {
        const size_t source_layer = weight_count == 1 ? 0 : module_index;
        plan.source_layer_indices.push_back(source_layer);
        plan.module_configs.push_back(makeSingleLayerMTPModelConfig(model_config, source_layer));
    }
    validateHomogeneousMTPCacheLayouts(plan.module_configs);
    return plan;
}

}  // namespace rtp_llm
