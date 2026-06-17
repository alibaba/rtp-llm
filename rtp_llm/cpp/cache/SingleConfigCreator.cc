#include "rtp_llm/cpp/cache/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <numeric>

namespace rtp_llm {

namespace {

KVCacheSpecPtr getLayerDefaultSpec(const ModelConfig& model_config, int64_t layer_id) {
    const auto it = model_config.kv_cache_specs.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != model_config.kv_cache_specs.end(),
                            "single cache config missing kv_cache_specs for layer %ld",
                            layer_id);
    RTP_LLM_CHECK_WITH_INFO(it->second.size() == 1,
                            "single cache config requires exactly one spec for layer %ld, got %zu",
                            layer_id,
                            it->second.size());
    auto spec = it->second[0];
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "single cache config got null kv_cache spec for layer %ld", layer_id);
    RTP_LLM_CHECK_WITH_INFO(spec->tag == "default",
                            "single cache config requires tag=default for layer %ld, got=%s",
                            layer_id,
                            spec->tag.c_str());
    return spec;
}

KVCacheSpecPtr getDefaultSpecFromModel(const ModelConfig&       model_config,
                                       const ParallelismConfig& parallelism_config,
                                       rtp_llm::DataType        dtype) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "single cache config requires layer-wise kv_cache_specs for every layer, got %zu/%ld",
                            model_config.kv_cache_specs.size(),
                            model_config.num_layers);
    auto spec = getLayerDefaultSpec(model_config, 0)->clone();

    for (int64_t layer_id = 1; layer_id < model_config.num_layers; ++layer_id) {
        auto layer_spec = getLayerDefaultSpec(model_config, layer_id);
        RTP_LLM_CHECK_WITH_INFO(layer_spec->fingerprint() == spec->fingerprint(),
                                "single cache config default spec differs at layer %ld",
                                layer_id);
    }

    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto* mla_spec = dynamic_cast<MLAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mla_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadLatentAttention,
                                "default kv_cache spec must be MLAKVCacheSpec for MLA model");
        spec->local_head_num_kv  = 1;
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        mla_spec->kv_lora_rank   = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        mla_spec->rope_head_dim  = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
    } else {
        auto* mha_spec = dynamic_cast<MHAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mha_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadAttention,
                                "default kv_cache spec must be MHAKVCacheSpec for MHA/GQA model");
        spec->local_head_num_kv = static_cast<uint32_t>(
            (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                model_config.attn_config.kv_head_num
                    / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        mha_spec->size_per_head  = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    }
    spec->dtype = dtype;
    return spec;
}

}  // namespace

CacheConfig SingleConfigCreator::createSingleConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    auto layer_num = model_config.num_layers;

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

    config.use_mla   = model_config.attn_config.use_mla;
    config.dtype     = dtype;
    config.is_sparse = model_config.attn_config.is_sparse;

    KVCacheSpecPtr spec = getDefaultSpecFromModel(model_config, parallelism_config, dtype);

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

    LayerKVCacheSpecs layer_specs;
    for (int layer_id : all_layer_ids) {
        layer_specs[static_cast<int64_t>(layer_id)] = {spec};
    }
    config.fromLayerSpecs(layer_specs);
    return config;
}

}  // namespace rtp_llm
