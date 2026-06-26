#include "rtp_llm/cpp/cache/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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

    KVCacheSpecPtr spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        spec = std::make_shared<MLAKVCacheSpec>(model_config.attn_config, parallelism_config);
    } else {
        spec = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    }
    spec->dtype = dtype;
    config.cache_specs.push_back(spec);
    config.group_types.push_back(CacheGroupType::FULL);

    // Using spec interface for block size and scale
    config.kv_block_stride_bytes = config.cache_specs[0]->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;

    // Scale handling - no need to check dtype as scale_block_size_bytes() returns 0 if no scale support
    config.kv_scale_stride_bytes = config.cache_specs[0]->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;

    if (config.is_sparse) {
        auto indexer_dim = model_config.attn_config.indexer_head_dim;
        // Indexer KV cache stride: FP8 = head_dim + head_dim/128*4 bytes/token
        // (128B FP8 K + 4B FP32 scale); FP4 = head_dim/2 + head_dim/32 bytes/token
        // (64B FP4 K + 4B packed UE8M0 scale on HD=128). Keep in sync with
        // MLAKVCacheSpec::scale_block_size_bytes and the Python view in
        // IndexerOp._head_dim_with_sf.
        size_t per_token_bytes;
        if (model_config.attn_config.indexer_quant_dtype == "fp4") {
            per_token_bytes = static_cast<size_t>(indexer_dim / 2 + indexer_dim / 32);
        } else {
            per_token_bytes = static_cast<size_t>(indexer_dim + indexer_dim / 128 * 4);
        }
        config.kv_scale_stride_bytes = per_token_bytes * spec->seq_size_per_block;
        config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.group_layer_num  = layer_num;  // only 1 group for SingleConfig

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    // Global layer ids are the indices used by BlockPool::convertIndexToAddr (0..N-1 in a single-model case).
    config.global_layer_ids.push_back(all_layer_ids);
    config.layer_ids.push_back(all_layer_ids);
    config.layer_to_group_id.assign(config.layer_num, 0);
    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);
    // Populate region mapping: single group uses DEFAULT region.
    config.group_region_names.push_back(KVCacheRegionName::DEFAULT);
    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.resize(config.layer_num);
    for (size_t i = 0; i < config.layer_num; i++) {
        config.layer_region_to_group_id[i].assign(region_count, -1);
        config.layer_region_to_group_id[i][static_cast<size_t>(KVCacheRegionName::DEFAULT)] = 0;
    }
    return config;
}

}  // namespace rtp_llm