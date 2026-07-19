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

    config.use_mla                   = model_config.attn_config.use_mla;
    config.dtype                     = dtype;
    config.is_sparse                 = model_config.attn_config.is_sparse;
    config.use_opaque_kv_cache_store = model_config.use_opaque_kv_cache_store;

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
        auto indexer_dim             = model_config.attn_config.indexer_head_dim;
        config.kv_scale_stride_bytes = (indexer_dim + indexer_dim / 128 * 4) * spec->seq_size_per_block;
        config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    } else if (!config.use_mla && model_config.attn_config.indexer_head_dim > 0) {
        // MiniMax-M3 MSA: the sparse-attention layers maintain a BF16 indexer K
        // cache (idx_K, one head, indexer_head_dim per token). Rather than a
        // separate side pool, piggyback it on the MHA scale region of the main
        // paged pool so it is addressed by the same block table and travels with
        // the main K/V during PD separation. is_mla stays false (the main K/V
        // keeps its HND layout); the scale region is exposed to Python as FP32
        // and reinterpreted as BF16 there. BF16 => 2 bytes/elem, so the per-block
        // stride is indexer_head_dim * 2 * seq_size_per_block bytes.
        auto indexer_dim             = static_cast<size_t>(model_config.attn_config.indexer_head_dim);
        config.kv_scale_stride_bytes = indexer_dim * 2 * spec->seq_size_per_block;
        config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
        // PD transfer: the idx_K BF16 cache in the scale slot is a single logical
        // block (not k/v separable), and the main K/V HND block is not k/v-split
        // friendly for byte-half partitioning. Use opaque whole-block PD transfer
        // (single kv_/kv_scale_ blocks) like GLM5/DSV4, so prefill-store and
        // decode-load keys/sizes match and the cache-store load completes.
        config.use_opaque_kv_cache_store = true;
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