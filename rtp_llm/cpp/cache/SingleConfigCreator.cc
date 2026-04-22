#include "rtp_llm/cpp/cache/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheConfig SingleConfigCreator::createSingleConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config, buildDeviceType());

    auto layer_num = model_config.num_layers;

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    KVCacheSpecPtr spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        spec = std::make_shared<MLAKVCacheSpec>(model_config.attn_config, parallelism_config);
    } else {
        spec = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    }
    spec->dtype = dtype;

    const bool is_sparse       = model_config.attn_config.is_sparse;
    const bool use_mla         = model_config.attn_config.use_mla;
    size_t     kv_block_stride = spec->block_size_bytes();
    size_t     kv_scale_stride = spec->scale_block_size_bytes();

    if (is_sparse) {
        const auto indexer_dim = model_config.attn_config.indexer_head_dim;
        kv_scale_stride        = (indexer_dim + indexer_dim / 128 * 4) * spec->seq_size_per_block;
    }

    const size_t kv_block_size      = static_cast<size_t>(layer_num) * kv_block_stride;
    const size_t kv_scale_size      = static_cast<size_t>(layer_num) * kv_scale_stride;
    const size_t block_size         = kv_block_size + kv_scale_size;
    const size_t per_layer_stride   = kv_block_stride + kv_scale_stride;
    const size_t seq_size_per_block = static_cast<size_t>(model_config.attn_config.tokens_per_block);

    std::vector<int> layer_to_group_id(layer_num, 0);
    std::vector<int> layer_to_block_stride(layer_num, static_cast<int>(per_layer_stride));

    // Build allocator config for main model (model_id=0).
    KVCacheAllocatorConfig alloc_config;
    alloc_config.model_id                    = 0;
    alloc_config.layer_num                   = static_cast<uint32_t>(layer_num);
    alloc_config.dtype                       = dtype;
    alloc_config.use_mla                     = use_mla;
    alloc_config.is_sparse                   = is_sparse;
    alloc_config.cache_specs                 = {spec};
    alloc_config.group_types                 = {CacheGroupType::FULL};
    alloc_config.layer_ids                   = {all_layer_ids};
    alloc_config.layer_to_group_id           = layer_to_group_id;
    alloc_config.layer_to_block_stride_bytes = layer_to_block_stride;
    alloc_config.block_num                   = 0;  // filled in by createConfig()
    alloc_config.seq_size_per_block          = seq_size_per_block;
    alloc_config.kv_block_size_bytes         = kv_block_size;
    alloc_config.kv_scale_size_bytes         = kv_scale_size;
    alloc_config.block_size_bytes            = block_size;
    alloc_config.kv_block_stride_bytes       = kv_block_stride;
    alloc_config.kv_scale_stride_bytes       = kv_scale_stride;
    alloc_config.linear_step                 = 1;
    alloc_config.group_layer_num             = layer_num;  // only 1 group for SingleConfig
    alloc_config.linear_group_num            = 0;
    alloc_config.full_group_num              = 1;

    // Build CacheConfig (cross-model shared fields only).
    CacheConfig config;
    config.seq_size_per_block          = seq_size_per_block;
    config.layer_all_num               = static_cast<uint32_t>(layer_num);
    config.layer_to_group_id           = layer_to_group_id;
    config.layer_to_block_stride_bytes = layer_to_block_stride;
    config.allocator_configs.push_back(std::move(alloc_config));

    return config;
}

}  // namespace rtp_llm