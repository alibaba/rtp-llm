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
        auto indexer_dim             = model_config.attn_config.indexer_head_dim;
        config.kv_scale_stride_bytes = (indexer_dim + indexer_dim / 128 * 4) * spec->seq_size_per_block;
        size_t indexer_slot_num      = config.layer_num;
        if (model_config.enable_glm52_shared_indexer_kv_cache) {
            RTP_LLM_CHECK_WITH_INFO(!is_mtp, "GLM5.2 shared Indexer KV cache must not be enabled for the MTP model");
            RTP_LLM_CHECK_WITH_INFO(model_config.model_type == "glm_5",
                                    "GLM5.2 shared Indexer KV cache only supports model_type=glm_5");
            const auto& mapping = model_config.glm52_indexer_kv_slot_mapping;
            RTP_LLM_CHECK_WITH_INFO(mapping.size() == config.layer_num,
                                    "GLM5.2 Indexer KV mapping size(%zu) != layer_num(%u)",
                                    mapping.size(),
                                    config.layer_num);
            RTP_LLM_CHECK_WITH_INFO(!mapping.empty() && mapping.front() == 0,
                                    "GLM5.2 Indexer KV mapping must start at physical slot 0");

            int  previous_slot = -1;
            bool has_shared    = false;
            for (size_t layer_id = 0; layer_id < mapping.size(); ++layer_id) {
                const int slot = mapping[layer_id];
                RTP_LLM_CHECK_WITH_INFO(slot == previous_slot || slot == previous_slot + 1,
                                        "invalid GLM5.2 Indexer KV slot at layer %zu: slot=%d previous=%d",
                                        layer_id,
                                        slot,
                                        previous_slot);
                has_shared |= slot == previous_slot;
                previous_slot = slot;
            }
            RTP_LLM_CHECK_WITH_INFO(has_shared, "GLM5.2 Indexer KV mapping contains no shared layer");
            indexer_slot_num                = static_cast<size_t>(previous_slot + 1);
            config.layer_to_indexer_kv_slot = mapping;
            RTP_LLM_LOG_INFO("GLM5.2 shared Indexer KV cache enabled: logical_layers=%u, physical_slots=%zu, "
                             "shared_layers=%zu",
                             config.layer_num,
                             indexer_slot_num,
                             static_cast<size_t>(config.layer_num) - indexer_slot_num);
        }
        config.kv_scale_size_bytes = indexer_slot_num * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.group_layer_num  = layer_num;  // only 1 group for SingleConfig

    // Cache connectors keep one logical slot per layer. Compact shared layers
    // have no scale tensor, so their logical payload contains MLA KV only.
    config.layer_to_block_stride_bytes.resize(static_cast<size_t>(config.layer_all_num));
    for (size_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        const bool owns_indexer_slot =
            config.layer_to_indexer_kv_slot.empty() || layer_id == 0
            || config.layer_to_indexer_kv_slot[layer_id] != config.layer_to_indexer_kv_slot[layer_id - 1];
        const size_t scale_bytes                     = owns_indexer_slot ? config.kv_scale_stride_bytes : 0;
        config.layer_to_block_stride_bytes[layer_id] = static_cast<int>(config.kv_block_stride_bytes + scale_bytes);
    }

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
