#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

struct HybridPoolLayers {
    std::vector<int> full_layers;
    std::vector<int> linear_layers;
    std::vector<int> swa_layers;
};

HybridPoolLayers splitHybridPoolLayers(const ModelConfig& model_config) {
    const auto layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);
    RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.hybrid_attention_types.size()
                                == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            model_config.hybrid_attention_config.hybrid_attention_types.size(),
                            layer_num);

    HybridPoolLayers layers;
    layers.full_layers.reserve(static_cast<size_t>(layer_num));
    layers.linear_layers.reserve(static_cast<size_t>(layer_num));
    layers.swa_layers.reserve(static_cast<size_t>(layer_num));
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        switch (model_config.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                layers.linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                layers.swa_layers.push_back(i);
                break;
            case HybridAttentionType::NONE:
            default:
                layers.full_layers.push_back(i);
                break;
        }
    }
    return layers;
}

KVCacheSpecPtr createFullAttentionSpec(const ModelConfig&       model_config,
                                       const ParallelismConfig& parallelism_config,
                                       rtp_llm::DataType        dtype,
                                       uint32_t                 layer_num) {
    KVCacheSpecPtr spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        spec = std::make_shared<MLAKVCacheSpec>(model_config.attn_config, parallelism_config);
    } else {
        spec = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    }
    spec->dtype     = dtype;
    spec->layer_num = layer_num;
    return spec;
}

KVCacheSpecPtr createLinearAttentionSpec(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         rtp_llm::DataType        dtype,
                                         uint32_t                 layer_num) {
    auto spec = std::make_shared<LinearKVCacheSpec>(
        model_config.attn_config, parallelism_config, model_config.linear_attention_config);
    spec->dtype     = dtype;
    spec->layer_num = layer_num;
    return spec;
}

void appendGroup(CacheConfig&            config,
                 const std::vector<int>& layer_ids,
                 CacheGroupType          group_type,
                 KVCacheSpecPtr          spec,
                 KVCacheRegionName       region_name = KVCacheRegionName::DEFAULT) {
    if (layer_ids.empty()) {
        return;
    }
    config.global_layer_ids.push_back(layer_ids);
    config.layer_ids.push_back(layer_ids);
    config.cache_specs.push_back(spec);
    config.group_types.push_back(group_type);
    config.group_region_names.push_back(region_name);
}

void populateDefaultRegionMappings(CacheConfig& config) {
    config.layer_to_group_id.assign(config.layer_num, -1);
    config.layer_to_group_ids.assign(config.layer_num, std::vector<int>());
    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);

    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.assign(config.layer_num, std::vector<int>(region_count, -1));

    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        const auto region_name =
            gid < config.group_region_names.size() ? config.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        const auto region_id = static_cast<size_t>(region_name);
        RTP_LLM_CHECK_WITH_INFO(
            region_id < region_count, "invalid hybrid-pool region name %zu for group %zu", region_id, gid);
        for (int layer_id : config.layer_ids[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num,
                                    "invalid hybrid-pool layer id %d",
                                    layer_id);
            const auto layer = static_cast<size_t>(layer_id);
            config.layer_to_group_ids[layer].push_back(static_cast<int>(gid));
            config.layer_region_to_group_id[layer][region_id] = static_cast<int>(gid);
            if (region_name == KVCacheRegionName::DEFAULT) {
                config.layer_to_group_id[layer] = static_cast<int>(gid);
                config.layer_group_types[layer] = config.group_types[gid];
            }
        }
    }

    const auto swa_region_id = static_cast<size_t>(KVCacheRegionName::SWA_KV);
    for (size_t layer = 0; layer < static_cast<size_t>(config.layer_num); ++layer) {
        if (config.layer_to_group_id[layer] >= 0) {
            continue;
        }
        int fallback_gid = -1;
        if (swa_region_id < config.layer_region_to_group_id[layer].size()) {
            fallback_gid = config.layer_region_to_group_id[layer][swa_region_id];
        }
        if (fallback_gid < 0 && !config.layer_to_group_ids[layer].empty()) {
            fallback_gid = config.layer_to_group_ids[layer].back();
        }
        RTP_LLM_CHECK_WITH_INFO(fallback_gid >= 0, "missing hybrid-pool group mapping for layer %zu", layer);
        config.layer_to_group_id[layer] = fallback_gid;
        if (static_cast<size_t>(fallback_gid) < config.group_types.size()) {
            config.layer_group_types[layer] = config.group_types[static_cast<size_t>(fallback_gid)];
        }
    }
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto group_num               = static_cast<size_t>(config.groupNums());
    config.group_block_nums.resize(group_num, 0);
    config.group_seq_size_per_block.resize(group_num, config.seq_size_per_block);
    config.group_kv_block_stride_bytes.resize(group_num, 0);
    config.group_kv_scale_stride_bytes.resize(group_num, 0);
    config.group_block_size_bytes.resize(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    size_t   swa_kv_block_bytes      = 0;
    size_t   swa_scale_block_bytes   = 0;
    size_t   state_kv_block_bytes    = 0;
    size_t   state_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    // All groups share the global bpk (= seq_size_per_block / kernel_seq_size).
    // FULL paged + SWA fixed pools both view the spec's per-block bytes as
    // kernel-block-sized; multiplying by bpk gives the BlockPool's per-physical
    // allocation stride. Non-DSV4 paths default bpk=1 → no scaling.
    const size_t global_bpk = config.kernelBlocksPerKvBlock();
    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        const auto& spec = config.cache_specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count                = static_cast<uint32_t>(config.global_layer_ids[gid].size());
        const size_t kernel_kv_stride           = spec->block_size_bytes();
        const auto   kernel_scale               = spec->scale_block_size_bytes();
        const size_t kv_stride                  = kernel_kv_stride * global_bpk;
        const size_t scale_stride               = kernel_scale * global_bpk;
        config.group_kv_block_stride_bytes[gid] = kv_stride;
        config.group_kv_scale_stride_bytes[gid] = scale_stride;
        config.group_block_size_bytes[gid]      = static_cast<size_t>(layer_count) * (kv_stride + scale_stride);
        const auto region =
            gid < config.group_region_names.size() ? config.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        const bool is_state = isStateRegion(region);
        const bool is_swa   = gid < config.group_types.size() && config.group_types[gid] == CacheGroupType::SWA;
        if (is_state) {
            state_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            state_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        } else if (is_swa) {
            swa_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            swa_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        } else {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);

        for (int layer_id : config.global_layer_ids[gid]) {
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] =
                static_cast<int>(kv_stride + scale_stride);
        }
    }

    config.group_layer_num         = static_cast<int>(std::max<uint32_t>(1, max_group_layers));
    config.kv_block_stride_bytes   = max_kv_stride;
    config.kv_scale_stride_bytes   = max_scale_stride;
    config.kv_block_size_bytes     = total_kv_block_bytes;
    config.kv_scale_size_bytes     = total_scale_block_bytes;
    config.swa_block_size_bytes    = swa_kv_block_bytes + swa_scale_block_bytes;
    config.state_block_size_bytes  = state_kv_block_bytes + state_scale_block_bytes;
    const size_t paged_block_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    if (paged_block_bytes == 0) {
        RTP_LLM_CHECK_WITH_INFO(is_mtp && config.use_typed_cache_regions,
                                "hybrid-pool paged groups produced zero block bytes");
        config.kv_block_size_bytes = 1;
        config.kv_scale_size_bytes = 0;
        config.block_size_bytes    = 1;
    } else {
        config.block_size_bytes = paged_block_bytes;
    }
    config.fixed_pool_reserve_bytes = 0;
}

void populateHybridAttentionGroups(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config) {
    const auto dtype  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto layers = splitHybridPoolLayers(model_config);

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();

    appendGroup(config,
                layers.full_layers,
                CacheGroupType::FULL,
                createFullAttentionSpec(
                    model_config, parallelism_config, dtype, static_cast<uint32_t>(layers.full_layers.size())));
    appendGroup(config,
                layers.swa_layers,
                CacheGroupType::SWA,
                createFullAttentionSpec(
                    model_config, parallelism_config, dtype, static_cast<uint32_t>(layers.swa_layers.size())));
    appendGroup(config,
                layers.linear_layers,
                CacheGroupType::LINEAR,
                createLinearAttentionSpec(
                    model_config, parallelism_config, dtype, static_cast<uint32_t>(layers.linear_layers.size())));
}

void setupGroupCounts(CacheConfig& config) {
    config.full_group_num   = 0;
    config.swa_group_num    = 0;
    config.linear_group_num = 0;
    config.linear_fixed_cap = 0;
    for (auto group_type : config.group_types) {
        if (group_type == CacheGroupType::FULL) {
            ++config.full_group_num;
        } else if (group_type == CacheGroupType::SWA) {
            ++config.swa_group_num;
        } else {
            ++config.linear_group_num;
        }
    }
}

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            const KVCacheConfig&     kv_cache_config,
                                            bool                     is_mtp) {
    const auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    if (!model_config.attn_config.layer_compress_ratios.empty()) {
        DSV4CacheConfigHelper::applyConfig(config, model_config, kv_cache_config);
    } else {
        RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.enable_hybrid_attention,
                                "HybridPoolConfigCreator requires DSV4 layer_compress_ratios or hybrid attention");
        populateHybridAttentionGroups(config, model_config, parallelism_config);
    }

    RTP_LLM_CHECK_WITH_INFO(!config.cache_specs.empty(), "hybrid-pool config produced no cache specs");
    setupGroupCounts(config);
    populateDefaultRegionMappings(config);
    setupIndependentPoolSizes(config, is_mtp);
    if (!model_config.attn_config.layer_compress_ratios.empty()) {
        config.dsv4_fixed_pool_blocks = kv_cache_config.dsv4_fixed_pool_blocks;
    }
    return config;
}

}  // namespace

CacheConfig HybridPoolConfigCreator::createConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp) {
    return createHybridAttentionPoolConfig(model_config, parallelism_config, kv_cache_config, is_mtp);
}

}  // namespace rtp_llm
