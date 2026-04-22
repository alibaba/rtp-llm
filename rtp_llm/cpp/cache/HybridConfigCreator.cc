#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/core/DeviceData.h"

namespace rtp_llm {

std::vector<std::vector<int>> HybridConfigCreator::splitIntoGroups(const std::vector<int>& ids, int group_layer_num) {
    std::vector<std::vector<int>> groups;
    if (ids.empty()) {
        return groups;
    }
    const int n = static_cast<int>(ids.size());
    const int s = std::max(group_layer_num, 1);
    groups.reserve((n + s - 1) / s);
    for (int i = 0; i < n; i += s) {
        const int end = std::min(i + s, n);
        groups.emplace_back(ids.begin() + i, ids.begin() + end);
    }
    return groups;
}

int HybridConfigCreator::calculateGroupLayerNum(int linear_layer_count, int full_layer_count) {
    int group_layer_num = 0;
    if (linear_layer_count > 0 && full_layer_count > 0) {
        group_layer_num = std::gcd(linear_layer_count, full_layer_count);
    } else {
        group_layer_num = std::max(linear_layer_count, full_layer_count);
    }
    group_layer_num = std::max(group_layer_num, 1);
    return group_layer_num;
}

std::pair<std::vector<int>, std::vector<int>>
HybridConfigCreator::splitLayersByAttentionType(const ModelConfig& model_config) {
    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    std::vector<int> linear_layers;
    std::vector<int> full_layers;
    linear_layers.reserve(layer_num);
    full_layers.reserve(layer_num);

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        if (types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            linear_layers.push_back(i);
        } else {
            full_layers.push_back(i);
        }
    }

    return std::make_pair(std::move(linear_layers), std::move(full_layers));
}

CacheConfig HybridConfigCreator::initializeConfig(const ModelConfig&      model_config,
                                                  const std::vector<int>& linear_layers,
                                                  const std::vector<int>& full_layers,
                                                  rtp_llm::DataType       dtype) {
    int64_t layer_num = model_config.num_layers;

    CacheConfig config;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);

    return config;
}

KVCacheSpecPtr HybridConfigCreator::createFullAttentionSpec(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            rtp_llm::DataType        dtype) {
    KVCacheSpecPtr full_spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        full_spec = std::make_shared<MLAKVCacheSpec>(model_config.attn_config, parallelism_config);
    } else {
        full_spec = std::make_shared<MHAKVCacheSpec>(model_config.attn_config, parallelism_config);
    }
    full_spec->dtype = dtype;
    return full_spec;
}

KVCacheSpecPtr HybridConfigCreator::createLinearAttentionSpec(const ModelConfig&       model_config,
                                                              const ParallelismConfig& parallelism_config,
                                                              rtp_llm::DataType        dtype) {
    auto linear_spec = std::make_shared<LinearKVCacheSpec>(
        model_config.attn_config, parallelism_config, model_config.linear_attention_config);
    linear_spec->dtype = dtype;
    return linear_spec;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> HybridConfigCreator::createLayerGroups(
    const std::vector<int>& linear_layers, const std::vector<int>& full_layers, int& group_layer_num) {
    const int linear_cnt = static_cast<int>(linear_layers.size());
    const int full_cnt   = static_cast<int>(full_layers.size());
    group_layer_num      = HybridConfigCreator::calculateGroupLayerNum(linear_cnt, full_cnt);

    const auto linear_groups = HybridConfigCreator::splitIntoGroups(linear_layers, group_layer_num);
    const auto full_groups   = HybridConfigCreator::splitIntoGroups(full_layers, group_layer_num);

    return std::make_pair(std::move(linear_groups), std::move(full_groups));
}

std::vector<std::vector<int>>
HybridConfigCreator::setupCacheConfigSpecs(CacheConfig&                         config,
                                           const std::vector<std::vector<int>>& linear_groups,
                                           const std::vector<std::vector<int>>& full_groups,
                                           const KVCacheSpecPtr&                linear_spec,
                                           const KVCacheSpecPtr&                full_spec,
                                           std::vector<KVCacheSpecPtr>&         out_cache_specs,
                                           std::vector<CacheGroupType>&         out_group_types,
                                           int&                                 out_linear_group_num,
                                           int&                                 out_full_group_num) {
    out_cache_specs.clear();
    out_group_types.clear();

    std::vector<std::vector<int>> layer_ids;
    // Keep order: all full groups first, then linear groups.
    for (const auto& g : full_groups) {
        layer_ids.push_back(g);
        out_cache_specs.push_back(full_spec);
        out_group_types.push_back(CacheGroupType::FULL);
    }
    for (const auto& g : linear_groups) {
        layer_ids.push_back(g);
        out_cache_specs.push_back(linear_spec);
        out_group_types.push_back(CacheGroupType::LINEAR);
    }
    out_linear_group_num = static_cast<int>(linear_groups.size());
    out_full_group_num   = static_cast<int>(full_groups.size());
    return layer_ids;
}

void HybridConfigCreator::setupPhysicalSizes(int                   group_layer_num,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec,
                                             size_t&               out_kv_block_stride_bytes,
                                             size_t&               out_kv_scale_stride_bytes,
                                             size_t&               out_kv_block_size_bytes,
                                             size_t&               out_kv_scale_size_bytes,
                                             size_t&               out_block_size_bytes) {
    // Decide the physical KV block/scale sizes by taking max between full and linear specs.
    const size_t full_kv_block_stride_bytes   = full_spec->block_size_bytes();
    const size_t linear_kv_block_stride_bytes = linear_spec->block_size_bytes();

    // now we only support that linear attention block have padding
    RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                            "not support full attention with padding now");

    out_kv_block_stride_bytes = full_kv_block_stride_bytes;
    out_kv_block_size_bytes   = static_cast<size_t>(group_layer_num) * out_kv_block_stride_bytes;
    out_kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    out_kv_scale_size_bytes   = static_cast<size_t>(group_layer_num) * out_kv_scale_stride_bytes;
    out_block_size_bytes      = out_kv_block_size_bytes + out_kv_scale_size_bytes;
}

void HybridConfigCreator::setupLayerToGroupMapping(uint32_t                             layer_num,
                                                   const std::vector<std::vector<int>>& layer_ids,
                                                   std::vector<int>&                    out_layer_to_group_id) {
    out_layer_to_group_id.assign(layer_num, 0);
    for (size_t gid = 0; gid < layer_ids.size(); ++gid) {
        for (int layer_id : layer_ids[gid]) {
            if (layer_id >= 0 && static_cast<size_t>(layer_id) < layer_num) {
                out_layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int32_t>(gid);
            }
        }
    }
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config, buildDeviceType());

    const uint32_t layer_num = static_cast<uint32_t>(model_config.num_layers);

    // Split layers by attention type
    auto [linear_layers, full_layers] = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config = HybridConfigCreator::initializeConfig(model_config, linear_layers, full_layers, dtype);

    // Create attention specs
    auto full_spec   = HybridConfigCreator::createFullAttentionSpec(model_config, parallelism_config, dtype);
    auto linear_spec = HybridConfigCreator::createLinearAttentionSpec(model_config, parallelism_config, dtype);

    // Create layer groups and calculate group layer number
    int group_layer_num = 0;
    auto [linear_groups, full_groups] =
        HybridConfigCreator::createLayerGroups(linear_layers, full_layers, group_layer_num);

    // Build per-group spec/type lists and layer_ids
    std::vector<KVCacheSpecPtr> cache_specs;
    std::vector<CacheGroupType> group_types;
    int                         linear_group_num     = 0;
    int                         full_group_num       = 0;
    auto                        layer_ids_for_groups = HybridConfigCreator::setupCacheConfigSpecs(config,
                                                                           linear_groups,
                                                                           full_groups,
                                                                           linear_spec,
                                                                           full_spec,
                                                                           cache_specs,
                                                                           group_types,
                                                                           linear_group_num,
                                                                           full_group_num);

    // Compute physical block sizes
    size_t kv_block_stride = 0, kv_scale_stride = 0;
    size_t kv_block_size = 0, kv_scale_size = 0, block_size = 0;
    HybridConfigCreator::setupPhysicalSizes(group_layer_num,
                                            full_spec,
                                            linear_spec,
                                            kv_block_stride,
                                            kv_scale_stride,
                                            kv_block_size,
                                            kv_scale_size,
                                            block_size);

    // Build layer-to-group mapping
    std::vector<int> layer_to_group_id;
    HybridConfigCreator::setupLayerToGroupMapping(layer_num, layer_ids_for_groups, layer_to_group_id);

    // Per-layer block stride (kv + scale).
    const size_t     per_layer_stride = kv_block_stride + kv_scale_stride;
    std::vector<int> layer_to_block_stride(layer_num, static_cast<int>(per_layer_stride));

    // Build allocator config for main model (model_id=0).
    KVCacheAllocatorConfig alloc_config;
    alloc_config.model_id                    = 0;
    alloc_config.layer_num                   = layer_num;
    alloc_config.dtype                       = dtype;
    alloc_config.use_mla                     = model_config.attn_config.use_mla;
    alloc_config.is_sparse                   = false;
    alloc_config.cache_specs                 = cache_specs;
    alloc_config.group_types                 = group_types;
    alloc_config.layer_ids                   = layer_ids_for_groups;
    alloc_config.layer_to_group_id           = layer_to_group_id;
    alloc_config.layer_to_block_stride_bytes = layer_to_block_stride;
    alloc_config.block_num                   = 0;  // filled in by createConfig()
    alloc_config.seq_size_per_block          = config.seq_size_per_block;
    alloc_config.kv_block_size_bytes         = kv_block_size;
    alloc_config.kv_scale_size_bytes         = kv_scale_size;
    alloc_config.block_size_bytes            = block_size;
    alloc_config.kv_block_stride_bytes       = kv_block_stride;
    alloc_config.kv_scale_stride_bytes       = kv_scale_stride;
    alloc_config.linear_step                 = 1;
    alloc_config.group_layer_num             = group_layer_num;
    alloc_config.linear_group_num            = linear_group_num;
    alloc_config.full_group_num              = full_group_num;

    // Fill cross-model fields in CacheConfig.
    config.layer_to_group_id           = layer_to_group_id;
    config.layer_to_block_stride_bytes = layer_to_block_stride;
    config.allocator_configs.push_back(std::move(alloc_config));

    return config;
}

}  // namespace rtp_llm