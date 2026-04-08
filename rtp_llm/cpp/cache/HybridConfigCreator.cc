#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

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

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
HybridConfigCreator::splitLayersByAttentionType3(const ModelConfig& model_config) {
    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    std::vector<int> linear_layers;
    std::vector<int> sliding_window_layers;
    std::vector<int> full_layers;

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        auto t = types[static_cast<size_t>(i)];
        if (t == HybridAttentionType::LINEAR) {
            linear_layers.push_back(i);
        } else if (t == HybridAttentionType::SLIDING_WINDOW) {
            sliding_window_layers.push_back(i);
        } else {
            full_layers.push_back(i);
        }
    }

    return std::make_tuple(std::move(linear_layers), std::move(sliding_window_layers), std::move(full_layers));
}

CacheConfig HybridConfigCreator::initializeConfig(const ModelConfig&      model_config,
                                                  const std::vector<int>& linear_layers,
                                                  const std::vector<int>& full_layers,
                                                  rtp_llm::DataType       dtype) {
    int64_t layer_num = model_config.num_layers;

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;

    config.global_layer_ids.push_back(linear_layers);
    config.global_layer_ids.push_back(full_layers);
    config.layer_ids.push_back(linear_layers);
    config.layer_ids.push_back(full_layers);

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

KVCacheSpecPtr HybridConfigCreator::createCustomMHASpec(int                      kv_head_num,
                                                        int                      size_per_head,
                                                        int                      tokens_per_block,
                                                        const ParallelismConfig& parallelism_config,
                                                        rtp_llm::DataType        dtype) {
    auto spec              = std::make_shared<MHAKVCacheSpec>();
    spec->type             = KVCacheSpecType::MultiHeadAttention;
    spec->layer_num        = 1;
    spec->local_head_num_kv = static_cast<uint32_t>(
        std::max(1,
                 (kv_head_num > 1) ?
                     static_cast<int>(kv_head_num / parallelism_config.get_attn_tp_size()) :
                     kv_head_num));
    spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    spec->size_per_head      = static_cast<uint32_t>(size_per_head);
    spec->dtype              = dtype;
    return spec;
}

KVCacheSpecPtr HybridConfigCreator::createSlidingWindowSpec(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            rtp_llm::DataType        dtype) {
    const auto& hybrid_config = model_config.hybrid_attention_config;
    return createCustomMHASpec(
        hybrid_config.sliding_window_kv_head_num,
        hybrid_config.sliding_window_size_per_head,
        model_config.attn_config.tokens_per_block,
        parallelism_config,
        dtype);
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

void HybridConfigCreator::setupCacheConfigSpecs(CacheConfig&                         config,
                                                const std::vector<std::vector<int>>& linear_groups,
                                                const std::vector<std::vector<int>>& full_groups,
                                                const KVCacheSpecPtr&                linear_spec,
                                                const KVCacheSpecPtr&                full_spec) {
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.cache_specs.clear();
    config.group_types.clear();

    // Keep order: all full groups first, then linear groups.
    for (const auto& g : full_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(full_spec);
        config.group_types.push_back(CacheGroupType::FULL);
    }
    for (const auto& g : linear_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(linear_spec);
        config.group_types.push_back(CacheGroupType::LINEAR);
    }
    config.linear_group_num = static_cast<int>(linear_groups.size());
    config.full_group_num   = static_cast<int>(full_groups.size());
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig&          config,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec) {
    setupPhysicalSizes(config, {full_spec, linear_spec});
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig& config, const std::vector<KVCacheSpecPtr>& all_specs) {
    size_t max_kv_block_stride_bytes = 0;
    size_t max_scale_stride_bytes    = 0;

    for (const auto& spec : all_specs) {
        if (spec) {
            max_kv_block_stride_bytes = std::max(max_kv_block_stride_bytes, spec->block_size_bytes());
            max_scale_stride_bytes    = std::max(max_scale_stride_bytes, spec->scale_block_size_bytes());
        }
    }

    config.kv_block_stride_bytes = max_kv_block_stride_bytes;
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = max_scale_stride_bytes;
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
}

void HybridConfigCreator::setupLayerToGroupMapping(CacheConfig& config) {
    config.layer_to_group_id.assign(config.layer_num, 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            if (layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num) {
                config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int32_t>(gid);
            }
        }
    }
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config, buildDeviceType());

    const auto& hybrid_config = model_config.hybrid_attention_config;
    const bool  has_sliding_window = hybrid_config.sliding_window_kv_head_num > 0;

    // Check if we have a three-way split (SLIDING_WINDOW + FULL, e.g., Gemma4)
    if (has_sliding_window) {
        auto [linear_layers, sw_layers, full_layers] =
            HybridConfigCreator::splitLayersByAttentionType3(model_config);

        RTP_LLM_CHECK_WITH_INFO(linear_layers.empty(),
            "Models with SLIDING_WINDOW layers cannot also have LINEAR layers");

        int64_t layer_num = model_config.num_layers;
        CacheConfig config;
        config.layer_num          = static_cast<uint32_t>(layer_num);
        config.layer_all_num      = static_cast<uint32_t>(layer_num);
        config.block_num          = 0;
        config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        config.use_mla            = model_config.attn_config.use_mla;
        config.dtype              = dtype;
        config.linear_step        = 1;

        // Create specs for each type
        auto sw_spec = HybridConfigCreator::createSlidingWindowSpec(model_config, parallelism_config, dtype);

        KVCacheSpecPtr full_spec;
        if (hybrid_config.global_kv_head_num > 0) {
            full_spec = createCustomMHASpec(
                hybrid_config.global_kv_head_num,
                hybrid_config.global_size_per_head,
                model_config.attn_config.tokens_per_block,
                parallelism_config,
                dtype);
        } else {
            full_spec = HybridConfigCreator::createFullAttentionSpec(model_config, parallelism_config, dtype);
        }

        // Calculate group layer num from the two MHA types
        const int sw_cnt   = static_cast<int>(sw_layers.size());
        const int full_cnt = static_cast<int>(full_layers.size());
        int group_layer_num = calculateGroupLayerNum(sw_cnt, full_cnt);
        config.group_layer_num = group_layer_num;

        auto sw_groups   = splitIntoGroups(sw_layers, group_layer_num);
        auto full_groups = splitIntoGroups(full_layers, group_layer_num);

        // Setup groups: sliding window first (larger cache), then full attention
        config.global_layer_ids.clear();
        config.layer_ids.clear();
        config.cache_specs.clear();
        config.group_types.clear();

        for (const auto& g : sw_groups) {
            config.global_layer_ids.push_back(g);
            config.layer_ids.push_back(g);
            config.cache_specs.push_back(sw_spec);
            config.group_types.push_back(CacheGroupType::SLIDING_WINDOW);
        }
        for (const auto& g : full_groups) {
            config.global_layer_ids.push_back(g);
            config.layer_ids.push_back(g);
            config.cache_specs.push_back(full_spec);
            config.group_types.push_back(CacheGroupType::FULL);
        }
        config.linear_group_num = 0;
        config.full_group_num   = static_cast<int>(full_groups.size());

        // Setup physical sizes using max of all specs
        setupPhysicalSizes(config, {sw_spec, full_spec});

        // Setup layer to group mapping
        setupLayerToGroupMapping(config);

        config.layer_attn_types.assign(config.layer_num, CacheGroupType::FULL);
        for (size_t layer_id = 0; layer_id < config.layer_to_group_id.size(); ++layer_id) {
            const int gid = config.layer_to_group_id[layer_id];
            if (gid >= 0 && static_cast<size_t>(gid) < config.group_types.size()) {
                config.layer_attn_types[layer_id] = config.group_types[static_cast<size_t>(gid)];
            }
        }

        const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
        config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                                  static_cast<int>(per_layer_stride_bytes));
        return config;
    }

    // Original two-way split path (LINEAR + FULL)
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
    config.group_layer_num = group_layer_num;

    // Setup cache config specs
    HybridConfigCreator::setupCacheConfigSpecs(config, linear_groups, full_groups, linear_spec, full_spec);

    // Setup physical sizes
    HybridConfigCreator::setupPhysicalSizes(config, full_spec, linear_spec);

    // Setup layer to group mapping
    HybridConfigCreator::setupLayerToGroupMapping(config);

    config.layer_attn_types.assign(config.layer_num, CacheGroupType::FULL);
    for (size_t layer_id = 0; layer_id < config.layer_to_group_id.size(); ++layer_id) {
        const int gid = config.layer_to_group_id[layer_id];
        if (gid >= 0 && static_cast<size_t>(gid) < config.group_types.size()) {
            config.layer_attn_types[layer_id] = config.group_types[static_cast<size_t>(gid)];
        }
    }

    // Per-layer block stride (kv + scale).
    // For hybrid attention, the physical per-layer stride follows the selected physical layout stride.
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

}  // namespace rtp_llm
