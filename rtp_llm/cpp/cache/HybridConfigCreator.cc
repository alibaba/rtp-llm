#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <initializer_list>
#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

namespace rtp_llm {
namespace {

int calculateGroupLayerNumForCounts(std::initializer_list<int> counts) {
    int group_layer_num = 0;
    for (int count : counts) {
        if (count <= 0) {
            continue;
        }
        group_layer_num = group_layer_num == 0 ? count : std::gcd(group_layer_num, count);
    }
    return std::max(group_layer_num, 1);
}

}  // namespace

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
    // All full attention layers must reside in one cache group (full_group_num <= 1).
    // prepare_fmha_impl binds the block table of group 0 once; it is not re-bound per layer.
    // group_layer_num must be >= full_layer_count to satisfy this.
    // When gcd is already sufficient it works directly; the fallback handles all other cases
    // (coprime gcd==1, or gcd>1 but still smaller than full_layer_count).
    int group_layer_num = 0;
    if (linear_layer_count > 0 && full_layer_count > 0) {
        group_layer_num = std::gcd(linear_layer_count, full_layer_count);
        // Fallback: when gcd < full_layer_count, force group_layer_num = full_layer_count
        // to guarantee all full layers fit in one group.
        // e.g. Kimi Linear 20:7 -> gcd=1 < 7 -> group_layer_num=7, linear groups=[7,7,6],
        // last group wastes 1 layer slot per block, negligible.
        if (group_layer_num < full_layer_count) {
            group_layer_num = full_layer_count;
        }
    } else {
        group_layer_num = std::max(linear_layer_count, full_layer_count);
    }
    group_layer_num = std::max(group_layer_num, 1);
    return group_layer_num;
}

HybridConfigCreator::AttentionLayerSplit
HybridConfigCreator::splitLayersByAttentionType(const ModelConfig& model_config) {
    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    AttentionLayerSplit split;
    split.linear_layers.reserve(layer_num);
    split.swa_layers.reserve(layer_num);
    split.full_layers.reserve(layer_num);

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        if (types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            split.linear_layers.push_back(i);
        } else if (types[static_cast<size_t>(i)] == HybridAttentionType::SLIDING_WINDOW) {
            split.swa_layers.push_back(i);
        } else {
            split.full_layers.push_back(i);
        }
    }

    return split;
}

CacheConfig HybridConfigCreator::initializeConfig(const ModelConfig&      model_config,
                                                  const std::vector<int>& linear_layers,
                                                  const std::vector<int>& swa_layers,
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
    config.global_layer_ids.push_back(swa_layers);
    config.global_layer_ids.push_back(full_layers);
    config.layer_ids.push_back(linear_layers);
    config.layer_ids.push_back(swa_layers);
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
    full_spec->dtype     = dtype;
    full_spec->use_mla   = model_config.attn_config.use_mla;
    full_spec->is_sparse = model_config.attn_config.is_sparse;
    return full_spec;
}

KVCacheSpecPtr HybridConfigCreator::createLinearAttentionSpec(const ModelConfig&       model_config,
                                                              const ParallelismConfig& parallelism_config,
                                                              rtp_llm::DataType        dtype) {
    auto linear_spec = std::make_shared<LinearKVCacheSpec>(
        model_config.attn_config, parallelism_config, model_config.linear_attention_config);
    linear_spec->dtype     = dtype;
    linear_spec->use_mla   = false;
    linear_spec->is_sparse = false;
    return linear_spec;
}

HybridConfigCreator::AttentionGroupSplit HybridConfigCreator::createLayerGroups(const std::vector<int>& linear_layers,
                                                                                const std::vector<int>& swa_layers,
                                                                                const std::vector<int>& full_layers,
                                                                                int& group_layer_num) {
    const int linear_cnt = static_cast<int>(linear_layers.size());
    const int swa_cnt    = static_cast<int>(swa_layers.size());
    const int full_cnt   = static_cast<int>(full_layers.size());
    group_layer_num      = calculateGroupLayerNumForCounts({linear_cnt, swa_cnt, full_cnt});

    AttentionGroupSplit groups;
    groups.linear_groups = HybridConfigCreator::splitIntoGroups(linear_layers, group_layer_num);
    groups.swa_groups    = HybridConfigCreator::splitIntoGroups(swa_layers, group_layer_num);
    groups.full_groups   = HybridConfigCreator::splitIntoGroups(full_layers, group_layer_num);
    return groups;
}

void HybridConfigCreator::setupCacheConfigSpecs(CacheConfig&                         config,
                                                const std::vector<std::vector<int>>& linear_groups,
                                                const std::vector<std::vector<int>>& swa_groups,
                                                const std::vector<std::vector<int>>& full_groups,
                                                const KVCacheSpecPtr&                linear_spec,
                                                const KVCacheSpecPtr&                full_spec) {
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.cache_specs.clear();
    config.group_types.clear();

    // Keep order: all full groups first, then SWA groups, then linear groups.
    for (const auto& g : full_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(full_spec);
        config.group_types.push_back(CacheGroupType::FULL);
    }
    for (const auto& g : swa_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(full_spec);
        config.group_types.push_back(CacheGroupType::SWA);
    }
    for (const auto& g : linear_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(linear_spec);
        config.group_types.push_back(CacheGroupType::LINEAR);
    }
    config.linear_group_num = static_cast<int>(linear_groups.size());
    config.swa_group_num    = static_cast<int>(swa_groups.size());
    config.full_group_num   = static_cast<int>(full_groups.size());
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig&          config,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec) {
    // Decide the physical KV block/scale sizes by taking max between full and linear specs.
    const size_t full_kv_block_stride_bytes   = full_spec->block_size_bytes();
    const size_t linear_kv_block_stride_bytes = linear_spec->block_size_bytes();

    // now we only support that linear attention block have padding
    RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                            "not support full attention with padding now");

    config.kv_block_stride_bytes = full_kv_block_stride_bytes;
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
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
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    // Split layers by attention type
    auto split = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config = HybridConfigCreator::initializeConfig(
        model_config, split.linear_layers, split.swa_layers, split.full_layers, dtype);

    // Create attention specs
    auto full_spec   = HybridConfigCreator::createFullAttentionSpec(model_config, parallelism_config, dtype);
    auto linear_spec = HybridConfigCreator::createLinearAttentionSpec(model_config, parallelism_config, dtype);

    // Create layer groups and calculate group layer number
    int  group_layer_num = 0;
    auto groups          = HybridConfigCreator::createLayerGroups(
        split.linear_layers, split.swa_layers, split.full_layers, group_layer_num);
    config.group_layer_num = group_layer_num;

    // Setup cache config specs
    HybridConfigCreator::setupCacheConfigSpecs(
        config, groups.linear_groups, groups.swa_groups, groups.full_groups, linear_spec, full_spec);

    // Hard check: current only supports a single full attention group.
    RTP_LLM_CHECK_WITH_INFO(
        config.full_group_num <= 1,
        "Multiple full attention groups (%d) are not supported in hybrid mode. "
        "prepare_fmha_impl is called once before the layer loop, binding the block table from group 0. "
        "To support multiple full groups, implement per-group fmha preparation.",
        config.full_group_num);

    // Setup physical sizes
    HybridConfigCreator::setupPhysicalSizes(config, full_spec, linear_spec);

    // Setup layer to group mapping
    HybridConfigCreator::setupLayerToGroupMapping(config);

    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);
    for (size_t layer_id = 0; layer_id < config.layer_to_group_id.size(); ++layer_id) {
        const int gid = config.layer_to_group_id[layer_id];
        if (gid >= 0 && static_cast<size_t>(gid) < config.group_types.size()) {
            config.layer_group_types[layer_id] = config.group_types[static_cast<size_t>(gid)];
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
