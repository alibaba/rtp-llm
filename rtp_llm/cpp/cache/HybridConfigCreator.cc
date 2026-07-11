#include "rtp_llm/cpp/cache/HybridConfigCreator.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

uint32_t mhaLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp      = static_cast<uint32_t>(attn_tp);
    const uint32_t kv      = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
    RTP_LLM_CHECK_WITH_INFO(kv > 0, "local kv head num requires positive kv_head_num");
    return (kv % tp == 0) ? kv / tp : kv / std::gcd(kv, tp);
}

uint32_t linearLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp      = static_cast<uint32_t>(attn_tp);
    const uint32_t value_heads       = static_cast<uint32_t>(model_config.linear_attention_config.linear_num_value_heads);
    RTP_LLM_CHECK_WITH_INFO(value_heads > 0, "local kv head num requires positive linear_num_value_heads");
    RTP_LLM_CHECK_WITH_INFO(value_heads % tp == 0,
                            "linear_num_value_heads must be divisible by attention TP, global=%u tp=%u",
                            value_heads,
                            tp);
    const uint32_t local_value_heads = value_heads / tp;
    RTP_LLM_CHECK_WITH_INFO(local_value_heads > 0,
                            "invalid local linear value heads: global=%u tp=%u",
                            value_heads,
                            tp);
    return local_value_heads;
}

uint32_t localKvHeadNumForSpec(KVCacheSpecType          type,
                               const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config) {
    switch (type) {
        case KVCacheSpecType::MultiHeadAttention:
            return mhaLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::LinearAttention:
            return linearLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::MultiHeadLatentAttention:
            return 1;
        default:
            RTP_LLM_FAIL("unknown KVCacheSpecType=%d", static_cast<int>(type));
    }
    return 1;
}

CacheGroupType groupTypeForSpec(KVCacheSpecType type) {
    return type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

std::string layoutFingerprint(const KVCacheSpec& spec) {
    std::ostringstream os;
    os << "type=" << static_cast<int>(spec.type) << ";dtype=" << static_cast<int>(spec.memoryLayoutDType())
       << ";seq_size_per_block=" << spec.seq_size_per_block << ";block_elems=" << spec.block_size()
       << ";k_block_elems=" << spec.k_block_size() << ";v_block_elems=" << spec.v_block_size()
       << ";block_bytes=" << spec.block_size_bytes() << ";k_block_bytes=" << spec.k_block_size_bytes()
       << ";v_block_bytes=" << spec.v_block_size_bytes() << ";scale_block_bytes=" << spec.scale_block_size_bytes()
       << ";k_scale_block_bytes=" << spec.k_scale_block_size_bytes()
       << ";v_scale_block_bytes=" << spec.v_scale_block_size_bytes();
    return os.str();
}

const KVCacheSpecPtr& singleRuntimeSpecForLayer(const LayerKVCacheSpecs& runtime_specs, int layer_id) {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < runtime_specs.size(),
                            "missing runtime kv_cache specs for layer %d",
                            layer_id);
    const auto& layer_specs = runtime_specs[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == 1,
                            "hybrid layer %d must have exactly one runtime kv_cache spec, got %zu",
                            layer_id,
                            layer_specs.size());
    RTP_LLM_CHECK_WITH_INFO(layer_specs[0] != nullptr, "hybrid layer %d has null kv_cache spec", layer_id);
    RTP_LLM_CHECK_WITH_INFO(!layer_specs[0]->tag.empty(), "hybrid layer %d has empty kv_cache spec tag", layer_id);
    return layer_specs[0];
}

std::vector<GroupBase> buildTaggedGroups(const LayerKVCacheSpecs& runtime_specs,
                                         const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config) {
    const int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(layer_num),
                            "runtime kv_cache specs size %zu != num_layers %ld",
                            runtime_specs.size(),
                            layer_num);
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            layer_num);

    std::vector<GroupBase> groups;
    for (int layer_id = 0; layer_id < static_cast<int>(layer_num); ++layer_id) {
        const auto& spec          = singleRuntimeSpecForLayer(runtime_specs, layer_id);
        const auto  group_type    = groupTypeForSpec(spec->type);
        const auto  expected_type = types[static_cast<size_t>(layer_id)] == HybridAttentionType::LINEAR ?
                                        CacheGroupType::LINEAR :
                                        CacheGroupType::FULL;
        RTP_LLM_CHECK_WITH_INFO(group_type == expected_type,
                                "hybrid layer %d desc tag=%s cache type %d does not match attention type %d",
                                layer_id,
                                spec->tag.c_str(),
                                static_cast<int>(group_type),
                                static_cast<int>(expected_type));

        auto it = std::find_if(
            groups.begin(), groups.end(), [&](const GroupBase& group) { return group.spec->tag == spec->tag; });
        if (it == groups.end()) {
            GroupBase group;
            group.spec              = spec;
            group.policy            = defaultCacheGroupPolicy(group_type);
            group.local_kv_head_num = localKvHeadNumForSpec(spec->type, model_config, parallelism_config);
            group.layer_ids.push_back(layer_id);
            groups.push_back(std::move(group));
            continue;
        }

        RTP_LLM_CHECK_WITH_INFO(it->policy.group_type == group_type,
                                "hybrid tag=%s maps to multiple cache group types",
                                spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(layoutFingerprint(*it->spec) == layoutFingerprint(*spec),
                                "hybrid tag=%s maps to different kv cache spec layouts",
                                spec->tag.c_str());
        it->layer_ids.push_back(layer_id);
    }

    std::stable_partition(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::FULL;
    });
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "hybrid config requires at least one cache group");
    const bool has_full_group = std::any_of(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::FULL;
    });
    if (has_full_group
        && (groups[0].policy.group_type != CacheGroupType::FULL || groups[0].spec == nullptr
            || groups[0].spec->tag != "full")) {
        RTP_LLM_LOG_WARNING("hybrid full cache group is expected at gid 0 with tag=full, got tag=%s type=%d",
                            groups[0].spec == nullptr ? "<null>" : groups[0].spec->tag.c_str(),
                            static_cast<int>(groups[0].policy.group_type));
    }
    return groups;
}

KVCacheSpecPtr representativeSpec(const std::vector<GroupBase>& groups, CacheGroupType group_type) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (const auto& group : groups) {
        if (group.policy.group_type != group_type) {
            continue;
        }
        if (result == nullptr) {
            result      = group.spec;
            fingerprint = layoutFingerprint(*group.spec);
        } else {
            RTP_LLM_CHECK_WITH_INFO(fingerprint == layoutFingerprint(*group.spec),
                                    "hybrid %d cache groups have different kv cache spec layouts",
                                    static_cast<int>(group_type));
        }
    }
    return result == nullptr ? nullptr : result->clone();
}

int groupLayerNumForGroups(const std::vector<GroupBase>& groups) {
    size_t group_layer_num = 0;
    for (const auto& group : groups) {
        group_layer_num = std::max(group_layer_num, group.layer_ids.size());
    }
    RTP_LLM_CHECK_WITH_INFO(group_layer_num > 0, "hybrid cache groups must not be empty");
    return static_cast<int>(group_layer_num);
}

void setupTopologyFromGroups(CacheConfig& config, std::vector<GroupBase> groups) {
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));

    for (size_t gid = 0; gid < groups.size(); ++gid) {
        const auto& group = groups[gid];
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "hybrid tag=%s has invalid layer id %d",
                                    group.spec->tag.c_str(),
                                    layer_id);
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(static_cast<int>(gid));
            layer.tag_to_gid[group.spec->tag] = static_cast<int>(gid);
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
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
    int group_layer_num = 0;
    if (linear_layer_count > 0 && full_layer_count > 0) {
        group_layer_num = std::gcd(linear_layer_count, full_layer_count);
        if (group_layer_num < full_layer_count) {
            group_layer_num = full_layer_count;
        }
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
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            layer_num);
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
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;

    return config;
}

KVCacheSpecPtr HybridConfigCreator::getSpecFromLayers(const LayerKVCacheSpecs& runtime_specs,
                                                      const std::vector<int>&  layer_ids,
                                                      const char*              spec_role) {
    KVCacheSpecPtr result;
    std::string    fingerprint;
    for (int layer_id : layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(layer_id) < runtime_specs.size()
                                    && !runtime_specs[static_cast<size_t>(layer_id)].empty(),
                                "missing runtime kv_cache specs for %s layer %d",
                                spec_role,
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(runtime_specs[static_cast<size_t>(layer_id)].size() == 1,
                                "%s layer %d must have exactly one runtime kv_cache spec, got %zu",
                                spec_role,
                                layer_id,
                                runtime_specs[static_cast<size_t>(layer_id)].size());
        const auto& spec = runtime_specs[static_cast<size_t>(layer_id)][0];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "%s layer %d has null kv_cache spec", spec_role, layer_id);
        if (result == nullptr) {
            result      = spec;
            fingerprint = spec->fingerprint();
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                fingerprint == spec->fingerprint(), "%s layers have different kv_cache spec fingerprints", spec_role);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "no %s layers found", spec_role);
    return result->clone();
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
                                                const KVCacheSpecPtr&                full_spec,
                                                uint32_t                             linear_local_kv_head_num,
                                                uint32_t                             full_local_kv_head_num) {
    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));

    auto append_group = [&](const KVCacheSpecPtr& spec,
                            CacheGroupType         type,
                            const std::vector<int>& layer_ids,
                            uint32_t               local_kv_head_num) {
        GroupBase group;
        group.spec              = spec;
        group.policy            = defaultCacheGroupPolicy(type);
        group.layer_ids         = layer_ids;
        group.local_kv_head_num = local_kv_head_num;
        const int gid   = static_cast<int>(groups.size());
        groups.push_back(group);
        for (int layer_id : layer_ids) {
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(gid);
            layer.tag_to_gid[spec->tag] = gid;
        }
    };

    // Keep order: all full groups first, then linear groups.
    for (const auto& g : full_groups) {
        append_group(full_spec, CacheGroupType::FULL, g, full_local_kv_head_num);
    }
    for (const auto& g : linear_groups) {
        append_group(linear_spec, CacheGroupType::LINEAR, g, linear_local_kv_head_num);
    }
    config.setTopology(std::move(groups), std::move(layers));
}

void HybridConfigCreator::setupPhysicalSizes(CacheConfig&          config,
                                             const KVCacheSpecPtr& full_spec,
                                             const KVCacheSpecPtr& linear_spec) {
    RTP_LLM_CHECK_WITH_INFO(full_spec != nullptr || linear_spec != nullptr,
                            "hybrid config requires at least one cache spec");

    const size_t full_kv_block_stride_bytes = full_spec == nullptr ? 0 : full_spec->block_size_bytes();
    const size_t linear_kv_block_stride_bytes = linear_spec == nullptr ? 0 : linear_spec->block_size_bytes();

    if (full_spec != nullptr && linear_spec != nullptr) {
        // now we only support that linear attention block have padding
        RTP_LLM_CHECK_WITH_INFO(full_kv_block_stride_bytes >= linear_kv_block_stride_bytes,
                                "not support full attention with padding now");
    }

    const auto& physical_spec = full_spec != nullptr ? full_spec : linear_spec;
    config.kv_block_stride_bytes = std::max(full_kv_block_stride_bytes, linear_kv_block_stride_bytes);
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = physical_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
}

CacheConfig HybridConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp,
                                                    int                      gen_num_per_cycle) {
    (void)is_mtp;

    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto tokens_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(tokens_per_block > 0, "hybrid seq_size_per_block must be > 0");
    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    // Split layers by attention type
    auto [linear_layers, full_layers] = HybridConfigCreator::splitLayersByAttentionType(model_config);

    // Initialize config
    CacheConfig config        = HybridConfigCreator::initializeConfig(model_config, linear_layers, full_layers, dtype);
    config.seq_size_per_block = tokens_per_block;

    auto cache_groups = buildTaggedGroups(runtime_specs, model_config, parallelism_config);
    auto full_spec    = representativeSpec(cache_groups, CacheGroupType::FULL);
    auto linear_spec  = representativeSpec(cache_groups, CacheGroupType::LINEAR);

    config.group_layer_num = groupLayerNumForGroups(cache_groups);
    setupTopologyFromGroups(config, std::move(cache_groups));

    // Setup physical sizes
    HybridConfigCreator::setupPhysicalSizes(config, full_spec, linear_spec);

    // Per-layer block stride (kv + scale).
    // For hybrid attention, the physical per-layer stride follows the selected physical layout stride.
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

}  // namespace rtp_llm