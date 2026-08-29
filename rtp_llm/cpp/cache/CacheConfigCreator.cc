#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool blockNumFitsBudget(uint32_t block_num, size_t total_budget_bytes, const KVCacheBlockBudget& budget, int step) {
    if (budget.explicit_pool_reserve_bytes > total_budget_bytes) {
        return false;
    }

    size_t remaining = total_budget_bytes - budget.explicit_pool_reserve_bytes;
    if (budget.paged_block_bytes > 0) {
        if (static_cast<size_t>(block_num) > remaining / budget.paged_block_bytes) {
            return false;
        }
        remaining -= static_cast<size_t>(block_num) * budget.paged_block_bytes;
    }

    const auto safe_step  = static_cast<uint32_t>(std::max(1, step));
    const auto swa_blocks = block_num / safe_step + (block_num % safe_step != 0 ? 1u : 0u);
    return budget.swa_block_bytes == 0 || static_cast<size_t>(swa_blocks) <= remaining / budget.swa_block_bytes;
}

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    if (!config.use_independent_block_pools) {
        budget.paged_block_bytes = config.block_size_bytes;
        return budget;
    }

    budget.explicit_pool_reserve_bytes = config.explicitly_sized_pool_reserve_bytes;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.usesExplicitIndependentBlocks(gid)) {
            continue;
        }
        const auto group_bytes = config.blockSizeBytesForGroup(gid);
        switch (config.typeForGroup(gid)) {
            case CacheGroupType::FULL:
            case CacheGroupType::LINEAR:
                budget.paged_block_bytes += group_bytes;
                break;
            case CacheGroupType::SWA:
                budget.swa_block_bytes += group_bytes;
                break;
        }
    }
    return budget;
}

void addBlockBudget(KVCacheBlockBudget& total, const KVCacheBlockBudget& addition, size_t multiplier = 1) {
    const auto add = [multiplier](size_t& dst, size_t value, const char* name) {
        RTP_LLM_CHECK_WITH_INFO(multiplier == 0 || value <= (std::numeric_limits<size_t>::max() - dst) / multiplier,
                                "kv cache %s budget overflow: current=%zu addition=%zu multiplier=%zu",
                                name,
                                dst,
                                value,
                                multiplier);
        dst += value * multiplier;
    };
    add(total.explicit_pool_reserve_bytes, addition.explicit_pool_reserve_bytes, "explicit reserve");
    add(total.paged_block_bytes, addition.paged_block_bytes, "paged block bytes");
    add(total.swa_block_bytes, addition.swa_block_bytes, "SWA block bytes");
}

void setupKernelSeqSize(CacheConfig& config, const KVCacheConfig& kv_cache_config, const char* config_name) {
    const auto previous_kernel_seq_size_per_block = config.kernel_seq_size_per_block;
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const auto kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block % kernel_seq_size_per_block == 0,
                                "%s seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                config_name,
                                config.seq_size_per_block,
                                kernel_seq_size_per_block);
        config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    } else if (config.kernel_seq_size_per_block == 0 || config.kernel_seq_size_per_block == config.seq_size_per_block) {
        config.kernel_seq_size_per_block = config.seq_size_per_block;
    }

    if (config.kernel_seq_size_per_block == previous_kernel_seq_size_per_block || config.groupNums() == 0) {
        return;
    }

    auto groups           = config.topology().groups();
    bool topology_changed = false;
    for (auto& group : groups) {
        const auto expected_kernel_seq_size_per_block =
            group.policy.group_type == CacheGroupType::FULL && config.kernel_seq_size_per_block > 0 ?
                std::min(config.kernel_seq_size_per_block, group.seq_size_per_block) :
                group.seq_size_per_block;
        if (group.kernel_seq_size_per_block != expected_kernel_seq_size_per_block) {
            group.kernel_seq_size_per_block = expected_kernel_seq_size_per_block;
            topology_changed                = true;
        }
    }
    if (topology_changed) {
        config.setTopology(std::move(groups), config.topology().layers());
    }
}

uint32_t computeBlockNum(CacheConfig&                                     config,
                         const ModelConfig&                               model_config,
                         const RuntimeConfig&                             runtime_config,
                         const KVCacheConfig&                             kv_cache_config,
                         const ParallelismConfig&                         parallelism_config,
                         const std::optional<WarmUpResult>&               warm_up_result,
                         const std::optional<SpeculativeExecutionConfig>& sp_config) {
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        config.finalizeBlockNums(kv_cache_config.test_block_num, runtime_config);
        return static_cast<uint32_t>(kv_cache_config.test_block_num);
    }

    const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    config.finalizeBlockNums(0, runtime_config);

    const auto block_budget = blockBudgetForConfig(config);
    if (block_budget.explicit_pool_reserve_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > block_budget.explicit_pool_reserve_bytes,
                                "kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                "(reduce explicitly sized pool blocks if needed)",
                                kv_cache_mem_size / 1024 / 1024,
                                block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
        RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB",
                         kv_cache_mem_size / 1024 / 1024,
                         block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
    }
    return maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, config.linear_step);
}

uint32_t mhaLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp      = static_cast<uint32_t>(attn_tp);
    const uint32_t kv      = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
    RTP_LLM_CHECK_WITH_INFO(kv > 0, "local kv head num requires positive kv_head_num");
    return (kv % tp == 0) ? kv / tp : kv / std::gcd(kv, tp);
}

uint32_t linearLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp     = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp          = static_cast<uint32_t>(attn_tp);
    const uint32_t value_heads = static_cast<uint32_t>(model_config.linear_attention_config.linear_num_value_heads);
    RTP_LLM_CHECK_WITH_INFO(value_heads > 0, "local kv head num requires positive linear_num_value_heads");
    RTP_LLM_CHECK_WITH_INFO(value_heads % tp == 0,
                            "linear_num_value_heads must be divisible by attention TP, global=%u tp=%u",
                            value_heads,
                            tp);
    const uint32_t local_value_heads = value_heads / tp;
    RTP_LLM_CHECK_WITH_INFO(
        local_value_heads > 0, "invalid local linear value heads: global=%u tp=%u", value_heads, tp);
    return local_value_heads;
}

uint32_t localKvHeadNumForType(KVCacheSpecType          type,
                               const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config) {
    switch (type) {
        case KVCacheSpecType::MultiHeadAttention:
            return mhaLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::LinearAttention:
            return linearLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::OpaqueKV:
        case KVCacheSpecType::OpaqueState:
            return 1;
        default:
            RTP_LLM_FAIL("unknown KVCacheSpecType=%d", static_cast<int>(type));
    }
    return 1;
}

KVCacheSpecPtr getSingleSpecFromLayers(const ModelConfig& model_config, const LayerKVCacheSpecs& runtime_specs) {
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "single cache config requires layer-wise runtime specs for every layer, got %zu/%ld",
                            runtime_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(!runtime_specs.empty(), "single cache config requires at least one runtime spec");
    RTP_LLM_CHECK_WITH_INFO(runtime_specs[0].size() == 1,
                            "single cache config requires exactly one spec for layer 0, got %zu",
                            runtime_specs[0].size());
    auto spec = runtime_specs[0][0];
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "single cache config got null runtime spec for layer 0");
    const auto& expected_tag = spec->tag;
    const auto  fingerprint  = spec->fingerprint();
    for (int64_t layer_id = 1; layer_id < model_config.num_layers; ++layer_id) {
        const auto layer = static_cast<size_t>(layer_id);
        RTP_LLM_CHECK_WITH_INFO(runtime_specs[layer].size() == 1,
                                "single cache config requires exactly one spec for layer %ld, got %zu",
                                layer_id,
                                runtime_specs[layer].size());
        const auto& layer_spec = runtime_specs[layer][0];
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec != nullptr, "single cache config got null runtime spec for layer %ld", layer_id);
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec->tag == expected_tag,
            "single cache config requires consistent tag across layers, layer %ld has tag=%s but layer 0 has tag=%s",
            layer_id,
            layer_spec->tag.c_str(),
            expected_tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            layer_spec->fingerprint() == fingerprint, "single cache config spec differs at layer %ld", layer_id);
    }
    return spec->clone();
}

CacheConfig createSingleConfig(const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config,
                               int                      gen_num_per_cycle) {
    const auto dtype            = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto tokens_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(tokens_per_block > 0, "single seq_size_per_block must be > 0");

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    const auto runtime_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.is_sparse          = model_config.attn_config.is_sparse;

    auto             spec = getSingleSpecFromLayers(model_config, runtime_specs);
    std::vector<int> layer_ids(static_cast<size_t>(model_config.num_layers));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    GroupBase group;
    group.tag       = spec->tag;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR :
                                                                                            CacheGroupType::FULL);
    group.layer_ids = layer_ids;
    group.local_kv_head_num = localKvHeadNumForType(spec->type, model_config, parallelism_config);

    std::vector<LayerBase> layers(static_cast<size_t>(model_config.num_layers));
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        layers[static_cast<size_t>(layer_id)] = {static_cast<int>(layer_id), {spec->tag}};
    }
    config.setTopology({group}, std::move(layers));
    RTP_LLM_CHECK_WITH_INFO(config.groupNums() == 1, "single config expected one cache group");

    config.kv_block_stride_bytes        = spec->block_size_bytes();
    config.kv_block_size_bytes          = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes        = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes          = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes             = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.group_layer_num              = static_cast<int>(model_config.num_layers);
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    auto groups                     = config.topology().groups();
    groups[0].kv_block_stride_bytes = config.kv_block_stride_bytes;
    groups[0].kv_scale_stride_bytes = config.kv_scale_stride_bytes;
    config.setTopology(std::move(groups), config.topology().layers());
    return config;
}

std::string hybridLayoutFingerprint(const KVCacheSpec& spec) {
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

const KVCacheSpecPtr& hybridSpecForLayer(const LayerKVCacheSpecs& runtime_specs, int layer_id) {
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

std::vector<GroupBase> buildHybridGroups(const LayerKVCacheSpecs& runtime_specs,
                                         const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config) {
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(
        model_config.num_layers > 0, "invalid model_config.num_layers=%ld", model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(runtime_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "runtime kv_cache specs size %zu != num_layers %ld",
                            runtime_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);

    std::vector<GroupBase> groups;
    for (int layer_id = 0; layer_id < static_cast<int>(model_config.num_layers); ++layer_id) {
        const auto& spec = hybridSpecForLayer(runtime_specs, layer_id);
        const auto  group_type =
            spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
        const auto expected_type = types[static_cast<size_t>(layer_id)] == HybridAttentionType::LINEAR ?
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
            group.tag               = spec->tag;
            group.spec              = spec;
            group.policy            = defaultCacheGroupPolicy(group_type);
            group.local_kv_head_num = localKvHeadNumForType(spec->type, model_config, parallelism_config);
            group.layer_ids.push_back(layer_id);
            groups.push_back(std::move(group));
        } else {
            RTP_LLM_CHECK_WITH_INFO(it->policy.group_type == group_type,
                                    "hybrid tag=%s maps to multiple cache group types",
                                    spec->tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(hybridLayoutFingerprint(*it->spec) == hybridLayoutFingerprint(*spec),
                                    "hybrid tag=%s maps to different kv cache spec layouts",
                                    spec->tag.c_str());
            it->layer_ids.push_back(layer_id);
        }
    }
    std::stable_partition(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::FULL;
    });
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "hybrid config requires at least one cache group");
    const auto full_group_num = std::count_if(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::FULL;
    });
    RTP_LLM_CHECK_WITH_INFO(
        full_group_num <= 1,
        "multiple full attention cache groups (%zu) are not supported: FMHA parameters bind one block table before "
        "the layer loop",
        static_cast<size_t>(full_group_num));
    if (full_group_num != 0
        && (groups[0].policy.group_type != CacheGroupType::FULL || groups[0].spec == nullptr
            || groups[0].spec->tag != "full")) {
        RTP_LLM_LOG_WARNING("hybrid full cache group is expected at gid 0 with tag=full, got tag=%s type=%d",
                            groups[0].spec == nullptr ? "<null>" : groups[0].spec->tag.c_str(),
                            static_cast<int>(groups[0].policy.group_type));
    }
    return groups;
}

int hybridGroupLayerNum(const ModelConfig& model_config) {
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    const int linear_count = static_cast<int>(std::count(types.begin(), types.end(), HybridAttentionType::LINEAR));
    const int full_count   = static_cast<int>(types.size()) - linear_count;
    if (linear_count > 0 && full_count > 0) {
        return std::max(std::gcd(linear_count, full_count), full_count);
    }
    return std::max({linear_count, full_count, 1});
}

void setupHybridTopology(CacheConfig& config, std::vector<GroupBase> groups) {
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    for (const auto& group : groups) {
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "hybrid tag=%s has invalid layer id %d",
                                    group.spec->tag.c_str(),
                                    layer_id);
            layers[static_cast<size_t>(layer_id)].group_tags.push_back(group.tag);
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
}

void setupHybridPoolSizes(CacheConfig& config) {
    config.use_independent_block_pools = true;
    const auto            group_num    = static_cast<size_t>(config.groupNums());
    std::vector<uint32_t> group_block_nums(group_num, 0);
    std::vector<size_t>   group_kv_block_stride_bytes(group_num, 0);
    std::vector<size_t>   group_kv_scale_stride_bytes(group_num, 0);
    size_t                max_kv_stride           = 0;
    size_t                max_scale_stride        = 0;
    size_t                total_kv_block_bytes    = 0;
    size_t                total_scale_block_bytes = 0;
    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = config.specForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count         = static_cast<uint32_t>(config.layerIdsForGroup(gid).size());
        const size_t group_bpk           = config.kernelBlocksPerKvBlockForGroup(gid);
        const size_t kv_stride           = spec->block_size_bytes() * group_bpk;
        const size_t scale_stride        = spec->scale_block_size_bytes() * group_bpk;
        group_kv_block_stride_bytes[gid] = kv_stride;
        group_kv_scale_stride_bytes[gid] = scale_stride;
        total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
        total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        for (int layer_id : config.layerIdsForGroup(gid)) {
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] =
                static_cast<int>(kv_stride + scale_stride);
        }
    }
    config.kv_block_stride_bytes               = max_kv_stride;
    config.kv_scale_stride_bytes               = max_scale_stride;
    config.kv_block_size_bytes                 = total_kv_block_bytes;
    config.kv_scale_size_bytes                 = total_scale_block_bytes;
    config.block_size_bytes                    = total_kv_block_bytes + total_scale_block_bytes;
    config.explicitly_sized_pool_reserve_bytes = 0;
    config.setGroupBlockLayout(group_block_nums, group_kv_block_stride_bytes, group_kv_scale_stride_bytes);
}

CacheConfig createHybridConfig(const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config,
                               int                      gen_num_per_cycle) {
    const auto dtype            = MemoryEvaluationHelper::getDataTypeForCache(model_config);
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

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.group_layer_num    = hybridGroupLayerNum(model_config);
    setupHybridTopology(config, buildHybridGroups(runtime_specs, model_config, parallelism_config));
    setupHybridPoolSizes(config);
    return config;
}

void validateIndependentDescs(const ModelConfig& model_config,
                              uint32_t           kernel_tokens_per_block,
                              int                gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
        "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
        model_config.kv_cache_spec_descs.size(),
        model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "hybrid-pool desc config requires non-negative gen_num_per_cycle, got %d",
                            gen_num_per_cycle);
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "hybrid-pool desc config layer %ld has no descs", layer_id);
        for (const auto& desc : descs) {
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                RTP_LLM_CHECK_WITH_INFO(
                    desc.compression_ratio > 0,
                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                    desc.tag.c_str(),
                    desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(
                    kernel_tokens_per_block > 0,
                    "desc tag=%s derives entries from kernel block but kernel_tokens_per_block is 0",
                    desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_tokens_per_block);
                RTP_LLM_CHECK_WITH_INFO(desc.kernel_tokens_per_block_alignment > 0,
                                        "desc tag=%s has invalid kernel_tokens_per_block_alignment=0",
                                        desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(
                    kernel_tokens_per_block >= desc.kernel_tokens_per_block_alignment
                        && kernel_tokens_per_block % desc.kernel_tokens_per_block_alignment == 0,
                    "desc tag=%s derives entries from kernel block, so kernel_seq_size_per_block(%u) "
                    "must be >= %u and a multiple of %u",
                    desc.tag.c_str(),
                    kernel_tokens_per_block,
                    desc.kernel_tokens_per_block_alignment,
                    desc.kernel_tokens_per_block_alignment);
            }
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "state ring desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

void populateIndependentGroups(CacheConfig&                 config,
                               const LayerKVCacheSpecDescs& layer_descs,
                               const LayerKVCacheSpecs&     layer_specs,
                               const ModelConfig&           model_config,
                               const ParallelismConfig&     parallelism_config) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(config.layer_num),
                            "hybrid-pool layer desc count %zu != layer_num %u",
                            layer_descs.size(),
                            config.layer_num);
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(config.layer_num),
                            "hybrid-pool layer spec count %zu != layer_num %u",
                            layer_specs.size(),
                            config.layer_num);
    struct GroupBuildState {
        KVCacheSpecPtr   spec;
        std::string      fingerprint;
        CacheGroupType   type;
        CacheGroupPolicy policy;
        uint32_t         local_kv_head_num = 1;
        std::vector<int> layer_ids;
    };
    std::map<std::string, GroupBuildState> group_by_tag;
    std::vector<std::string>               ordered_tags;
    for (uint32_t layer = 0; layer < config.layer_num; ++layer) {
        const auto& descs = layer_descs[layer];
        const auto& specs = layer_specs[layer];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "hybrid-pool layer %u has no descs", layer);
        RTP_LLM_CHECK_WITH_INFO(descs.size() == specs.size(),
                                "hybrid-pool layer %u desc count %zu != spec count %zu",
                                layer,
                                descs.size(),
                                specs.size());
        std::set<std::string> layer_tags;
        for (size_t idx = 0; idx < descs.size(); ++idx) {
            const auto& desc = descs[idx];
            const auto& spec = specs[idx];
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool layer %u has null spec", layer);
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(spec->tag).second,
                                    "hybrid-pool layer %u has duplicate tag=%s",
                                    layer,
                                    spec->tag.c_str());
            const auto policy            = SpecBuilder::groupPolicy(desc);
            const auto type              = SpecBuilder::groupType(desc);
            const auto local_kv_head_num = localKvHeadNumForType(desc.cache_type, model_config, parallelism_config);
            auto       group_it          = group_by_tag.find(spec->tag);
            if (group_it == group_by_tag.end()) {
                GroupBuildState state;
                state.spec              = spec;
                state.fingerprint       = spec->fingerprint();
                state.type              = type;
                state.policy            = policy;
                state.local_kv_head_num = local_kv_head_num;
                group_it                = group_by_tag.emplace(spec->tag, std::move(state)).first;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(group_it->second.fingerprint == spec->fingerprint(),
                                        "hybrid-pool tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(
                    group_it->second.type == type, "hybrid-pool tag=%s has inconsistent group type", spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group_it->second.policy, policy),
                                        "hybrid-pool tag=%s has inconsistent policy",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group_it->second.local_kv_head_num == local_kv_head_num,
                                        "hybrid-pool tag=%s has inconsistent local_kv_head_num",
                                        spec->tag.c_str());
            }
            group_it->second.layer_ids.push_back(static_cast<int>(layer));
        }
    }
    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers(static_cast<size_t>(config.layer_num));
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto& state = group_by_tag.at(tag);
        GroupBase   group;
        group.tag               = tag;
        group.spec              = state.spec;
        group.policy            = state.policy;
        group.layer_ids         = state.layer_ids;
        group.local_kv_head_num = state.local_kv_head_num;
        groups.push_back(group);
        for (int layer_id : state.layer_ids) {
            layers[static_cast<size_t>(layer_id)].group_tags.push_back(tag);
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto            group_num    = static_cast<size_t>(config.groupNums());
    std::vector<uint32_t> group_block_nums(group_num, 0);
    std::vector<size_t>   group_kv_block_stride_bytes(group_num, 0);
    std::vector<size_t>   group_kv_scale_stride_bytes(group_num, 0);
    size_t                max_kv_stride           = 0;
    size_t                max_scale_stride        = 0;
    size_t                total_kv_block_bytes    = 0;
    size_t                total_scale_block_bytes = 0;
    uint32_t              max_group_layers        = 0;
    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = config.specForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count         = static_cast<uint32_t>(config.layerIdsForGroup(gid).size());
        const size_t group_bpk           = config.kernelBlocksPerKvBlockForGroup(gid);
        const size_t kv_stride           = spec->block_size_bytes() * group_bpk;
        const size_t scale_stride        = spec->scale_block_size_bytes() * group_bpk;
        group_kv_block_stride_bytes[gid] = kv_stride;
        group_kv_scale_stride_bytes[gid] = scale_stride;
        const auto type                  = config.typeForGroup(gid);
        const bool is_paged_group        = type == CacheGroupType::FULL || type == CacheGroupType::LINEAR;
        if (is_paged_group && !config.usesExplicitIndependentBlocks(gid)) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);
        for (int layer_id : config.layerIdsForGroup(gid)) {
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] =
                static_cast<int>(kv_stride + scale_stride);
        }
    }
    config.group_layer_num         = static_cast<int>(std::max<uint32_t>(1, max_group_layers));
    config.kv_block_stride_bytes   = max_kv_stride;
    config.kv_scale_stride_bytes   = max_scale_stride;
    config.kv_block_size_bytes     = total_kv_block_bytes;
    config.kv_scale_size_bytes     = total_scale_block_bytes;
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
    config.explicitly_sized_pool_reserve_bytes = 0;
    config.setGroupBlockLayout(group_block_nums, group_kv_block_stride_bytes, group_kv_scale_stride_bytes);
}

CacheConfig createIndependentConfig(const ModelConfig&       model_config,
                                    const ParallelismConfig& parallelism_config,
                                    const KVCacheConfig&     kv_cache_config,
                                    bool                     is_mtp,
                                    int                      gen_num_per_cycle) {
    const auto    dtype                  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    constexpr int kDefaultKvCacheSeqSize = 64;
    const bool    has_seq_override =
        kv_cache_config.seq_size_per_block > 0 && kv_cache_config.seq_size_per_block != kDefaultKvCacheSeqSize;
    const auto physical_tokens_per_block = has_seq_override ?
                                               static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                               static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block   = kv_cache_config.kernel_seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                               physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "hybrid-pool seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "hybrid-pool kernel_seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(
        physical_tokens_per_block >= kernel_tokens_per_block
            && physical_tokens_per_block % kernel_tokens_per_block == 0,
        "hybrid-pool seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
        physical_tokens_per_block,
        kernel_tokens_per_block);

    CacheConfig config;
    config.layer_num                 = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num             = config.layer_num;
    config.block_num                 = 0;
    config.seq_size_per_block        = physical_tokens_per_block;
    config.kernel_seq_size_per_block = kernel_tokens_per_block;
    config.use_mla                   = model_config.attn_config.use_mla;
    config.dtype                     = dtype;
    config.linear_step               = 1;
    config.is_sparse                 = model_config.attn_config.is_sparse;

    if (!model_config.kv_cache_spec_descs.empty()) {
        validateIndependentDescs(model_config, kernel_tokens_per_block, gen_num_per_cycle);
        SpecBuildContext ctx;
        ctx.dtype                   = dtype;
        ctx.seq_size_per_block      = physical_tokens_per_block;
        ctx.attn_config             = &model_config.attn_config;
        ctx.linear_attention_config = &model_config.linear_attention_config;
        ctx.parallelism_config      = &parallelism_config;
        ctx.kernel_tokens_per_block = kernel_tokens_per_block;
        ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
        const auto layer_specs      = CacheConfigCreator::buildLayerSpecsFromDescs(
            model_config.kv_cache_spec_descs, ctx, model_config.num_layers);
        populateIndependentGroups(
            config, model_config.kv_cache_spec_descs, layer_specs, model_config, parallelism_config);
        for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
            const auto& spec               = config.specForGroup(gid);
            config.use_typed_cache_regions = config.use_typed_cache_regions || spec->type == KVCacheSpecType::OpaqueKV
                                             || spec->type == KVCacheSpecType::OpaqueState;
            config.use_opaque_kv_cache_store = config.use_opaque_kv_cache_store
                                               || spec->type == KVCacheSpecType::OpaqueKV
                                               || spec->type == KVCacheSpecType::OpaqueState;
        }
        for (const auto& layer_descs : model_config.kv_cache_spec_descs) {
            for (const auto& desc : layer_descs) {
                config.is_sparse = config.is_sparse || desc.cache_type == KVCacheSpecType::OpaqueKV;
            }
        }
        config.disable_decode_first_malloc_device_reuse =
            config.disable_decode_first_malloc_device_reuse || config.use_opaque_kv_cache_store;
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "CacheConfigCreator requires kv_cache_spec_descs");
    }
    RTP_LLM_CHECK_WITH_INFO(config.groupNums() > 0, "hybrid-pool config produced no cache specs");
    setupIndependentPoolSizes(config, is_mtp);
    return config;
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");

    uint32_t low  = 0;
    uint32_t high = std::numeric_limits<uint32_t>::max();
    while (low < high) {
        const uint32_t mid = low + static_cast<uint32_t>((static_cast<uint64_t>(high) - low + 1) / 2);
        if (blockNumFitsBudget(mid, total_budget_bytes, budget, linear_step)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

LayerKVCacheSpecs CacheConfigCreator::buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                               const SpecBuildContext&      ctx,
                                                               int64_t                      expected_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(expected_layer_num),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            layer_descs.size(),
                            expected_layer_num);
    LayerKVCacheSpecs layer_specs(layer_descs.size());
    for (size_t layer_id = 0; layer_id < layer_descs.size(); ++layer_id) {
        const auto& descs = layer_descs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "kv_cache_spec_descs layer %zu has no descs", layer_id);
        auto& specs = layer_specs[layer_id];
        specs.reserve(descs.size());
        for (const auto& desc : descs) {
            specs.push_back(SpecBuilder::build(desc, ctx));
        }
    }
    return layer_specs;
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    CacheConfig config;
    if (model_config.hybrid_attention_config.enable_independent_kv_cache_pools) {
        KVCacheConfig no_override_config;
        no_override_config.seq_size_per_block        = 0;
        no_override_config.kernel_seq_size_per_block = 0;
        config =
            createIndependentConfig(model_config, parallelism_config, no_override_config, is_mtp, gen_num_per_cycle);
    } else if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        config = createHybridConfig(model_config, parallelism_config, gen_num_per_cycle);
    } else {
        config = createSingleConfig(model_config, parallelism_config, gen_num_per_cycle);
    }

    if (!model_config.hybrid_attention_config.enable_independent_kv_cache_pools) {
        const auto full_group_num = std::count_if(
            config.topology().groups().begin(), config.topology().groups().end(), [](const GroupBase& group) {
                return group.policy.group_type == CacheGroupType::FULL && group.spec
                       && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
            });
        RTP_LLM_CHECK_WITH_INFO(full_group_num == 1,
                                "cache config requires exactly one FULL MHA/MLA cache group, got %zu",
                                static_cast<size_t>(full_group_num));
    }
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config = model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
                             createIndependentConfig(model_config, parallelism_config, kv_cache_config, false, 0) :
                             CacheConfigCreator::createBasicConfig(model_config, parallelism_config, false, 0);

    config.linear_step = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "cache");

    uint32_t block_num = computeBlockNum(
        config, model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_num);
    config.finalizeBlockNums(block_num, runtime_config);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_num, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_num,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

CacheConfig CacheConfigCreator::createSpConfig(const ModelConfig&                 score_model_config,
                                               const ModelConfig&                 propose_model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               const std::optional<WarmUpResult>& warm_up_result,
                                               bool                               is_mtp,
                                               bool                               is_eagle) {
    CacheConfig score_config =
        score_model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
            createIndependentConfig(
                score_model_config, parallelism_config, kv_cache_config, false, sp_config.gen_num_per_cycle) :
            CacheConfigCreator::createBasicConfig(
                score_model_config, parallelism_config, false, sp_config.gen_num_per_cycle);
    CacheConfig propose_config =
        propose_model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
            createIndependentConfig(
                propose_model_config, parallelism_config, kv_cache_config, is_mtp, sp_config.gen_num_per_cycle) :
            CacheConfigCreator::createBasicConfig(
                propose_model_config, parallelism_config, is_mtp, sp_config.gen_num_per_cycle);

    const int joint_step       = std::max(1, kv_cache_config.linear_step);
    score_config.linear_step   = joint_step;
    propose_config.linear_step = joint_step;

    setupKernelSeqSize(score_config, kv_cache_config, "score");
    setupKernelSeqSize(propose_config, kv_cache_config, "propose");

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle || sp_config.type == SP_TYPE_DSPARK) {
            // DSpARK is one multi-layer block-draft model; gamma is its
            // proposal width, not a count of independent one-layer modules.
            num_mtp_modules = 1;
        }
    }

    score_config.finalizeBlockNums(0, runtime_config);
    propose_config.finalizeBlockNums(0, runtime_config);

    uint32_t total_layer_num = score_config.layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.layer_num;
    }

    size_t total_block_size_bytes = score_config.block_size_bytes;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.block_size_bytes;
    }

    KVCacheBlockBudget joint_budget = blockBudgetForConfig(score_config);
    addBlockBudget(joint_budget, blockBudgetForConfig(propose_config), static_cast<size_t>(num_mtp_modules));
    const size_t explicit_pool_reserve = joint_budget.explicit_pool_reserve_bytes;

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        if (explicit_pool_reserve > 0) {
            RTP_LLM_CHECK_WITH_INFO(
                kv_cache_mem_size > explicit_pool_reserve,
                "sp kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                "(reduce explicitly sized pool blocks if needed)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024);
            RTP_LLM_LOG_INFO(
                "sp kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB (score=%zu MiB + propose=%zu MiB x %d)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024,
                score_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                propose_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                num_mtp_modules);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, joint_budget, joint_step);
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config                         = score_config;
    config.linear_step                         = joint_step;
    config.layer_all_num                       = score_config.layer_num;
    config.block_size_bytes                    = total_block_size_bytes;
    config.block_num                           = block_num;
    config.explicitly_sized_pool_reserve_bytes = explicit_pool_reserve;

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);

    const size_t score_layers = static_cast<size_t>(main_layer_num);
    RTP_LLM_CHECK_WITH_INFO(score_config.layer_to_block_stride_bytes.size() == score_layers,
                            "score_config.layer_to_block_stride_bytes size mismatch, got=%zu need=%zu",
                            score_config.layer_to_block_stride_bytes.size(),
                            score_layers);
    for (size_t l = 0; l < score_layers; ++l) {
        config.layer_to_block_stride_bytes[l] = score_config.layer_to_block_stride_bytes[l];
    }

    for (int m = 0; m < num_mtp_modules; ++m) {
        RTP_LLM_CHECK_WITH_INFO(propose_config.layer_to_block_stride_bytes.size() == static_cast<size_t>(mtp_layer_num),
                                "sub_cfg.layer_to_block_stride_bytes size mismatch, got=%zu need=%u",
                                propose_config.layer_to_block_stride_bytes.size(),
                                mtp_layer_num);
        auto sub_cfg = config.mergeMTPModule(propose_config, m, main_layer_num);
        sub_cfg->finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    config.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
    config.explicitly_sized_pool_reserve_bytes = explicit_pool_reserve;

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_config.block_size_bytes,
                     num_mtp_modules,
                     propose_config.block_size_bytes);

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }

    return config;
}

}  // namespace rtp_llm
