#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

#include "absl/numeric/int128.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

struct TopologyStorageSummary {
    KVCacheBlockBudget block_budget;
    bool               use_typed_cache_regions   = false;
    bool               use_opaque_kv_cache_store = false;
    bool               is_sparse                 = false;
};

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    budget.explicit_pool_reserve_bytes = config.explicitlySizedPoolReserveBytes();
    budget.paged_block_bytes           = config.pagedBlockSizeBytes();
    budget.swa_block_bytes             = config.swaBlockSizeBytes();
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

uint32_t computeLocalBlockNum(const KVCacheBlockBudget&                        block_budget,
                              const ModelConfig&                               model_config,
                              const RuntimeConfig&                             runtime_config,
                              const KVCacheConfig&                             kv_cache_config,
                              const ParallelismConfig&                         parallelism_config,
                              const std::optional<WarmUpResult>&               warm_up_result,
                              const std::optional<SpeculativeExecutionConfig>& sp_config,
                              int                                              linear_step) {
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        return static_cast<uint32_t>(kv_cache_config.test_block_num);
    }

    const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    if (block_budget.explicit_pool_reserve_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > block_budget.explicit_pool_reserve_bytes,
                                "kv cache budget %zu MiB must be greater than explicitly-sized pool reservation %zu "
                                "MiB "
                                "(reduce explicitly sized pool blocks if needed)",
                                kv_cache_mem_size / 1024 / 1024,
                                block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
        RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB",
                         kv_cache_mem_size / 1024 / 1024,
                         block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
    }
    return maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, linear_step);
}

std::pair<uint32_t, uint32_t> resolveSeqSizes(const ModelConfig& model_config, const KVCacheConfig& kv_cache_config) {
    constexpr int kDefaultKvCacheSeqSize = 64;
    const bool    has_seq_override =
        kv_cache_config.seq_size_per_block > 0 && kv_cache_config.seq_size_per_block != kDefaultKvCacheSeqSize;
    const auto seq_size_per_block = has_seq_override ? static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                                       static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_seq_size_per_block = kv_cache_config.kernel_seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                               seq_size_per_block;
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_seq_size_per_block > 0, "cache kernel_seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block >= kernel_seq_size_per_block
                                && seq_size_per_block % kernel_seq_size_per_block == 0,
                            "cache seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
                            seq_size_per_block,
                            kernel_seq_size_per_block);
    return {seq_size_per_block, kernel_seq_size_per_block};
}

SpecBuildContext makeSpecBuildContext(const ModelConfig&       model_config,
                                      const ParallelismConfig& parallelism_config,
                                      uint32_t                 seq_size_per_block,
                                      uint32_t                 kernel_seq_size_per_block,
                                      int                      gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        gen_num_per_cycle >= 0, "gen_num_per_cycle must be non-negative, got %d", gen_num_per_cycle);
    SpecBuildContext ctx;
    ctx.dtype                     = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    ctx.seq_size_per_block        = seq_size_per_block;
    ctx.kernel_seq_size_per_block = kernel_seq_size_per_block;
    ctx.attn_config               = &model_config.attn_config;
    ctx.linear_attention_config   = &model_config.linear_attention_config;
    ctx.parallelism_config        = &parallelism_config;
    ctx.gen_num_per_cycle         = static_cast<uint32_t>(gen_num_per_cycle);
    return ctx;
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

void validateDescs(const ModelConfig& model_config, uint32_t kernel_tokens_per_block, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "cache config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(
        gen_num_per_cycle >= 0, "cache config requires non-negative gen_num_per_cycle, got %d", gen_num_per_cycle);
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "cache config layer %ld has no descs", layer_id);
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
            }
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "state ring desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

CacheTopologyPair populateGroupsFromLayerSpecs(const LayerBuiltSpecs&   layer_specs,
                                               const ModelConfig&       model_config,
                                               const ParallelismConfig& parallelism_config) {
    RTP_LLM_CHECK_WITH_INFO(!layer_specs.empty(), "cache topology requires at least one layer");

    std::map<std::string, CacheGroup> groups_by_tag;
    std::vector<std::string>          ordered_tags;
    std::vector<CacheLayer>           layers(layer_specs.size());
    for (uint32_t layer_id = 0; layer_id < layer_specs.size(); ++layer_id) {
        const auto& specs = layer_specs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!specs.empty(), "cache layer %u has no specs", layer_id);
        std::set<std::string> layer_tags;
        for (const auto& built : specs) {
            RTP_LLM_CHECK_WITH_INFO(!built.tag.empty(), "cache layer %u has empty spec tag", layer_id);
            RTP_LLM_CHECK_WITH_INFO(
                built.spec != nullptr, "cache layer %u tag=%s has null spec", layer_id, built.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(built.tag).second,
                                    "hybrid-pool layer %u has duplicate tag=%s",
                                    layer_id,
                                    built.tag.c_str());

            const auto local_heads = localKvHeadNumForType(built.spec->type, model_config, parallelism_config);
            auto [it, inserted]    = groups_by_tag.emplace(built.tag, CacheGroup{});
            auto& group            = it->second;
            if (inserted) {
                group.tag               = built.tag;
                group.spec              = built.spec;
                group.policy            = built.policy;
                group.local_kv_head_num = local_heads;
                ordered_tags.push_back(built.tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(group.spec->fingerprint() == built.spec->fingerprint(),
                                        "hybrid-pool tag=%s has multiple physical prototypes",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == built.policy.group_type,
                                        "hybrid-pool tag=%s has inconsistent group type",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group.policy, built.policy),
                                        "hybrid-pool tag=%s has inconsistent policy",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group.local_kv_head_num == local_heads,
                                        "hybrid-pool tag=%s has inconsistent local_kv_head_num",
                                        built.tag.c_str());
            }
            layers[layer_id].push_back(built.tag);
        }
    }

    std::vector<CacheGroup> groups;
    groups.reserve(groups_by_tag.size());
    for (const auto& tag : ordered_tags) {
        groups.push_back(std::move(groups_by_tag.at(tag)));
    }
    return {std::move(groups), std::move(layers)};
}

TopologyStorageSummary finalizeGroupStorage(CacheTopologyPair& topology) {
    auto&                                   groups = topology.first;
    auto&                                   layers = topology.second;
    TopologyStorageSummary                  summary;
    std::unordered_map<std::string, size_t> layer_counts;
    for (const auto& layer : layers) {
        for (const auto& tag : layer) {
            ++layer_counts[tag];
        }
    }

    for (auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "cache group tag=%s has null spec", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->kernel_seq_size_per_block > 0
                                    && group.spec->seq_size_per_block % group.spec->kernel_seq_size_per_block == 0,
                                "cache group tag=%s physical block %u must be divisible by kernel block %u",
                                group.tag.c_str(),
                                group.spec->seq_size_per_block,
                                group.spec->kernel_seq_size_per_block);
        group.kv_block_stride_bytes = group.spec->block_size_bytes();
        group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();

        const auto   count_it    = layer_counts.find(group.tag);
        const size_t layer_count = count_it == layer_counts.end() ? 0 : count_it->second;
        RTP_LLM_CHECK_WITH_INFO(group.kv_scale_stride_bytes
                                    <= std::numeric_limits<size_t>::max() - group.kv_block_stride_bytes,
                                "cache group tag=%s stride overflow",
                                group.tag.c_str());
        const size_t group_bytes = group.kv_block_stride_bytes + group.kv_scale_stride_bytes;
        RTP_LLM_CHECK_WITH_INFO(layer_count == 0 || group_bytes <= std::numeric_limits<size_t>::max() / layer_count,
                                "cache group tag=%s layer stride overflow",
                                group.tag.c_str());
        const size_t bytes_per_pool = group_bytes * layer_count;
        if (group.policy.explicit_block_num > 0) {
            const auto explicit_blocks = static_cast<size_t>(group.policy.explicit_block_num);
            RTP_LLM_CHECK_WITH_INFO(explicit_blocks == 0
                                        || bytes_per_pool <= (std::numeric_limits<size_t>::max()
                                                              - summary.block_budget.explicit_pool_reserve_bytes)
                                                                 / explicit_blocks,
                                    "cache group tag=%s explicit reserve bytes overflow",
                                    group.tag.c_str());
            summary.block_budget.explicit_pool_reserve_bytes += explicit_blocks * bytes_per_pool;
        } else if (group.policy.group_type == CacheGroupType::SWA) {
            summary.block_budget.swa_block_bytes += bytes_per_pool;
        } else if (group.policy.group_type == CacheGroupType::FULL
                   || group.policy.group_type == CacheGroupType::LINEAR) {
            summary.block_budget.paged_block_bytes += bytes_per_pool;
        }

        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        summary.use_typed_cache_regions   = summary.use_typed_cache_regions || opaque;
        summary.use_opaque_kv_cache_store = summary.use_opaque_kv_cache_store || opaque;
        summary.is_sparse                 = summary.is_sparse || group.spec->type == KVCacheSpecType::OpaqueKV;
    }
    return summary;
}

struct BuiltConfigData {
    CacheTopologyPair      topology;
    TopologyStorageSummary storage;
};

BuiltConfigData buildConfigDataFromDescs(const ModelConfig& model_config, const SpecBuildContext& ctx) {
    RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr, "cache spec build context requires parallelism config");
    validateDescs(model_config, ctx.kernel_seq_size_per_block, static_cast<int>(ctx.gen_num_per_cycle));
    auto layer_specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);
    auto topology = populateGroupsFromLayerSpecs(layer_specs, model_config, *ctx.parallelism_config);
    RTP_LLM_CHECK_WITH_INFO(!topology.first.empty(), "cache config produced no cache specs");
    auto storage = finalizeGroupStorage(topology);
    return {std::move(topology), storage};
}

CacheConfig createConfigFromDescs(const ModelConfig& model_config, const SpecBuildContext& ctx) {
    auto        data = buildConfigDataFromDescs(model_config, ctx);
    CacheConfig config(std::move(data.topology.first),
                       std::move(data.topology.second),
                       static_cast<uint32_t>(model_config.num_layers));
    config.seq_size_per_block                       = ctx.seq_size_per_block;
    config.use_mla                                  = model_config.attn_config.use_mla;
    config.enable_hybrid_attention                  = model_config.hybrid_attention_config.enable_hybrid_attention;
    config.dtype                                    = ctx.dtype;
    config.linear_step                              = 1;
    config.is_sparse                                = model_config.attn_config.is_sparse || data.storage.is_sparse;
    config.use_typed_cache_regions                  = data.storage.use_typed_cache_regions;
    config.use_opaque_kv_cache_store                = data.storage.use_opaque_kv_cache_store;
    config.disable_decode_first_malloc_device_reuse = config.use_opaque_kv_cache_store;
    return config;
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");
    if (budget.explicit_pool_reserve_bytes > total_budget_bytes) {
        return 0;
    }

    using Wide           = absl::uint128;
    const Wide remaining = Wide(total_budget_bytes - budget.explicit_pool_reserve_bytes);
    const Wide paged     = Wide(budget.paged_block_bytes);
    const Wide swa       = Wide(budget.swa_block_bytes);
    const Wide step      = Wide(static_cast<uint32_t>(std::max(1, linear_step)));
    Wide       result    = 0;

    if (paged == 0) {
        result = (remaining / swa) * step;
    } else {
        const Wide chunk_cost = step * paged + swa;
        const Wide chunks     = remaining / chunk_cost;
        const Wide remainder  = remaining % chunk_cost;
        result                = chunks * step;
        if (remainder >= swa + paged) {
            result += std::min(step - 1, (remainder - swa) / paged);
        }
    }

    const Wide max_result = Wide(std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(std::min(result, max_result));
}

LayerBuiltSpecs CacheConfigCreator::buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                             const SpecBuildContext&      ctx,
                                                             int64_t                      expected_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(expected_layer_num),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            layer_descs.size(),
                            expected_layer_num);
    LayerBuiltSpecs layer_specs(layer_descs.size());
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

CacheTopologyPair CacheConfigCreator::mergeMTPModule(CacheTopologyPair&       target,
                                                     const CacheTopologyPair& propose,
                                                     int                      module_index,
                                                     uint32_t                 main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(!target.first.empty(), "mergeMTPModule requires destination topology");
    RTP_LLM_CHECK_WITH_INFO(!propose.first.empty(), "mergeMTPModule requires propose topology");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "mergeMTPModule invalid module_index=%d", module_index);
    RTP_LLM_CHECK_WITH_INFO(target.second.size() >= static_cast<size_t>(main_layer_num),
                            "mergeMTPModule target layers=%zu less than main_layer_num=%u",
                            target.second.size(),
                            main_layer_num);

    const uint32_t propose_layer_num = static_cast<uint32_t>(propose.second.size());
    const size_t   old_layer_num     = target.second.size();
    RTP_LLM_CHECK_WITH_INFO(
        old_layer_num == static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index) * propose_layer_num,
        "mergeMTPModule target layer history=%zu expected=%zu",
        old_layer_num,
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index) * propose_layer_num);
    target.second.resize(old_layer_num + propose_layer_num);

    std::unordered_map<std::string, size_t> propose_index;
    for (size_t i = 0; i < propose.first.size(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(propose_index.emplace(propose.first[i].tag, i).second,
                                "mergeMTPModule duplicate propose tag=%s",
                                propose.first[i].tag.c_str());
    }

    std::optional<std::string> alias_target;
    const bool                 has_exact_default = std::any_of(
        target.first.begin(), target.first.end(), [](const CacheGroup& group) { return group.tag == "default"; });
    if (!has_exact_default && propose.first.size() == 1 && propose.first.front().tag == "default") {
        const auto& source = propose.first.front();
        if (source.spec != nullptr && source.policy.group_type == CacheGroupType::FULL
            && (source.spec->type == KVCacheSpecType::MultiHeadAttention
                || source.spec->type == KVCacheSpecType::MultiHeadLatentAttention)) {
            for (const auto& group : target.first) {
                if (group.tag != "default" && CacheConfig::samePolicy(group.policy, source.policy)
                    && group.spec != nullptr && group.spec->type == source.spec->type
                    && group.spec->memoryLayoutDType() == source.spec->memoryLayoutDType()
                    && group.spec->block_size_bytes() == source.spec->block_size_bytes()
                    && group.spec->scale_block_size_bytes() == source.spec->scale_block_size_bytes()
                    && group.seqSizePerBlock() == source.seqSizePerBlock()) {
                    RTP_LLM_CHECK_WITH_INFO(!alias_target.has_value(),
                                            "mergeMTPModule ambiguous default propose alias");
                    alias_target = group.tag;
                }
            }
        }
    }

    std::vector<CacheGroup> sub_groups;
    std::vector<CacheLayer> sub_layers(propose_layer_num);
    sub_groups.reserve(target.first.size());
    for (const auto& target_group : target.first) {
        const auto        exact     = propose_index.find(target_group.tag);
        const bool        has_exact = exact != propose_index.end();
        const bool        use_alias = !has_exact && alias_target.has_value() && target_group.tag == *alias_target;
        const CacheGroup* source_group_ptr =
            has_exact ? &propose.first[exact->second] : (use_alias ? &propose.first.front() : nullptr);
        const auto&      source_group = source_group_ptr != nullptr ? *source_group_ptr : target_group;
        std::vector<int> source_layer_ids;
        for (size_t layer_id = 0; layer_id < propose.second.size(); ++layer_id) {
            if (std::find(propose.second[layer_id].begin(), propose.second[layer_id].end(), source_group.tag)
                != propose.second[layer_id].end()) {
                source_layer_ids.push_back(static_cast<int>(layer_id));
            }
        }
        std::vector<int> current_layer_ids;
        for (size_t layer_id = 0; layer_id < old_layer_num; ++layer_id) {
            if (std::find(target.second[layer_id].begin(), target.second[layer_id].end(), target_group.tag)
                != target.second[layer_id].end()) {
                current_layer_ids.push_back(static_cast<int>(layer_id));
            }
        }
        std::vector<int> expected_layer_ids;
        for (int layer_id : current_layer_ids) {
            if (layer_id < static_cast<int>(main_layer_num)) {
                expected_layer_ids.push_back(layer_id);
            }
        }
        for (int prior = 0; prior < module_index; ++prior) {
            for (int local_layer_id : source_layer_ids) {
                expected_layer_ids.push_back(static_cast<int>(
                    CacheConfig::mtpGlobalLayerId(main_layer_num, prior, propose_layer_num, local_layer_id)));
            }
        }
        RTP_LLM_CHECK_WITH_INFO(current_layer_ids == expected_layer_ids,
                                "mergeMTPModule tag-local history mismatch for tag=%s",
                                target_group.tag.c_str());
        CacheGroup sub_group = source_group;
        if (use_alias) {
            sub_group.tag  = target_group.tag;
            sub_group.spec = source_group.spec->clone();
        }
        for (int local_layer_id : source_layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(local_layer_id >= 0 && static_cast<uint32_t>(local_layer_id) < propose_layer_num,
                                    "mergeMTPModule invalid source layer=%d",
                                    local_layer_id);
            sub_layers[static_cast<size_t>(local_layer_id)].push_back(target_group.tag);
            target.second[old_layer_num + static_cast<size_t>(local_layer_id)].push_back(target_group.tag);
        }
        sub_groups.push_back(std::move(sub_group));
    }
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(
            !sub_layers[layer_id].empty(), "mergeMTPModule missing group mapping for sub layer=%zu", layer_id);
    }
    return {std::move(sub_groups), std::move(sub_layers)};
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle) {
    const auto [seq_size_per_block, kernel_seq_size_per_block] = resolveSeqSizes(model_config, kv_cache_config);
    const auto ctx                                             = makeSpecBuildContext(
        model_config, parallelism_config, seq_size_per_block, kernel_seq_size_per_block, gen_num_per_cycle);
    auto config        = createConfigFromDescs(model_config, ctx);
    config.linear_step = std::max(1, kv_cache_config.linear_step);
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig    config    = createBasicConfig(model_config, parallelism_config, kv_cache_config, 0);
    const uint32_t block_num = computeLocalBlockNum(blockBudgetForConfig(config),
                                                    model_config,
                                                    runtime_config,
                                                    kv_cache_config,
                                                    parallelism_config,
                                                    warm_up_result,
                                                    sp_config,
                                                    config.linear_step);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.totalGroupBlockSizeBytes() / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
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
    const auto [seq_size_per_block, kernel_seq_size_per_block] = resolveSeqSizes(score_model_config, kv_cache_config);
    const auto score_ctx                                       = makeSpecBuildContext(score_model_config,
                                                parallelism_config,
                                                seq_size_per_block,
                                                kernel_seq_size_per_block,
                                                sp_config.gen_num_per_cycle);
    const auto propose_ctx                                     = makeSpecBuildContext(propose_model_config,
                                                  parallelism_config,
                                                  seq_size_per_block,
                                                  kernel_seq_size_per_block,
                                                  sp_config.gen_num_per_cycle);
    auto       score_data   = buildConfigDataFromDescs(score_model_config, score_ctx);
    auto       propose_data = buildConfigDataFromDescs(propose_model_config, propose_ctx);

    const int joint_step      = std::max(1, kv_cache_config.linear_step);
    int       num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle || sp_config.type == SP_TYPE_DSPARK) {
            num_mtp_modules = 1;
        }
    }

    const uint32_t main_layer_num    = static_cast<uint32_t>(score_data.topology.second.size());
    const uint32_t propose_layer_num = static_cast<uint32_t>(propose_data.topology.second.size());
    uint32_t       total_layer_num   = main_layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        RTP_LLM_CHECK_WITH_INFO(propose_layer_num <= std::numeric_limits<uint32_t>::max() - total_layer_num,
                                "CacheConfig total layer count overflow: main=%u propose=%u modules=%d",
                                main_layer_num,
                                propose_layer_num,
                                num_mtp_modules);
        total_layer_num += propose_layer_num;
    }

    const auto topologyBlockBytes = [](const CacheTopologyPair& topology) {
        std::unordered_map<std::string, size_t> layer_counts;
        for (const auto& layer : topology.second) {
            for (const auto& tag : layer) {
                ++layer_counts[tag];
            }
        }
        size_t total = 0;
        for (const auto& group : topology.first) {
            const size_t stride = group.kv_block_stride_bytes + group.kv_scale_stride_bytes;
            const size_t bytes  = stride * layer_counts[group.tag];
            RTP_LLM_CHECK_WITH_INFO(bytes <= std::numeric_limits<size_t>::max() - total,
                                    "joint score/propose block size overflow");
            total += bytes;
        }
        return total;
    };
    const size_t score_block_size_bytes   = topologyBlockBytes(score_data.topology);
    const size_t propose_block_size_bytes = topologyBlockBytes(propose_data.topology);
    RTP_LLM_CHECK_WITH_INFO(num_mtp_modules == 0
                                || propose_block_size_bytes
                                       <= (std::numeric_limits<size_t>::max() - score_block_size_bytes)
                                              / static_cast<size_t>(num_mtp_modules),
                            "joint score/propose block size overflow: score=%zu propose=%zu modules=%d",
                            score_block_size_bytes,
                            propose_block_size_bytes,
                            num_mtp_modules);
    const size_t total_block_size_bytes =
        score_block_size_bytes + static_cast<size_t>(num_mtp_modules) * propose_block_size_bytes;

    KVCacheBlockBudget joint_budget = score_data.storage.block_budget;
    addBlockBudget(joint_budget, propose_data.storage.block_budget, static_cast<size_t>(num_mtp_modules));
    const uint32_t block_num = computeLocalBlockNum(joint_budget,
                                                    score_model_config,
                                                    runtime_config,
                                                    kv_cache_config,
                                                    parallelism_config,
                                                    warm_up_result,
                                                    sp_config,
                                                    joint_step);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    auto makeConfig = [](BuiltConfigData&&  data,
                         const ModelConfig& model_config,
                         uint32_t           layer_all_num,
                         uint32_t           seq_size,
                         DataType           dtype,
                         int                linear_step) {
        CacheConfig config(std::move(data.topology.first),
                           std::move(data.topology.second),
                           static_cast<uint32_t>(model_config.num_layers));
        RTP_LLM_CHECK_WITH_INFO(config.layer_all_num == layer_all_num,
                                "CacheConfig topology layer count %u != expected %u",
                                config.layer_all_num,
                                layer_all_num);
        config.seq_size_per_block                       = seq_size;
        config.use_mla                                  = model_config.attn_config.use_mla;
        config.enable_hybrid_attention                  = model_config.hybrid_attention_config.enable_hybrid_attention;
        config.dtype                                    = dtype;
        config.linear_step                              = linear_step;
        config.is_sparse                                = model_config.attn_config.is_sparse || data.storage.is_sparse;
        config.use_typed_cache_regions                  = data.storage.use_typed_cache_regions;
        config.use_opaque_kv_cache_store                = data.storage.use_opaque_kv_cache_store;
        config.disable_decode_first_malloc_device_reuse = config.use_opaque_kv_cache_store;
        return config;
    };

    CacheTopologyPair                         target  = std::move(score_data.topology);
    const CacheTopologyPair&                  propose = propose_data.topology;
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;
    mtp_sub_configs.reserve(num_mtp_modules);
    for (int module_index = 0; module_index < num_mtp_modules; ++module_index) {
        auto sub_topology = CacheConfigCreator::mergeMTPModule(target, propose, module_index, main_layer_num);
        auto sub_storage  = finalizeGroupStorage(sub_topology);
        BuiltConfigData sub_data{std::move(sub_topology), sub_storage};
        mtp_sub_configs.push_back(std::make_shared<CacheConfig>(makeConfig(std::move(sub_data),
                                                                           propose_model_config,
                                                                           propose_layer_num,
                                                                           seq_size_per_block,
                                                                           propose_ctx.dtype,
                                                                           joint_step)));
    }

    auto            main_storage = finalizeGroupStorage(target);
    BuiltConfigData main_data{std::move(target), main_storage};
    CacheConfig     config = makeConfig(
        std::move(main_data), score_model_config, total_layer_num, seq_size_per_block, score_ctx.dtype, joint_step);
    config.mtp_sub_configs             = std::move(mtp_sub_configs);
    uint32_t validated_total_layer_num = config.layer_num;
    for (size_t module_index = 0; module_index < config.mtp_sub_configs.size(); ++module_index) {
        const auto& sub_config = config.mtp_sub_configs[module_index];
        RTP_LLM_CHECK_WITH_INFO(sub_config != nullptr, "CacheConfig MTP sub-config %zu is null", module_index);
        RTP_LLM_CHECK_WITH_INFO(sub_config->layer_num == sub_config->layer_all_num,
                                "CacheConfig MTP sub-config %zu main layers %u != topology layers %u",
                                module_index,
                                sub_config->layer_num,
                                sub_config->layer_all_num);
        RTP_LLM_CHECK_WITH_INFO(sub_config->layer_all_num
                                    <= std::numeric_limits<uint32_t>::max() - validated_total_layer_num,
                                "CacheConfig total MTP layer count overflow at module %zu",
                                module_index);
        validated_total_layer_num += sub_config->layer_all_num;
    }
    RTP_LLM_CHECK_WITH_INFO(validated_total_layer_num == config.layer_all_num,
                            "CacheConfig topology layers %u != main and MTP layers %u",
                            config.layer_all_num,
                            validated_total_layer_num);
    config.finalizeBlockNums(block_num, runtime_config);

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_block_size_bytes,
                     num_mtp_modules,
                     propose_block_size_bytes);
    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }
    return config;
}

}  // namespace rtp_llm
