#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <cstring>
#include <rapidjson/document.h>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

// Kernel blocks feeding a compressed (OpaqueKV) pool must be a whole number of
// 128-token units: 128 is the HCA compression unit and also FlashMLA's block
// quantum. Divisibility alone (the generic check) is not enough -- a 64-token
// kernel block passes it and then produces a partial compression unit.
constexpr uint32_t kCompressedKernelSeqSizeAlignment = 128;

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

    const auto swa_blocks = CacheConfig::baselineBlockNum(CacheGroupType::SWA, block_num, step);
    return budget.swa_block_bytes == 0 || static_cast<size_t>(swa_blocks) <= remaining / budget.swa_block_bytes;
}

void addBudgetBytes(size_t& total, size_t bytes, size_t count = 1) {
    RTP_LLM_CHECK_WITH_INFO(count == 0 || bytes <= (std::numeric_limits<size_t>::max() - total) / count,
                            "kv cache budget overflow: current=%zu bytes=%zu count=%zu",
                            total,
                            bytes,
                            count);
    total += bytes * count;
}

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    budget.max_baseline_block_num = config.maxBaselineBlockNum();
    for (size_t gid = 0; gid < config.topology().groups().size(); ++gid) {
        const auto& group          = config.topology().groups()[gid];
        size_t      group_bytes    = 0;
        const auto  append_segment = [&](const CacheConfig& source, const GroupBase& segment, bool main) {
            const auto   layer_ids = source.layerIdsForGroup(segment.tag);
            const size_t layer_count =
                main ? std::count_if(
                           layer_ids.begin(),
                           layer_ids.end(),
                           [&source](int id) { return id >= 0 && static_cast<uint32_t>(id) < source.layer_num; }) :
                        layer_ids.size();
            addBudgetBytes(group_bytes, segment.kvBlockStrideBytes(), layer_count);
            addBudgetBytes(group_bytes, segment.kvScaleStrideBytes(), layer_count);
        };
        // Exactly the physical segments assembled by BlockPoolConfigHelper:
        // target layers use the target Spec, draft layers use their own Specs.
        append_segment(config, group, true);
        for (const auto& sub : config.mtp_sub_configs) {
            RTP_LLM_CHECK_WITH_INFO(sub != nullptr, "MTP cache configuration is null for tag=%s", group.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(gid < sub->topology().groups().size(),
                                    "MTP cache configuration is missing group index=%zu for tag=%s",
                                    gid,
                                    group.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(sub->topology().groups()[gid].tag == group.tag,
                                    "MTP group order is inconsistent for tag=%s",
                                    group.tag.c_str());
            append_segment(*sub, sub->topology().groups()[gid], false);
        }
        const auto& policy = group.policy;
        if (policy.explicit_block_num > 0) {
            if (policy.charge_to_paged_budget) {
                addBudgetBytes(budget.explicit_pool_reserve_bytes, group_bytes, policy.explicit_block_num);
            }
            continue;
        }
        switch (policy.group_type) {
            case CacheGroupType::FULL:
            case CacheGroupType::LINEAR:
                addBudgetBytes(budget.paged_block_bytes, group_bytes, config.blockWeight(group.tag));
                break;
            case CacheGroupType::SWA:
                addBudgetBytes(budget.swa_block_bytes, group_bytes, config.blockWeight(group.tag));
                break;
        }
    }
    return budget;
}

void resolvePoolBlockWeights(CacheConfig& config, const std::string& json) {
    if (json.empty()) {
        return;
    }
    rapidjson::Document document;
    document.Parse<rapidjson::kParseValidateEncodingFlag>(json.data(), json.size());
    RTP_LLM_CHECK_WITH_INFO(json.find('\0') == std::string::npos && !document.HasParseError() && document.IsObject(),
                            "KV_CACHE_POOL_BLOCK_WEIGHTS must be a JSON object of pool tags to positive integers");
    std::map<std::string, uint32_t> weights;
    for (auto it = document.MemberBegin(); it != document.MemberEnd(); ++it) {
        const std::string tag(it->name.GetString(), it->name.GetStringLength());
        RTP_LLM_CHECK_WITH_INFO(it->value.IsUint64() && it->value.GetUint64() > 0
                                    && it->value.GetUint64() <= std::numeric_limits<int32_t>::max(),
                                "KV_CACHE_POOL_BLOCK_WEIGHTS pool '%s' must have a positive int32 integer weight",
                                tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(weights.emplace(tag, static_cast<uint32_t>(it->value.GetUint64())).second,
                                "KV_CACHE_POOL_BLOCK_WEIGHTS has duplicate pool '%s'",
                                tag.c_str());
        const auto group = std::find_if(config.groups().begin(),
                                        config.groups().end(),
                                        [&](const GroupBase& candidate) { return candidate.tag == tag; });
        RTP_LLM_CHECK_WITH_INFO(group != config.groups().end(),
                                "KV_CACHE_POOL_BLOCK_WEIGHTS has unknown pool '%s' in merged target/draft topology",
                                tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group->policy.memory_placement == CacheMemoryPlacement::DEVICE,
                                "KV_CACHE_POOL_BLOCK_WEIGHTS pool '%s' must be DEVICE resident",
                                tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group->policy.explicit_block_num == 0,
                                "KV_CACHE_POOL_BLOCK_WEIGHTS pool '%s' conflicts with explicit_block_num=%u",
                                tag.c_str(),
                                group->policy.explicit_block_num);
    }
    config.pool_block_weights = std::move(weights);
    // Children have the merged logical topology, including empty target-only
    // groups. Their physical views must publish the same pool capacities.
    for (auto& sub : config.mtp_sub_configs) {
        RTP_LLM_CHECK_WITH_INFO(sub != nullptr, "MTP cache configuration is null while resolving pool weights");
        sub->pool_block_weights = config.pool_block_weights;
    }
}

std::pair<uint32_t, uint32_t> resolveSeqSizes(const ModelConfig& model_config, const KVCacheConfig& kv_cache_config) {
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.seq_size_per_block >= 0 && kv_cache_config.kernel_seq_size_per_block >= 0,
                            "cache block spans must be non-negative before resolution");
    const auto& attention = model_config.attn_config;
    RTP_LLM_CHECK_WITH_INFO(attention.tokens_per_block <= std::numeric_limits<uint32_t>::max()
                                && attention.kernel_tokens_per_block <= std::numeric_limits<uint32_t>::max(),
                            "model cache block spans exceed uint32 range");
    // Model construction has already applied CLI values and model-specific defaults.
    // Raw KVCacheConfig is a fallback for callers without resolved model geometry.
    // This is the base cache-key span (tokens/key block), before per-group projection.
    const auto physical_tokens_per_block = static_cast<uint32_t>(
        attention.tokens_per_block > 0 ? attention.tokens_per_block : kv_cache_config.seq_size_per_block);
    const auto kernel_tokens_per_block = static_cast<uint32_t>(
        attention.kernel_tokens_per_block > 0         ? attention.kernel_tokens_per_block :
        kv_cache_config.kernel_seq_size_per_block > 0 ? kv_cache_config.kernel_seq_size_per_block :
                                                        physical_tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "cache-key span in tokens must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "cache kernel_seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block >= kernel_tokens_per_block
                                && physical_tokens_per_block % kernel_tokens_per_block == 0,
                            "cache seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);
    return {physical_tokens_per_block, kernel_tokens_per_block};
}

void validateAttentionMetadata(const ModelConfig& model_config) {
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    // Linear model weight loading also needs metadata for FULL-only MTP layers.
    if (types.empty() && model_config.linear_attention_config.linear_num_value_heads == 0) {
        for (const auto& descs : model_config.kv_cache_spec_descs) {
            for (const auto& desc : descs) {
                RTP_LLM_CHECK_WITH_INFO(
                    desc.cache_type != KVCacheSpecType::LinearAttention,
                    "linear attention cache requires attention metadata and linear model dimensions");
            }
        }
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    for (size_t layer_id = 0; layer_id < types.size(); ++layer_id) {
        const bool  expects_linear = types[layer_id] == HybridAttentionType::LINEAR;
        const auto& descs          = model_config.kv_cache_spec_descs[layer_id];
        const bool  is_linear      = std::any_of(descs.begin(), descs.end(), [](const KVCacheSpecDesc& desc) {
            return desc.cache_type == KVCacheSpecType::LinearAttention;
        });
        RTP_LLM_CHECK_WITH_INFO(expects_linear == is_linear,
                                "hybrid layer %zu attention type does not match cache descriptor type",
                                layer_id);
    }
}

SpecBuildContext makeSpecBuildContext(const ModelConfig&       model_config,
                                      const ParallelismConfig& parallelism_config,
                                      uint32_t                 seq_size_per_block,
                                      uint32_t                 kernel_tokens_per_block,
                                      int                      gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        gen_num_per_cycle >= 0, "cache config requires non-negative gen_num_per_cycle, got %d", gen_num_per_cycle);
    SpecBuildContext ctx;
    ctx.dtype                   = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.kernel_tokens_per_block = kernel_tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    return ctx;
}

void validateDescs(const ModelConfig& model_config, uint32_t kernel_tokens_per_block) {
    RTP_LLM_CHECK_WITH_INFO(model_config.num_layers > 0 && model_config.num_layers <= std::numeric_limits<int>::max(),
                            "cache config requires a positive layer count within int range");
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "cache config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "cache config layer %ld has no descs", layer_id);
        for (const auto& desc : descs) {
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "desc tag=%s has invalid compression_ratio=%u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_tokens_per_block);
                if (desc.compression_ratio > 1) {
                    RTP_LLM_CHECK_WITH_INFO(
                        kernel_tokens_per_block >= kCompressedKernelSeqSizeAlignment
                            && kernel_tokens_per_block % kCompressedKernelSeqSizeAlignment == 0,
                        "desc tag=%s kernel_seq_size_per_block=%u must be a positive multiple of alignment=%u",
                        desc.tag.c_str(),
                        kernel_tokens_per_block,
                        kCompressedKernelSeqSizeAlignment);
                }
            }
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "state ring desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

void populateGroups(CacheConfig&                 config,
                    const LayerKVCacheSpecDescs& descs_by_layer,
                    const LayerKVCacheSpecs&     specs_by_layer,
                    const ModelConfig&           model_config) {
    struct BuildState {
        KVCacheSpecPtr   spec;
        std::string      fingerprint;
        CacheGroupPolicy policy;
    };
    std::map<std::string, BuildState> groups_by_tag;
    std::vector<std::string>          ordered_tags;
    std::vector<LayerBase>            layers(static_cast<size_t>(config.layer_num));
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
        const auto& descs         = descs_by_layer[layer_id];
        const auto& specs         = specs_by_layer[layer_id];
        RTP_LLM_CHECK_WITH_INFO(descs.size() == specs.size(),
                                "cache layer %u desc count %zu != spec count %zu",
                                layer_id,
                                descs.size(),
                                specs.size());
        std::set<std::string> layer_tags;
        for (size_t i = 0; i < descs.size(); ++i) {
            const auto& desc = descs[i];
            const auto& spec = specs[i];
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache layer %u has null spec", layer_id);
            RTP_LLM_CHECK_WITH_INFO(layer_tags.emplace(spec->tag).second,
                                    "cache layer %u has duplicate tag=%s",
                                    layer_id,
                                    spec->tag.c_str());
            auto policy = SpecBuilder::groupPolicy(desc);
            checkGroupResidencyBudget(policy, spec->tag);
            if (policy.group_type == CacheGroupType::SWA) {
                RTP_LLM_CHECK_WITH_INFO(model_config.attn_config.sliding_window >= 0,
                                        "SWA tag=%s has negative sliding window=%d",
                                        spec->tag.c_str(),
                                        model_config.attn_config.sliding_window);
                policy.sliding_window_size = model_config.attn_config.sliding_window;
            }
            auto [it, inserted] = groups_by_tag.emplace(spec->tag, BuildState{});
            if (inserted) {
                it->second.spec        = spec;
                it->second.fingerprint = spec->fingerprint();
                it->second.policy      = policy;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(it->second.fingerprint == spec->fingerprint(),
                                        "cache tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(it->second.policy, policy),
                                        "cache tag=%s has inconsistent policy",
                                        spec->tag.c_str());
            }
            layers[layer_id].group_tags.push_back(spec->tag);
        }
    }

    std::vector<GroupBase> groups;
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto& state = groups_by_tag.at(tag);
        GroupBase   group;
        group.tag    = tag;
        group.spec   = state.spec;
        group.policy = state.policy;
        groups.push_back(std::move(group));
    }
    config.setTopology(std::move(groups), std::move(layers));
}

CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                  const ParallelismConfig& parallelism_config,
                                  const KVCacheConfig&     kv_cache_config,
                                  int                      gen_num_per_cycle) {
    const auto [seq_size, kernel_seq_size] = resolveSeqSizes(model_config, kv_cache_config);
    validateDescs(model_config, kernel_seq_size);
    validateAttentionMetadata(model_config);
    const auto ctx =
        makeSpecBuildContext(model_config, parallelism_config, seq_size, kernel_seq_size, gen_num_per_cycle);
    const auto specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    CacheConfig config;
    config.dtype                   = ctx.dtype;
    config.layer_num               = static_cast<uint32_t>(model_config.num_layers);
    config.cache_min_replay_tokens = std::max(1, model_config.cache_min_replay_tokens);
    config.cache_key_hash_seed     = model_config.cache_key_hash_seed;

    config.seq_size_per_block = seq_size;
    config.use_mla            = model_config.attn_config.use_mla;
    config.is_sparse          = model_config.attn_config.is_sparse;
    populateGroups(config, model_config.kv_cache_spec_descs, specs, model_config);
    for (const auto& group : config.topology().groups()) {
        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        config.use_typed_cache_regions |= opaque;
        config.use_opaque_kv_cache_store |= opaque;
        config.is_sparse |= group.spec->type == KVCacheSpecType::OpaqueKV;
    }
    config.disable_decode_first_malloc_device_reuse |= config.use_opaque_kv_cache_store;
    return config;
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");

    uint32_t low  = 0;
    uint32_t high = budget.max_baseline_block_num;
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
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle) {
    auto config        = createConfigFromDescs(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    config.linear_step = std::max(1, kv_cache_config.linear_step);
    if (!kv_cache_config.enable_linear_cache_reclaim) {
        return config;
    }
    const auto& groups     = config.topology().groups();
    const bool  has_linear = std::any_of(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.group_type == CacheGroupType::LINEAR;
    });
    // Target and draft models share KVCacheConfig. A FULL-only draft has no
    // recurrent state to reclaim and must not inherit the target capability.
    if (!has_linear) {
        return config;
    }
    RTP_LLM_CHECK_WITH_INFO(model_config.linear_attention_config.supports_history_state_reclaim,
                            "enable_linear_cache_reclaim requires model supports_history_state_reclaim");
    for (const auto& group : groups) {
        if (group.policy.group_type == CacheGroupType::LINEAR) {
            RTP_LLM_CHECK_WITH_INFO(
                group.cacheKeyTokenStride() == config.seq_size_per_block,
                "enable_linear_cache_reclaim requires LINEAR cache-key span to match the base span: "
                "tag=%s group_span=%zu base_span=%zu",
                group.tag.c_str(),
                group.cacheKeyTokenStride(),
                config.seq_size_per_block);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config.prefill_cp_config.kv_cache_sharded,
                            "enable_linear_cache_reclaim does not support CP-sharded KV cache");
    config.enable_linear_cache_reclaim = true;
    return config;
}

CacheConfig CacheConfigCreator::createWarmupConfig(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   int                      gen_num_per_cycle) {
    KVCacheConfig options;
    options.seq_size_per_block        = 0;
    options.kernel_seq_size_per_block = 0;
    return createWarmupConfig(model_config, parallelism_config, options, gen_num_per_cycle);
}

CacheConfig CacheConfigCreator::createWarmupConfig(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   const KVCacheConfig&     kv_cache_config,
                                                   int                      gen_num_per_cycle) {
    auto config = createBasicConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    // Upstream warmup pools reserve the sentinel block plus one allocatable
    // block per group, and never inherit linear_step: the SWA
    // ceil(baseline/step) shrink would collapse those groups to sentinel-only.
    config.linear_step = 1;
    config.finalizeBlockNums(2, RuntimeConfig{});
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config,
                                             const ModelConfig*                               draft_model_config,
                                             bool                                             is_mtp,
                                             bool                                             is_eagle) {
    RTP_LLM_CHECK_WITH_INFO(draft_model_config == nullptr || sp_config.has_value(),
                            "draft cache configuration requires speculative execution configuration");
    RTP_LLM_CHECK_WITH_INFO(
        draft_model_config == nullptr
            || (sp_config->gen_num_per_cycle >= 0 && sp_config->gen_num_per_cycle <= std::numeric_limits<int>::max()),
        "draft proposal token count must fit a non-negative int");
    const int gen_num_per_cycle = draft_model_config != nullptr ? sp_config->gen_num_per_cycle : 0;
    auto      config = createBasicConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    if (draft_model_config != nullptr) {
        auto draft = createBasicConfig(*draft_model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
        // DSpARK/DFlash are single multi-layer block-draft models; gamma is
        // the proposal width, not a count of independent one-layer modules.
        int  num_mtp_modules = is_mtp && !is_eagle && !isBlockDraftType(sp_config->type) ? gen_num_per_cycle : 1;
        RTP_LLM_CHECK_WITH_INFO(num_mtp_modules > 0, "draft cache configuration requires at least one module");
        config.mtp_sub_configs.reserve(static_cast<size_t>(num_mtp_modules));
        for (int module = 0; module < num_mtp_modules; ++module) {
            config.mtp_sub_configs.push_back(config.mergeMTPModule(draft, module, config.layer_num));
        }
    }
    resolvePoolBlockWeights(config, kv_cache_config.kv_cache_pool_block_weights);
    return config;
}

uint32_t CacheConfigCreator::computeLocalBlockNum(const CacheConfig&                               config,
                                                  const ModelConfig&                               model_config,
                                                  const RuntimeConfig&                             runtime_config,
                                                  const KVCacheConfig&                             kv_cache_config,
                                                  const ParallelismConfig&                         parallelism_config,
                                                  const std::optional<WarmUpResult>&               warm_up_result,
                                                  const std::optional<SpeculativeExecutionConfig>& sp_config) {
    uint32_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified baseline kv cache block num %d",
                         kv_cache_config.test_block_num);
        block_num = static_cast<uint32_t>(kv_cache_config.test_block_num);
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
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
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, config.linear_step);
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least one baseline block, got %u", block_num);
    RTP_LLM_CHECK_WITH_INFO(block_num <= config.maxBaselineBlockNum(),
                            "cache baseline=%u exceeds weighted pool int32 block ID bounds",
                            block_num);
    const auto kv_cache_seq_len = static_cast<uint64_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("cache layout created: baseline_blocks=%u, baseline_token_span=%lu, draft_modules=%zu",
                     block_num,
                     kv_cache_seq_len,
                     config.mtp_sub_configs.size());
    if (kv_cache_seq_len < static_cast<uint64_t>(model_config.max_seq_len)) {
        RTP_LLM_LOG_WARNING("cache baseline token span %lu is below max_seq_len %ld (before per-pool block weights)",
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    // Only the baseline candidate is available here. Group capacities are
    // applied after runtime cross-rank agreement, before any pool allocation.
    return block_num;
}

uint32_t CacheConfigCreator::synchronizeBlockNum(uint32_t                 candidate_block_num,
                                                 const ParallelismConfig& parallelism_config) {
    size_t world_size = parallelism_config.tp_size * parallelism_config.dp_size;
    if (world_size > 1) {
        RTP_LLM_CHECK_WITH_INFO(candidate_block_num <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                                "candidate cache block count exceeds collective int32 range");
        size_t local_rank    = parallelism_config.tp_size * parallelism_config.dp_rank + parallelism_config.tp_rank;
        auto   block_num_t   = torch::empty({(int64_t)world_size}, torch::kInt32).pin_memory();
        auto   block_num_ptr = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = static_cast<int>(candidate_block_num);
        execAllGather({{block_num_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        return selectConfirmedBlockNum(
            block_num_ptr, world_size, parallelism_config.ffn_disaggregate_config.is_ffn_service());
    }
    return candidate_block_num;
}

CacheConfigCreator::SizingSignature
CacheConfigCreator::sizingSignature(const CacheConfig& config, uint32_t baseline, bool is_ffn_service) {
    // Stable byte encoding and two independently seeded 64-bit FNV streams.
    // Do not use std::hash or raw structs (padding and enum sizes are not a
    // portable collective contract). This is an agreement checksum, not an ID
    // used for prefix reuse or PD cache-store interoperability.
    uint64_t   first  = 14695981039346656037ull;
    uint64_t   second = 7809847782465536322ull;
    const auto byte   = [&](uint8_t value) {
        first  = (first ^ value) * 1099511628211ull;
        second = (second ^ value) * 1099511628211ull;
    };
    const auto number = [&](uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            byte(static_cast<uint8_t>(value & 0xff));
            value >>= 8;
        }
    };
    number(1);  // Signature format version.
    number(std::max(1, config.linear_step));
    number(config.groups().size());
    for (const auto& group : config.groups()) {
        number(group.tag.size());
        for (unsigned char character : group.tag) {
            byte(character);
        }
        const auto& policy = group.policy;
        number(static_cast<uint64_t>(policy.group_type));
        number(policy.explicit_block_num);
        number(policy.charge_to_paged_budget);
        number(static_cast<uint64_t>(policy.memory_placement));
        number(policy.enable_prefix_reuse);
        number(policy.reservable);
        number(policy.active_tail_blocks);
        number(policy.validate_tail_blocks);
        number(static_cast<uint64_t>(policy.cp_mapping));
        number(static_cast<uint64_t>(policy.cp_slice));
        number(policy.sliding_window_size);
        number(config.blockWeight(group.tag));
        if (baseline > 0) {
            number(config.blockNumForGroup(group, baseline));
        }
    }
    SizingSignature result{};
    // FFN disaggregation shares DP_AND_TP/WORLD with attention ranks, but its
    // existing baseline is one. Compare capacities only within each role.
    result[0] = is_ffn_service ? 1 : 0;
    // Encode into fixed-width int32 tensor words without signed narrowing.
    const std::array<uint32_t, 4> words{static_cast<uint32_t>(first),
                                        static_cast<uint32_t>(first >> 32),
                                        static_cast<uint32_t>(second),
                                        static_cast<uint32_t>(second >> 32)};
    std::memcpy(result.data() + 1, words.data(), sizeof(words));
    return result;
}

void CacheConfigCreator::validateSizingSignatures(const int32_t* signatures, size_t rank_count) {
    RTP_LLM_CHECK_WITH_INFO(signatures != nullptr && rank_count > 0,
                            "cross-rank cache sizing signatures must not be empty");
    constexpr size_t              width = std::tuple_size<SizingSignature>::value;
    std::array<const int32_t*, 2> first_by_role{};
    for (size_t rank = 0; rank < rank_count; ++rank) {
        const auto*   signature = signatures + rank * width;
        const int32_t role      = signature[0];
        RTP_LLM_CHECK_WITH_INFO(
            role == 0 || role == 1, "invalid cache sizing cohort=%d from TP/DP rank %zu", role, rank);
        const auto* first = first_by_role[role];
        if (first == nullptr) {
            first_by_role[role] = signature;
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(std::equal(first + 1, first + width, signature + 1),
                                "cache pool sizing mismatch within %s cohort at TP/DP rank %zu; "
                                "pool topology, policies, weights and final capacities must agree",
                                role == 1 ? "FFN" : "attention",
                                rank);
    }
}

uint32_t CacheConfigCreator::synchronizeBlockNum(uint32_t                 candidate_block_num,
                                                 const ParallelismConfig& parallelism_config,
                                                 const CacheConfig&       config) {
    const size_t world_size = parallelism_config.tp_size * parallelism_config.dp_size;
    if (world_size <= 1) {
        return candidate_block_num;
    }
    const size_t     local_rank = parallelism_config.tp_size * parallelism_config.dp_rank + parallelism_config.tp_rank;
    constexpr size_t width      = std::tuple_size<SizingSignature>::value;
    const auto       agree      = [&](uint32_t baseline) {
        const auto signature =
            sizingSignature(config, baseline, parallelism_config.ffn_disaggregate_config.is_ffn_service());
        // Shape never depends on local pool count or the JSON configuration.
        // Even inconsistent topologies therefore reach the same collective.
        auto  tensor = torch::empty({static_cast<int64_t>(world_size * width)}, torch::kInt32).pin_memory();
        auto* data   = tensor.data_ptr<int32_t>();
        std::copy(signature.begin(), signature.end(), data + local_rank * width);
        execAllGather({{tensor}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();
        validateSizingSignatures(data, world_size);
    };
    agree(0);
    const auto confirmed = synchronizeBlockNum(candidate_block_num, parallelism_config);
    agree(confirmed);
    return confirmed;
}

uint32_t CacheConfigCreator::selectConfirmedBlockNum(const int* candidates, size_t count, bool is_ffn_service) {
    RTP_LLM_CHECK_WITH_INFO(candidates != nullptr && count > 0, "cross-rank cache candidates must not be empty");
    if (is_ffn_service) {
        return 1;
    }
    const auto confirmed = *std::min_element(candidates, candidates + count);
    RTP_LLM_CHECK_WITH_INFO(confirmed > 0, "cross-rank cache block count must be positive");
    return static_cast<uint32_t>(confirmed);
}

}  // namespace rtp_llm
