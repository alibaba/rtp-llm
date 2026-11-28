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
#include "rtp_llm/cpp/config/RankLayout.h"
#include "rtp_llm/cpp/config/StageLayerRange.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
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

bool blockNumFitsBudget(uint32_t block_num, size_t total_budget_bytes, const KVCacheBlockBudget& budget) {
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

    // Main #1574 keeps SWA pool blocks at the same baseline count (tail
    // retention is a runtime concern), so no SWA downsampling here.
    return budget.swa_block_bytes == 0 || static_cast<size_t>(block_num) <= remaining / budget.swa_block_bytes;
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
    addBudgetBytes(budget.explicit_pool_reserve_bytes, config.linear_replay.reserve_bytes);
    // Each module owns an independent pool per group tag, so the paged-HBM
    // budget is the sum over every module's own group byte layout.
    for (size_t module_id = 0; module_id < config.moduleCount(); ++module_id) {
        const auto& module_topology = config.topology(module_id);
        for (const auto& group : module_topology.groups()) {
            const size_t group_bytes = module_topology.blockSizeBytesForGroup(group.tag);
            const auto&  policy      = group.policy;
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
    }
    return budget;
}

void resolvePoolBlockWeights(CacheConfig& config, const std::string& json, int64_t pp_size) {
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
        const auto tags = config.allGroupTags();
        if (std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            // Every PP rank sees the deployment config, including tags owned by other stages.
            RTP_LLM_CHECK_WITH_INFO(pp_size > 1,
                                    "KV_CACHE_POOL_BLOCK_WEIGHTS has unknown pool '%s' in cache topology",
                                    tag.c_str());
            continue;
        }
        const auto* group = &config.group(tag);
        RTP_LLM_CHECK_WITH_INFO(group->policy.memory_placement == CacheMemoryPlacement::DEVICE,
                                "KV_CACHE_POOL_BLOCK_WEIGHTS pool '%s' must be DEVICE resident",
                                tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group->policy.explicit_block_num == 0,
                                "KV_CACHE_POOL_BLOCK_WEIGHTS pool '%s' conflicts with explicit_block_num=%u",
                                tag.c_str(),
                                group->policy.explicit_block_num);
    }
    config.pool_block_weights = std::move(weights);

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

CacheGroupPolicy makeGroupPolicy(const KVCacheSpecDesc& desc, int sliding_window) {
    auto policy = SpecBuilder::groupPolicy(desc);
    checkGroupResidencyBudget(policy, desc.tag);
    if (policy.group_type == CacheGroupType::SWA) {
        RTP_LLM_CHECK_WITH_INFO(
            sliding_window >= 0, "SWA tag=%s has negative sliding window=%d", desc.tag.c_str(), sliding_window);
        policy.sliding_window_size = sliding_window;
    }
    return policy;
}

void populateGroups(CacheConfig&                 config,
                    size_t                       module_id,
                    const ModelConfig&           model_config,
                    const SpecBuildContext&      ctx,
                    const StageLayerRange&       range) {
    const auto& layer_descs = model_config.kv_cache_spec_descs;
    std::map<std::string, size_t> group_ids;
    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers;
    layers.reserve(range.size);
    for (size_t layer_id = 0; layer_id < layer_descs.size(); ++layer_id) {
        const bool is_local = layer_id >= range.begin && layer_id < range.end();
        if (is_local) {
            layers.push_back({static_cast<int>(layer_id - range.begin), {}});
        }
        for (const auto& desc : layer_descs[layer_id]) {
            const auto spec     = SpecBuilder::build(desc, ctx);
            const auto policy   = makeGroupPolicy(desc, model_config.attn_config.sliding_window);
            auto [it, inserted] = group_ids.try_emplace(desc.tag, groups.size());
            if (inserted) {
                groups.push_back({desc.tag, spec, policy});
            } else {
                const auto& group = groups[it->second];
                RTP_LLM_CHECK_WITH_INFO(group.spec->fingerprint() == spec->fingerprint(),
                                        "cache tag=%s has multiple physical prototypes",
                                        desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group.policy, policy),
                                        "cache tag=%s has inconsistent policy",
                                        desc.tag.c_str());
            }
            if (is_local) {
                layers.back().group_tags.push_back(desc.tag);
            }
        }
    }
    config.setTopology(module_id, std::move(groups), std::move(layers));
}

CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                  const ParallelismConfig& parallelism_config,
                                  const KVCacheConfig&     kv_cache_config,
                                  int                      gen_num_per_cycle,
                                  const StageLayerRange&   range) {
    const auto [seq_size, kernel_seq_size] = resolveSeqSizes(model_config, kv_cache_config);
    validateDescs(model_config, kernel_seq_size);
    validateAttentionMetadata(model_config);
    const auto ctx =
        makeSpecBuildContext(model_config, parallelism_config, seq_size, kernel_seq_size, gen_num_per_cycle);
    CacheConfig config;
    auto&       main = config.moduleMetadata(0);
    main.dtype       = ctx.dtype;
    config.global_layer_begin      = range.begin;
    config.cache_min_replay_tokens = std::max(1, model_config.cache_min_replay_tokens);
    config.cache_key_hash_seed     = model_config.cache_key_hash_seed;

    const auto& cp = parallelism_config.prefill_cp_config;
    config.local_kv_cache_sharded =
        parallelism_config.tp_size > 1
        && (model_config.use_page_rr_kv_cache
            || (cp.kv_cache_sharded && (cp.is_enabled() || cp.is_prefill_enabled())
                && parallelism_config.role_type != RoleType::DECODE));
    RTP_LLM_CHECK_WITH_INFO(!config.local_kv_cache_sharded || cp.kv_cache_sharded,
                            "local PageRR allocation requires kv_cache_sharded transport metadata");

    config.seq_size_per_block = seq_size;
    main.use_mla             = model_config.attn_config.use_mla;
    main.is_sparse           = model_config.attn_config.is_sparse;
    populateGroups(config, 0, model_config, ctx, range);
    for (const auto& group : config.topology().groups()) {
        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        /** The first decode allocation decides reuse for all cache groups in the request. */
        main.disable_decode_first_malloc_device_reuse |= opaque;
        if (config.layerIdsForGroup(group.tag).empty()) {
            continue;
        }
        main.use_opaque_kv_cache_store |= opaque;
        main.is_sparse |= group.spec->type == KVCacheSpecType::OpaqueKV;
    }
    return config;
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget) {
    if (budget.paged_block_bytes == 0 && budget.swa_block_bytes == 0) {
        return budget.explicit_pool_reserve_bytes <= total_budget_bytes ? 1u : 0u;
    }

    uint32_t low  = 0;
    uint32_t high = budget.max_baseline_block_num;
    while (low < high) {
        const uint32_t mid = low + static_cast<uint32_t>((static_cast<uint64_t>(high) - low + 1) / 2);
        if (blockNumFitsBudget(mid, total_budget_bytes, budget)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle,
                                                  const StageLayerRange&   range) {
    auto config =
        createConfigFromDescs(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, range);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.reuse_cache_tail_blocks > 0, "reuse_cache_tail_blocks must be positive");
    config.moduleMetadata(0).reuse_cache_tail_blocks = kv_cache_config.reuse_cache_tail_blocks;
    {
        auto groups = config.topology().groups();
        for (auto& group : groups) {
            const uint32_t override_blocks =
                group.policy.group_type == CacheGroupType::LINEAR ? kv_cache_config.linear_pool_blocks :
                group.policy.group_type == CacheGroupType::SWA    ? kv_cache_config.swa_pool_blocks :
                                                                    0;
            RTP_LLM_CHECK_WITH_INFO(override_blocks != 1
                                        && override_blocks <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
                                    "pool blocks for tag=%s must be 0 or in [2, INT_MAX]",
                                    group.tag.c_str());
            if (override_blocks > 0 && group.policy.explicit_block_num == 0) {
                group.policy.explicit_block_num     = override_blocks;
                group.policy.charge_to_paged_budget = group.policy.memory_placement == CacheMemoryPlacement::DEVICE;
            }
        }
        config.setTopology(0, std::move(groups), config.topology().layers());
    }
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
    config.moduleMetadata(0).enable_linear_cache_reclaim = true;
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

CacheConfig CacheConfigCreator::createMainModelCacheConfig(const ModelConfig&       model_config,
                                                           const ParallelismConfig& parallelism_config,
                                                           const KVCacheConfig&     kv_cache_config,
                                                           int                      gen_num_per_cycle) {
    const auto range = stageLayerRange(parallelism_config, parallelism_config.pp_rank, model_config.num_layers);
    return createBasicConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, range);
}

CacheConfig CacheConfigCreator::createWarmupConfig(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   const KVCacheConfig&     kv_cache_config,
                                                   int                      gen_num_per_cycle) {
    auto config = createMainModelCacheConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    config.finalizeBlockNums(2, RuntimeConfig{});
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.block_num = 2;
    }
    config.setTopology(0, std::move(groups), config.topology().layers());
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config,
                                             const ModelConfig*                               draft_model_config,
                                             bool /*is_mtp*/,
                                             bool /*is_eagle*/) {
    RTP_LLM_CHECK_WITH_INFO(draft_model_config == nullptr || sp_config.has_value(),
                            "draft cache configuration requires speculative execution configuration");
    // Main state-ring geometry must agree across PP stages, including stages
    // that lack a local draft module.
    const bool has_speculative_round =
        draft_model_config != nullptr
        || (parallelism_config.pp_size > 1 && sp_config && sp_config->type != SP_TYPE_NONE);
    RTP_LLM_CHECK_WITH_INFO(
        !has_speculative_round
            || (sp_config->gen_num_per_cycle >= 0 && sp_config->gen_num_per_cycle <= std::numeric_limits<int>::max()),
        "draft proposal token count must fit a non-negative int");
    const int  gen_num_per_cycle  = has_speculative_round ? sp_config->gen_num_per_cycle : 0;
    auto       config = createMainModelCacheConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    if (draft_model_config != nullptr) {
        const auto layout = RankLayout::fromParallelismConfig(parallelism_config);
        // The draft topology registers every draft tag on every PP stage;
        // stages without local draft layers keep the group definitions for
        // block ID allocation but bind no layers.
        const StageLayerRange draft_range{0, static_cast<uint32_t>(draft_model_config->num_layers)};
        auto draft_config = createBasicConfig(
            *draft_model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, draft_range);
        RTP_LLM_CHECK_WITH_INFO(config.local_kv_cache_sharded == draft_config.local_kv_cache_sharded,
                                "target and MTP must use the same local KV cache layout");
        RTP_LLM_CHECK_WITH_INFO(draft_config.seq_size_per_block == config.seq_size_per_block,
                                "target cache-key block span %zu differs from the independently parsed MTP span %zu",
                                config.seq_size_per_block,
                                draft_config.seq_size_per_block);
        if (!layout.isLastStage()) {
            draft_config.setTopology(0, draft_config.groups(), {});
        }
        // The draft module declares its own module-scoped tags in the Python
        // model description (``ModelConfig.cache_tag_prefix``); no implicit
        // renaming happens here.  Cross-module duplicates are rejected by
        // ``validateCrossModuleInvariants``.
        config.addMtpModule(draft_config.topologyPtr(0), draft_config.moduleMetadata(0));
    }
    config.validateCrossModuleInvariants();
    resolvePoolBlockWeights(config, kv_cache_config.kv_cache_pool_block_weights, parallelism_config.pp_size);
    return config;
}

void CacheConfigCreator::configureLinearReplay(CacheConfig&                                     config,
                                               const ModelConfig&                               model_config,
                                               const ParallelismConfig&                         parallelism_config,
                                               const RuntimeConfig&                             runtime_config,
                                               const std::optional<SpeculativeExecutionConfig>& sp_config,
                                               bool                                             is_mtp) {
    if (!runtime_config.enable_linear_replay || !is_mtp || parallelism_config.pp_size > 1
        || parallelism_config.role_type == RoleType::PREFILL || !sp_config || sp_config->gen_num_per_cycle <= 0
        || (sp_config->type != SP_TYPE_MTP && sp_config->type != SP_TYPE_EAGLE && sp_config->type != SP_TYPE_EAGLE3)) {
        return;
    }
    std::vector<std::string> group_tags;
    size_t           layer_count = 0;
    for (const auto& group : config.groups()) {
        if (group.policy.group_type != CacheGroupType::LINEAR
            || group.policy.memory_placement != CacheMemoryPlacement::DEVICE) {
            continue;
        }
        const auto target_layers = config.layerIdsForGroup(group.tag).size();
        if (target_layers > 0) {
            group_tags.push_back(group.tag);
            layer_count += target_layers;
        }
    }
    if (group_tags.empty()) {
        return;
    }
#if !USING_CUDA || USE_PPU
    // Replay kernels are CUDA-only; retain the normal LINEAR cache path elsewhere.
    return;
#endif
    const int64_t slot_count =
        runtime_config.use_batch_decode_scheduler ?
            std::max(runtime_config.max_generate_batch_size,
                     runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size) :
            runtime_config.max_generate_batch_size;
    RTP_LLM_CHECK_WITH_INFO(slot_count > 0 && slot_count <= std::numeric_limits<int>::max(),
                            "LINEAR replay requires a positive slot count fitting int");
    RTP_LLM_CHECK_WITH_INFO(sp_config->gen_num_per_cycle < std::numeric_limits<int>::max(),
                            "LINEAR replay verify width overflows int");
    const auto& linear = model_config.linear_attention_config;
    const auto  tp     = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    RTP_LLM_CHECK_WITH_INFO(linear.linear_num_key_heads > 0 && linear.linear_num_value_heads > 0
                                && linear.linear_num_key_heads % tp == 0 && linear.linear_num_value_heads % tp == 0,
                            "LINEAR replay requires head counts divisible by attention TP");
    RTP_LLM_CHECK_WITH_INFO(linear.linear_conv_kernel_dim > 1, "LINEAR replay requires convolution history");
    config.linear_replay = LinearReplayPlan::create(std::move(group_tags),
                                                    slot_count,
                                                    sp_config->gen_num_per_cycle + 1,
                                                    {linear.linear_num_key_heads / tp, linear.linear_key_head_dim},
                                                    {linear.linear_num_value_heads / tp, linear.linear_value_head_dim},
                                                    linear.replay_vector_gate,
                                                    linear.conv_state_dtype,
                                                    layer_count);
}

uint32_t CacheConfigCreator::computeLocalBlockNum(const CacheConfig&                               config,
                                                  const ModelConfig&                               model_config,
                                                  const RuntimeConfig&                             runtime_config,
                                                  const KVCacheConfig&                             kv_cache_config,
                                                  const ParallelismConfig&                         parallelism_config,
                                                  const std::optional<WarmUpResult>&               warm_up_result,
                                                  const std::optional<SpeculativeExecutionConfig>& sp_config,
                                                  BlockTreeCrcWorkspacePlan*                       crc_plan) {
    config.validateCrossModuleInvariants();
    if (crc_plan) {
        *crc_plan = {};
    }
    uint32_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified baseline kv cache block num %d",
                         kv_cache_config.test_block_num);
        block_num = static_cast<uint32_t>(kv_cache_config.test_block_num);
    } else {
        const auto block_budget       = blockBudgetForConfig(config);
        const bool explicit_kv_budget = kv_cache_config.kv_cache_mem_mb > 0;
        const auto default_crc_bytes =
            explicit_kv_budget && !crc_plan ? 0 : blockTreeTransferDeviceMemory(config, kv_cache_config);
        size_t kv_cache_mem_size;
        if (crc_plan && default_crc_bytes > 0) {
            const auto available_bytes = MemoryEvaluationHelper::availableCacheMemory(
                runtime_config, model_config, parallelism_config, warm_up_result, sp_config, &kv_cache_config);
            auto minimum_kv_bytes = block_budget.explicit_pool_reserve_bytes;
            addBudgetBytes(minimum_kv_bytes, block_budget.paged_block_bytes);
            addBudgetBytes(minimum_kv_bytes, block_budget.swa_block_bytes);
            size_t requested_kv_bytes = 0;
            if (explicit_kv_budget) {
                addBudgetBytes(requested_kv_bytes, 1024 * 1024, kv_cache_config.kv_cache_mem_mb);
            }
            const auto reserved_kv_bytes = std::max(minimum_kv_bytes, requested_kv_bytes);
            RTP_LLM_CHECK_WITH_INFO(
                available_bytes > reserved_kv_bytes,
                "available cache memory %zu bytes cannot fit KV cache (%zu bytes) and CRC workspaces",
                available_bytes,
                reserved_kv_bytes);
            const auto hard_limit = available_bytes - reserved_kv_bytes;
            // Ordinary sizing leaves most of the cache budget to KV pools.
            // An explicit pool budget takes priority over CRC concurrency.
            constexpr size_t kAutomaticCrcBudgetDivisor = 8;
            const auto       preferred_limit =
                explicit_kv_budget ? hard_limit : std::min(hard_limit, available_bytes / kAutomaticCrcBudgetDivisor);
            *crc_plan = planBlockTreeCrcWorkspaces(config, kv_cache_config, preferred_limit);
            RTP_LLM_CHECK_WITH_INFO(crc_plan->device_bytes <= hard_limit,
                                    "minimum CRC workspace needs %zu bytes, only %zu bytes remain after KV reservation",
                                    crc_plan->device_bytes,
                                    hard_limit);
            kv_cache_mem_size = explicit_kv_budget ? requested_kv_bytes : available_bytes - crc_plan->device_bytes;
            RTP_LLM_LOG_INFO("CRC transfer plan: batch=%zu slots_per_work_loop=%zu device_bytes=%zu kv_bytes=%zu",
                             crc_plan->batch_size,
                             crc_plan->slots_per_loop,
                             crc_plan->device_bytes,
                             kv_cache_mem_size);
        } else {
            kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(runtime_config,
                                                                             kv_cache_config,
                                                                             model_config,
                                                                             parallelism_config,
                                                                             warm_up_result,
                                                                             sp_config,
                                                                             default_crc_bytes);
        }
        if (block_budget.explicit_pool_reserve_bytes > 0) {
            RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size >= block_budget.explicit_pool_reserve_bytes,
                                    "kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                    "(reduce explicitly sized pool blocks if needed)",
                                    kv_cache_mem_size / 1024 / 1024,
                                    block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
            RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB",
                             kv_cache_mem_size / 1024 / 1024,
                             block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget);
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least one baseline block, got %u", block_num);
    RTP_LLM_CHECK_WITH_INFO(block_num <= config.maxBaselineBlockNum(),
                            "cache baseline=%u exceeds weighted pool int32 block ID bounds",
                            block_num);
    const auto kv_cache_seq_len = static_cast<uint64_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("cache layout created: baseline_blocks=%u, baseline_token_span=%lu, draft_modules=%zu",
                     block_num,
                     kv_cache_seq_len,
                     config.moduleCount() - 1);
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
    const auto layout     = RankLayout::fromParallelismConfig(parallelism_config);
    const auto group_size = static_cast<size_t>(layout.laneStride());
    if (group_size > 1) {
        RTP_LLM_CHECK_WITH_INFO(candidate_block_num <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                                "candidate cache block count exceeds collective int32 range");
        const auto local_rank     = static_cast<size_t>(layout.stageRank());
        auto       block_num_t    = torch::empty({static_cast<int64_t>(group_size)}, torch::kInt32).pin_memory();
        auto       block_num_ptr  = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = static_cast<int>(candidate_block_num);
        execAllGather({{block_num_t}, ParallelMode::STAGE});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        // Main retired the FFN-disaggregation role; the FFN sizing cohort is
        // gone, so the confirmation always runs in the attention role.
        return selectConfirmedBlockNum(block_num_ptr, group_size, /*is_ffn_service=*/false);
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
    number(3);  // Signature format version: 2 introduced the module-scoped
    // layout; 3 folds in the full-tag topology (groups without local layers
    // are recorded with zero baseline capacity).
    number(config.moduleCount());
    for (size_t module_id = 0; module_id < config.moduleCount(); ++module_id) {
        const auto& module_topology = config.topology(module_id);
        const auto& metadata       = config.moduleMetadata(module_id);
        number(static_cast<uint64_t>(metadata.reuse_cache_tail_blocks));
        number(metadata.enable_linear_cache_reclaim);
        number(module_topology.layers().empty());
        number(module_topology.groups().size());
        for (const auto& group : module_topology.groups()) {
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
                number(module_topology.layerIdsForGroup(group.tag).empty() ? 0
                                                                          : config.blockNumForGroup(group, baseline));
            }
        }
    }
    SizingSignature result{};
    // Legacy FFN disaggregation once shared the collective scope with attention
    // ranks while holding a baseline of one, so the signature still encodes the
    // role slot; attention ranks always fill 0. Keep the slot for signature
    // stability across versions.
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
    // Main retired the DP_AND_TP collective scope: the sizing agreement runs
    // over the same stage-scoped lane as the block-num confirmation below
    // (all TP/DP ranks of the current PP stage).
    const auto layout     = RankLayout::fromParallelismConfig(parallelism_config);
    const auto group_size = static_cast<size_t>(layout.laneStride());
    if (group_size <= 1) {
        return candidate_block_num;
    }
    const size_t     local_rank = static_cast<size_t>(layout.stageRank());
    constexpr size_t width      = std::tuple_size<SizingSignature>::value;
    const auto       agree      = [&](uint32_t baseline) {
        // Main retired the FFN-disaggregation role; the sizing agreement
        // always runs in the attention cohort.
        const auto signature = sizingSignature(config, baseline, /*is_ffn_service=*/false);
        // Shape never depends on local pool count or the JSON configuration.
        // Even inconsistent topologies therefore reach the same collective.
        auto  tensor = torch::empty({static_cast<int64_t>(group_size * width)}, torch::kInt32).pin_memory();
        auto* data   = tensor.data_ptr<int32_t>();
        std::copy(signature.begin(), signature.end(), data + local_rank * width);
        execAllGather({{tensor}, ParallelMode::STAGE});
        execSyncCommunication(false);
        cudaSyncAndCheck();
        validateSizingSignatures(data, group_size);
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
