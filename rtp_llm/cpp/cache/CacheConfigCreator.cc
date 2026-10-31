#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>

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

    const auto safe_step  = static_cast<uint32_t>(std::max(1, step));
    const auto swa_blocks = block_num / safe_step + (block_num % safe_step != 0 ? 1u : 0u);
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
                addBudgetBytes(budget.paged_block_bytes, group_bytes);
                break;
            case CacheGroupType::SWA:
                addBudgetBytes(budget.swa_block_bytes, group_bytes);
                break;
        }
    }
    return budget;
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
        KVCacheSpecPtr                      spec;
        std::string                         fingerprint;
        CacheGroupPolicy                    policy;
        std::set<std::pair<bool, uint32_t>> offload_runtime_groups;
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
            if (policy.offload.enabled) {
                // An unnamed storage must not alias an explicitly named group.
                it->second.offload_runtime_groups.emplace(desc.offload_runtime_group.has_value(),
                                                          desc.offload_runtime_group.value_or(layer_id));
            }
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
        if (state.policy.offload.enabled) {
            config.offload_runtime_counts[tag] = state.offload_runtime_groups.size();
        }
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
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle) {
    auto config        = createConfigFromDescs(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    config.linear_step = std::max(1, kv_cache_config.linear_step);
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
        int  num_mtp_modules = is_mtp && !is_eagle && sp_config->type != SP_TYPE_DSPARK ? gen_num_per_cycle : 1;
        RTP_LLM_CHECK_WITH_INFO(num_mtp_modules > 0, "draft cache configuration requires at least one module");
        config.mtp_sub_configs.reserve(static_cast<size_t>(num_mtp_modules));
        for (int module = 0; module < num_mtp_modules; ++module) {
            config.mtp_sub_configs.push_back(config.mergeMTPModule(draft, module, config.layer_num));
        }
    }

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
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
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
    const auto kv_cache_seq_len = static_cast<uint64_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("cache layout created: candidate_blocks=%u, candidate_tokens=%lu, draft_modules=%zu",
                     block_num,
                     kv_cache_seq_len,
                     config.mtp_sub_configs.size());
    if (kv_cache_seq_len < static_cast<uint64_t>(model_config.max_seq_len)) {
        RTP_LLM_LOG_WARNING("candidate cache capacity %lu tokens is below max_seq_len %ld tokens",
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    // Only the baseline candidate is available here. Group capacities are
    // applied after runtime cross-rank agreement, before any pool allocation.
    return block_num;
}

TieredCacheCapacity CacheConfigCreator::configureOffloadCapacity(CacheConfig&         config,
                                                                 const KVCacheConfig& kv_cache_config,
                                                                 size_t               gpu_bytes,
                                                                 size_t               max_write_rows) {
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.enable_sparse_kv_offload, "tiered cache capacity requires sparse offload");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.sparse_kv_offload_host_mb >= 0
                                && kv_cache_config.sparse_kv_offload_test_hbm_blocks >= 0,
                            "sparse KV offload capacities must be non-negative");
    TieredCacheBudget budget;
    budget.gpu_bytes       = gpu_bytes;
    budget.swa_step        = std::max(1, config.linear_step);
    budget.test_hbm_blocks = kv_cache_config.sparse_kv_offload_test_hbm_blocks;
    addBudgetBytes(budget.host_bytes, 1024 * 1024, kv_cache_config.sparse_kv_offload_host_mb);
    if (kv_cache_config.test_block_num > 0) {
        budget.max_blocks = kv_cache_config.test_block_num;
    }
    std::map<std::string, TieredCacheConfig> tiered;
    for (const auto& group : config.groups()) {
        const auto& policy      = group.policy;
        const auto  block_bytes = config.blockSizeBytesForGroup(group.tag);
        if (!policy.offload.enabled) {
            if (policy.memory_placement != CacheMemoryPlacement::DEVICE) {
                continue;
            }
            if (policy.explicit_block_num > 0) {
                addBudgetBytes(budget.fixed_gpu_bytes, block_bytes, policy.explicit_block_num);
            } else if (policy.group_type == CacheGroupType::SWA) {
                addBudgetBytes(budget.swa_block_bytes, block_bytes);
            } else {
                addBudgetBytes(budget.device_block_bytes, block_bytes);
            }
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(policy.group_type == CacheGroupType::FULL && policy.explicit_block_num == 0
                                    && policy.memory_placement == CacheMemoryPlacement::DEVICE,
                                "offload tag=%s requires a paged FULL group",
                                group.tag.c_str());
        const auto pages  = group.kernelBlocksPerKvBlock();
        const auto layout = group.spec->entryLayout();
        layout.validate(group.kvBlockStrideBytes() / pages, group.kvScaleStrideBytes() / pages);
        const uint32_t page_rows = layout.rows_per_page;
        RTP_LLM_CHECK_WITH_INFO(pages <= UINT32_MAX / page_rows, "offload entry count overflow");
        const uint32_t          rows     = page_rows * pages;
        const uint32_t          selected = policy.offload.max_selected_rows;
        const TieredCacheConfig capacity{0, 0, selected};
        const uint64_t          resident = capacity.residentRows(page_rows);
        RTP_LLM_CHECK_WITH_INFO(
            selected > 0 && resident < static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
            "offload tag=%s needs a positive selection fitting int32 after page alignment, S=%u page=%u",
            group.tag.c_str(),
            selected,
            page_rows);
        budget.max_blocks =
            std::min<uint32_t>(budget.max_blocks, (std::numeric_limits<int32_t>::max() - resident) / rows);
        const size_t owners        = config.layerIdsForGroup(group.tag).size();
        const auto   runtime_count = config.offload_runtime_counts.find(group.tag);
        const size_t runtimes = runtime_count == config.offload_runtime_counts.end() ? owners : runtime_count->second;
        RTP_LLM_CHECK_WITH_INFO(runtimes > 0 && runtimes <= owners,
                                "offload tag=%s has invalid runtime count=%zu for %zu storages",
                                group.tag.c_str(),
                                runtimes,
                                owners);
        addBudgetBytes(budget.offload_block_bytes, block_bytes);
        RTP_LLM_CHECK_WITH_INFO(
            block_bytes % pages == 0, "offload tag=%s has a partial page stride", group.tag.c_str());
        addBudgetBytes(budget.fixed_gpu_bytes, block_bytes / pages, resident / page_rows);
        // Shared selection state is per runtime; native KV and pack staging are per storage.
        addBudgetBytes(budget.metadata_block_bytes, 8 * static_cast<size_t>(rows), runtimes);
        addBudgetBytes(budget.metadata_block_bytes, sizeof(int64_t));  // generations shared by tag
        size_t runtime_fixed = 0;
        addBudgetBytes(runtime_fixed, 20, resident);
        addBudgetBytes(runtime_fixed, 12, selected);
        addBudgetBytes(runtime_fixed, 32);
        addBudgetBytes(budget.fixed_gpu_bytes, runtime_fixed, runtimes);
        size_t pack_indices = 0;
        addBudgetBytes(pack_indices, 12, max_write_rows);
        addBudgetBytes(budget.fixed_gpu_bytes, pack_indices, owners);
        addBudgetBytes(budget.fixed_gpu_bytes, block_bytes / pages, (max_write_rows + page_rows - 1) / page_rows);
        budget.max_selected_rows = std::max(budget.max_selected_rows, selected);
        tiered.emplace(group.tag, capacity);
    }
    RTP_LLM_CHECK_WITH_INFO(!tiered.empty(), "model declares no sparse KV offload groups");
    const auto result = solveTieredCacheCapacity(budget);
    for (auto& [tag, capacity] : tiered) {
        capacity.total_blocks    = result.config.total_blocks;
        capacity.hbm_blocks      = result.config.hbm_blocks;
        const auto&  group       = config.group(tag);
        const size_t block_bytes = config.blockSizeBytesForGroup(tag);
        const size_t page_rows   = group.spec->entryLayout().rows_per_page;
        const size_t resident    = capacity.residentRows(page_rows);
        const size_t hot_bytes   = (block_bytes / group.kernelBlocksPerKvBlock()) * (resident / page_rows);
        RTP_LLM_LOG_INFO("sparse KV offload tag=%s H=%u P=%u N=%u R=%u S=%u host_bytes=%zu resident_bytes=%zu",
                         tag.c_str(),
                         capacity.hbm_blocks,
                         capacity.total_blocks - capacity.hbm_blocks,
                         capacity.total_blocks,
                         static_cast<uint32_t>(resident),
                         capacity.max_selected_rows,
                         (capacity.total_blocks - capacity.hbm_blocks) * block_bytes,
                         capacity.hbm_blocks * block_bytes + hot_bytes);
    }
    config.tiered_cache_configs = std::move(tiered);
    RTP_LLM_LOG_INFO(
        "sparse KV offload bytes: GPU=%zu host=%zu HBM_backing=%zu device_only=%zu metadata=%zu fixed=%zu SWA=%zu",
        result.gpu_bytes,
        result.host_bytes,
        result.hbm_bytes,
        result.device_bytes,
        result.metadata_bytes,
        result.fixed_bytes,
        result.swa_bytes);
    return result;
}

uint32_t CacheConfigCreator::computeLocalOffloadConfig(CacheConfig&                       config,
                                                       const ModelConfig&                 model_config,
                                                       const RuntimeConfig&               runtime_config,
                                                       const KVCacheConfig&               kv_cache_config,
                                                       const ParallelismConfig&           parallelism_config,
                                                       const std::optional<WarmUpResult>& warm_up_result) {
    const auto bytes = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, std::nullopt);
    RTP_LLM_CHECK_WITH_INFO(runtime_config.max_generate_batch_size > 0,
                            "offload requires a positive decode batch size");
    const size_t query_tokens = std::max<int64_t>(1, model_config.gen_num_per_cycle + 1);
    RTP_LLM_CHECK_WITH_INFO(query_tokens <= std::numeric_limits<size_t>::max()
                                                / static_cast<size_t>(runtime_config.max_generate_batch_size),
                            "sparse KV offload write capacity overflow");
    return configureOffloadCapacity(
               config, kv_cache_config, bytes, runtime_config.max_generate_batch_size * query_tokens)
        .config.total_blocks;
}

uint32_t CacheConfigCreator::synchronizeOffloadConfig(CacheConfig&             config,
                                                      const ParallelismConfig& parallelism_config) {
    RTP_LLM_CHECK_WITH_INFO(!config.tiered_cache_configs.empty(), "missing sparse KV offload capacity");
    const auto&    local = config.tiered_cache_configs.begin()->second;
    const uint32_t hbm   = synchronizeBlockNum(local.hbm_blocks, parallelism_config);
    // Encode P + 1 because the existing capacity agreement requires positive counts.
    const uint32_t host = synchronizeBlockNum(local.total_blocks - local.hbm_blocks + 1, parallelism_config) - 1;
    RTP_LLM_CHECK_WITH_INFO(hbm >= 2, "sparse KV offload requires two HBM blocks on every rank");
    for (auto& [tag, capacity] : config.tiered_cache_configs) {
        capacity.hbm_blocks   = hbm;
        capacity.total_blocks = hbm + host;
    }
    for (auto& sub : config.mtp_sub_configs) {
        sub->tiered_cache_configs = config.tiered_cache_configs;
    }
    return hbm + host;
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
