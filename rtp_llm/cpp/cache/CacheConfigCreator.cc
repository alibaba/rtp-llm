#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
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
        const size_t group_bytes    = config.blockSizeBytesForGroup(group.tag);
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
                                      int                      gen_num_per_cycle,
                                      const ResolvedStateEntryLayouts* state_entry_layouts,
                                      const ResolvedSwaByteLayouts* swa_byte_layouts,
                                      const ResolvedCompressedLayouts* compressed_layouts,
                                      const RequiredSpecProjectionTags* required_spec_tags) {
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
    ctx.state_entry_layouts     = state_entry_layouts;
    ctx.swa_byte_layouts        = swa_byte_layouts;
    ctx.compressed_layouts      = compressed_layouts;
    ctx.required_spec_projection_tags = required_spec_tags;
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
                    const ModelConfig&           model_config,
                    const ResolvedCpGroupPolicies* cp_group_policies) {
    RTP_LLM_CHECK_WITH_INFO(cp_group_policies == nullptr || cp_group_policies->size() == descs_by_layer.size(),
                            "resolved CP group policy layer count differs from descriptors");
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
        RTP_LLM_CHECK_WITH_INFO(cp_group_policies == nullptr || (*cp_group_policies)[layer_id].size() == descs.size(),
                                "resolved CP group policy descriptor count differs at layer %u", layer_id);
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
            if (cp_group_policies != nullptr) {
                const auto& cp_policy = (*cp_group_policies)[layer_id][i];
                policy.cp_mapping = cp_policy.mapping;
                policy.cp_slice = cp_policy.slice;
            }
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

RequiredSpecProjectionTags resolveRequiredSpecProjectionTags(const ModelConfig& model) {
    RequiredSpecProjectionTags tags;
    for (const auto& layer : model.kv_cache_spec_descs) {
        for (const auto& desc : layer) {
            if (desc.cache_type != KVCacheSpecType::OpaqueKV && desc.cache_type != KVCacheSpecType::OpaqueState) {
                continue;
            }
            const bool scale = desc.cp.has_value() && desc.cp->scale_seq_size.value_or(false);
            const bool align = desc.cp.has_value() && desc.cp->align_payload.value_or(false);
            const auto slice = desc.cp.has_value() ?
                                   desc.cp->prefill_slice_layout.value_or(CpPrefillSliceLayout::NONE) :
                                   CpPrefillSliceLayout::NONE;
            if (scale || align || slice != CpPrefillSliceLayout::NONE) {
                tags.insert(desc.tag);
            }
        }
    }
    return tags;
}

ResolvedCpGroupPolicies resolveCpGroupPolicies(const ModelConfig& model) {
    ResolvedCpGroupPolicies policies;
    policies.reserve(model.kv_cache_spec_descs.size());
    for (const auto& layer : model.kv_cache_spec_descs) {
        auto& resolved_layer = policies.emplace_back();
        resolved_layer.reserve(layer.size());
        for (const auto& desc : layer) {
            ResolvedCpGroupPolicy policy;
            // Preserve the former FULL/SWA mapping defaults in cache policy resolution.
            switch (SpecBuilder::groupType(desc)) {
                case CacheGroupType::FULL:
                    policy.mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
                    break;
                case CacheGroupType::SWA:
                    policy.mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
                    break;
                case CacheGroupType::LINEAR:
                    policy.mapping = CpBlockMappingMode::NONE;
                    break;
            }
            if (desc.cp.has_value()) {
                policy.mapping = desc.cp->mapping.value_or(policy.mapping);
                policy.slice   = desc.cp->slice.value_or(policy.slice);
                policy.uses_physical_projection =
                    (desc.cache_type == KVCacheSpecType::OpaqueKV || desc.cache_type == KVCacheSpecType::OpaqueState)
                    && (desc.cp->scale_seq_size.value_or(false) || desc.cp->align_payload.value_or(false)
                        || desc.cp->prefill_slice_layout.value_or(CpPrefillSliceLayout::NONE)
                               != CpPrefillSliceLayout::NONE);
            }
            resolved_layer.push_back(policy);
        }
    }
    return policies;
}

LayerKVCacheSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs&   layer_descs,
                                           const SpecBuildContext&        ctx,
                                           int64_t                        expected_layer_num,
                                           const ResolvedCpGroupPolicies* cp_group_policies);

CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                  const ParallelismConfig& parallelism_config,
                                  const KVCacheConfig&     kv_cache_config,
                                  int                      gen_num_per_cycle) {
    const auto [seq_size, kernel_seq_size] = resolveSeqSizes(model_config, kv_cache_config);
    validateDescs(model_config, kernel_seq_size);
    validateAttentionMetadata(model_config);
    const auto state_entry_layouts = FixedStateCacheSpec::resolveEntryLayouts(
        model_config.kv_cache_spec_descs, parallelism_config, seq_size, gen_num_per_cycle, kernel_seq_size);
    const auto swa_byte_layouts = FixedStateCacheSpec::resolveSwaByteLayouts(
        model_config.kv_cache_spec_descs, parallelism_config, seq_size, gen_num_per_cycle, kernel_seq_size);
    const auto compressed_layouts = CompressedKVCacheSpec::resolveLayouts(
        model_config.kv_cache_spec_descs, parallelism_config, seq_size, kernel_seq_size, gen_num_per_cycle);
    const auto required_spec_tags = resolveRequiredSpecProjectionTags(model_config);
    const auto cp_group_policies  = resolveCpGroupPolicies(model_config);
    const auto ctx                = makeSpecBuildContext(model_config,
                                          parallelism_config,
                                          seq_size,
                                          kernel_seq_size,
                                          gen_num_per_cycle,
                                          &state_entry_layouts,
                                          &swa_byte_layouts,
                                          &compressed_layouts,
                                          &required_spec_tags);
    const auto specs =
        buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers, &cp_group_policies);

    CacheConfig config;
    config.dtype                   = ctx.dtype;
    config.layer_num               = static_cast<uint32_t>(model_config.num_layers);
    config.cache_min_replay_tokens = std::max(1, model_config.cache_min_replay_tokens);
    config.cache_key_hash_seed     = model_config.cache_key_hash_seed;

    config.seq_size_per_block = seq_size;
    config.use_mla            = model_config.attn_config.use_mla;
    config.is_sparse          = model_config.attn_config.is_sparse;
    populateGroups(config, model_config.kv_cache_spec_descs, specs, model_config, &cp_group_policies);
    for (const auto& entry : state_entry_layouts) {
        if (config.group(entry.first).policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            config.state_entry_tags.insert(entry.first);
        }
    }
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

uint32_t CacheConfigCreator::resolvedBaseBlockTokens(const ModelConfig& model_config,
                                                     const KVCacheConfig& kv_cache_config) {
    return resolveSeqSizes(model_config, kv_cache_config).first;
}

uint32_t CacheConfigCreator::resolvedKernelBlockTokens(const ModelConfig& model_config,
                                                       const KVCacheConfig& kv_cache_config) {
    return resolveSeqSizes(model_config, kv_cache_config).second;
}

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

namespace {

LayerKVCacheSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs&   layer_descs,
                                           const SpecBuildContext&        ctx,
                                           int64_t                        expected_layer_num,
                                           const ResolvedCpGroupPolicies* cp_group_policies) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(expected_layer_num),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            layer_descs.size(),
                            expected_layer_num);
    LayerKVCacheSpecs layer_specs(layer_descs.size());
    RTP_LLM_CHECK_WITH_INFO(cp_group_policies == nullptr || cp_group_policies->size() == layer_descs.size(),
                            "resolved CP group policy layer count differs from descriptors");
    const RequiredSpecProjectionTags no_projection_tags;
    for (size_t layer_id = 0; layer_id < layer_descs.size(); ++layer_id) {
        const auto& descs = layer_descs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "kv_cache_spec_descs layer %zu has no descs", layer_id);
        RTP_LLM_CHECK_WITH_INFO(cp_group_policies == nullptr || (*cp_group_policies)[layer_id].size() == descs.size(),
                                "resolved CP group policy descriptor count differs at layer %zu", layer_id);
        auto& specs = layer_specs[layer_id];
        specs.reserve(descs.size());
        for (size_t desc_id = 0; desc_id < descs.size(); ++desc_id) {
            auto local_ctx = ctx;
            if (cp_group_policies != nullptr && !(*cp_group_policies)[layer_id][desc_id].uses_physical_projection) {
                local_ctx.state_entry_layouts = nullptr;
                local_ctx.swa_byte_layouts = nullptr;
                local_ctx.compressed_layouts = nullptr;
                local_ctx.required_spec_projection_tags = &no_projection_tags;
            }
            specs.push_back(SpecBuilder::build(descs[desc_id], local_ctx));
        }
    }
    return layer_specs;
}

}  // namespace

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
    config.finalizeStorageLayout(parallelism_config, /*warmup=*/true);
    // Upstream warmup pools reserve the sentinel block plus one allocatable
    // block per group, and never inherit linear_step: the SWA
    // ceil(baseline/step) shrink would collapse those groups to sentinel-only.
    config.linear_step = 1;
    config.finalizeBlockNums(2, RuntimeConfig{});
    return config;
}

CacheDomainConfig CacheConfigCreator::createDomainConfig(const ModelConfig&       model_config,
                                                         const ParallelismConfig& parallelism_config,
                                                         const KVCacheConfig&     kv_cache_config,
                                                         const std::optional<SpeculativeExecutionConfig>& sp_config,
                                                         const ModelConfig* draft_model_config,
                                                         bool               is_mtp,
                                                         bool               is_eagle) {
    RTP_LLM_CHECK_WITH_INFO(draft_model_config == nullptr || sp_config.has_value(),
                            "draft cache configuration requires speculative execution configuration");
    RTP_LLM_CHECK_WITH_INFO(
        draft_model_config == nullptr
            || (sp_config->gen_num_per_cycle > 0 && sp_config->gen_num_per_cycle <= std::numeric_limits<int>::max()),
        "draft proposal token count must fit a positive int");
    const int                gen_num_per_cycle = draft_model_config == nullptr ? 0 : sp_config->gen_num_per_cycle;
    std::vector<CacheConfig> modules;
    modules.push_back(createBasicConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle));
    modules.front().finalizeStorageLayout(parallelism_config);
    if (draft_model_config != nullptr) {
        auto draft = createBasicConfig(*draft_model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
        draft.finalizeStorageLayout(parallelism_config);
        // Proposal count controls billed slots. Execution still binds only
        // the first draft module, including DSPARK's feature-cache commits.
        const int draft_count = !is_eagle && (is_mtp || sp_config->type == SP_TYPE_DSPARK) ? gen_num_per_cycle : 1;
        for (int slot = 0; slot < draft_count; ++slot) {
            modules.push_back(draft);
        }
    }
    return CacheDomainConfig(std::move(modules), draft_model_config == nullptr ? 1 : 2);
}

KVCacheBlockBudget CacheConfigCreator::blockBudget(const CacheDomainConfig& domain) {
    KVCacheBlockBudget total;
    for (size_t index = 0; index < domain.moduleCount(); ++index) {
        const auto leaf = blockBudgetForConfig(domain.module(CacheModuleId(index)));
        addBudgetBytes(total.explicit_pool_reserve_bytes, leaf.explicit_pool_reserve_bytes);
        addBudgetBytes(total.paged_block_bytes, leaf.paged_block_bytes);
        addBudgetBytes(total.swa_block_bytes, leaf.swa_block_bytes);
    }
    return total;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&       model_config,
                                             const ParallelismConfig& parallelism_config,
                                             const KVCacheConfig&     kv_cache_config) {
    return createDomainConfig(model_config, parallelism_config, kv_cache_config).main();
}

uint32_t CacheConfigCreator::computeLocalBlockNum(const CacheConfig&                               config,
                                                  const ModelConfig&                               model_config,
                                                  const RuntimeConfig&                             runtime_config,
                                                  const KVCacheConfig&                             kv_cache_config,
                                                  const ParallelismConfig&                         parallelism_config,
                                                  const std::optional<WarmUpResult>&               warm_up_result,
                                                  const std::optional<SpeculativeExecutionConfig>& sp_config) {
    return computeLocalBlockNum(CacheDomainConfig(config),
                                model_config,
                                runtime_config,
                                kv_cache_config,
                                parallelism_config,
                                warm_up_result,
                                sp_config);
}

uint32_t CacheConfigCreator::computeLocalBlockNum(const CacheDomainConfig&                         config,
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
        const auto block_budget = blockBudget(config);
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
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, config.main().linear_step);
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least one baseline block, got %u", block_num);
    const auto kv_cache_seq_len = static_cast<uint64_t>(block_num) * config.main().seq_size_per_block;
    RTP_LLM_LOG_INFO("cache layout created: candidate_blocks=%u, candidate_tokens=%lu, draft_modules=%zu",
                     block_num,
                     kv_cache_seq_len,
                     config.moduleCount() - 1);
    if (kv_cache_seq_len < static_cast<uint64_t>(model_config.max_seq_len)) {
        RTP_LLM_LOG_WARNING("candidate cache capacity %lu tokens is below max_seq_len %ld tokens",
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
