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

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto group_bytes = config.blockSizeBytesForGroup(gid);
        const auto policy      = config.policyForGroup(gid);
        if (policy.explicit_block_num > 0) {
            if (policy.charge_to_paged_budget) {
                budget.explicit_pool_reserve_bytes += static_cast<size_t>(policy.explicit_block_num) * group_bytes;
            }
            continue;
        }
        switch (policy.group_type) {
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

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    const auto            group_num    = static_cast<size_t>(config.groupNums());
    std::vector<uint32_t> group_block_nums(group_num, 0);
    std::vector<size_t>   group_kv_block_stride_bytes(group_num, 0);
    std::vector<size_t>   group_kv_scale_stride_bytes(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = config.specForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count         = static_cast<uint32_t>(config.layerIdsForGroup(gid).size());
        const size_t kernel_kv_stride    = spec->block_size_bytes();
        const auto   kernel_scale        = spec->scale_block_size_bytes();
        // Only compressed specs expose bytes/kernel page; other specs already cover a physical block.
        const size_t group_bpk = spec->type == KVCacheSpecType::OpaqueKV
                                     ? config.kernelBlocksPerKvBlockForGroup(gid) : 1;
        const size_t kv_stride           = kernel_kv_stride * group_bpk;
        const size_t scale_stride        = kernel_scale * group_bpk;
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
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] +=
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

bool hasMultiDescriptorLayer(const ModelConfig& model_config) {
    return std::any_of(model_config.kv_cache_spec_descs.begin(),
                       model_config.kv_cache_spec_descs.end(),
                       [](const auto& descs) { return descs.size() > 1; });
}

bool usesLegacyDescriptorRules(const ModelConfig& model_config) {
    return !model_config.hybrid_attention_config.enable_independent_kv_cache_pools
           && !hasMultiDescriptorLayer(model_config);
}

void validateLegacyDescriptorShape(const ModelConfig& model_config) {
    if (!usesLegacyDescriptorRules(model_config)) {
        return;
    }
    for (size_t layer_id = 0; layer_id < model_config.kv_cache_spec_descs.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs[layer_id].size() == 1,
                                "legacy cache layer %zu must have exactly one descriptor",
                                layer_id);
    }
    if (!model_config.hybrid_attention_config.enable_hybrid_attention) {
        return;
    }

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    for (size_t layer_id = 0; layer_id < types.size(); ++layer_id) {
        const bool expects_linear = types[layer_id] == HybridAttentionType::LINEAR;
        const bool is_linear =
            model_config.kv_cache_spec_descs[layer_id][0].cache_type == KVCacheSpecType::LinearAttention;
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

uint32_t localKvHeadNumForDesc(const KVCacheSpecDesc&   desc,
                               const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config) {
    if (desc.cache_type == KVCacheSpecType::MultiHeadAttention) {
        const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
        const uint32_t tp      = static_cast<uint32_t>(attn_tp);
        const uint32_t kv      = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
        RTP_LLM_CHECK_WITH_INFO(kv > 0, "local kv head num requires positive kv_head_num");
        return kv % tp == 0 ? kv / tp : kv / std::gcd(kv, tp);
    }
    if (desc.cache_type == KVCacheSpecType::LinearAttention) {
        const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
        const uint32_t tp      = static_cast<uint32_t>(attn_tp);
        const uint32_t heads   = static_cast<uint32_t>(model_config.linear_attention_config.linear_num_value_heads);
        RTP_LLM_CHECK_WITH_INFO(
            heads > 0 && heads % tp == 0,
            "linear_num_value_heads must be positive and divisible by attention TP, global=%u tp=%u",
            heads,
            tp);
        return heads / tp;
    }
    return 1;
}

void populateGroups(CacheConfig&                 config,
                    const LayerKVCacheSpecDescs& descs_by_layer,
                    const LayerKVCacheSpecs&     specs_by_layer,
                    const ModelConfig&           model_config,
                    const ParallelismConfig&     parallelism_config,
                    bool                         full_groups_first) {
    struct BuildState {
        KVCacheSpecPtr   spec;
        std::string      fingerprint;
        CacheGroupPolicy policy;
        uint32_t         local_kv_head_num = 1;
        std::vector<int> layer_ids;
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
            const auto policy = SpecBuilder::groupPolicy(desc);
            checkGroupResidencyBudget(policy, spec->tag);
            const auto local_heads = localKvHeadNumForDesc(desc, model_config, parallelism_config);
            auto [it, inserted]    = groups_by_tag.emplace(spec->tag, BuildState{});
            if (inserted) {
                it->second.spec              = spec;
                it->second.fingerprint       = spec->fingerprint();
                it->second.policy            = policy;
                it->second.local_kv_head_num = local_heads;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(it->second.fingerprint == spec->fingerprint(),
                                        "cache tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(it->second.policy, policy),
                                        "cache tag=%s has inconsistent policy",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(it->second.local_kv_head_num == local_heads,
                                        "cache tag=%s has inconsistent local_kv_head_num",
                                        spec->tag.c_str());
            }
            it->second.layer_ids.push_back(static_cast<int>(layer_id));
            layers[layer_id].group_tags.push_back(spec->tag);
        }
    }

    if (full_groups_first) {
        std::stable_partition(ordered_tags.begin(), ordered_tags.end(), [&groups_by_tag](const std::string& tag) {
            return groups_by_tag.at(tag).policy.group_type == CacheGroupType::FULL;
        });
    }

    std::vector<GroupBase> groups;
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto& state = groups_by_tag.at(tag);
        GroupBase   group;
        group.tag                   = tag;
        group.spec                  = state.spec;
        group.policy                = state.policy;
        group.layer_ids             = state.layer_ids;
        group.local_kv_head_num     = state.local_kv_head_num;
        group.kv_block_stride_bytes = state.spec->block_size_bytes();
        group.kv_scale_stride_bytes = state.spec->scale_block_size_bytes();
        groups.push_back(std::move(group));
    }
    config.setTopology(std::move(groups), std::move(layers));
}

CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                  const ParallelismConfig& parallelism_config,
                                  const KVCacheConfig&     kv_cache_config,
                                  bool                     is_mtp,
                                  int                      gen_num_per_cycle) {
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

    const auto seq_size = physical_tokens_per_block;
    const auto kernel_seq_size = kernel_tokens_per_block;
    validateLegacyDescriptorShape(model_config);
    validateDescs(model_config, kernel_seq_size);
    const auto ctx =
        makeSpecBuildContext(model_config, parallelism_config, seq_size, kernel_seq_size, gen_num_per_cycle);
    const auto specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    CacheConfig config;
    config.dtype                   = ctx.dtype;
    config.layer_num               = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num           = config.layer_num;
    config.seq_size_per_block      = seq_size;
    config.kernel_seq_size_per_block = kernel_seq_size;
    config.use_mla                 = model_config.attn_config.use_mla;
    config.is_sparse               = model_config.attn_config.is_sparse;
    config.enable_hybrid_attention = model_config.hybrid_attention_config.enable_hybrid_attention;
    const bool legacy_rules        = usesLegacyDescriptorRules(model_config);
    populateGroups(config,
                   model_config.kv_cache_spec_descs,
                   specs,
                   model_config,
                   parallelism_config,
                   legacy_rules && model_config.hybrid_attention_config.enable_hybrid_attention);
    for (const auto& group : config.topology().groups()) {
        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        config.use_typed_cache_regions |= opaque;
        config.use_opaque_kv_cache_store |= opaque;
        config.is_sparse |= group.spec->type == KVCacheSpecType::OpaqueKV;
    }
    if (legacy_rules) {
        if (!model_config.hybrid_attention_config.enable_hybrid_attention) {
            RTP_LLM_CHECK_WITH_INFO(config.groupNums() == 1,
                                    "single cache config requires one consistent cache group, got %d",
                                    config.groupNums());
        } else {
            const auto full_group_num =
                std::count_if(config.topology().groups().begin(),
                              config.topology().groups().end(),
                              [](const GroupBase& group) { return group.policy.group_type == CacheGroupType::FULL; });
            RTP_LLM_CHECK_WITH_INFO(
                full_group_num <= 1,
                "multiple full attention cache groups (%zu) are not supported: FMHA parameters bind one block table "
                "before the layer loop",
                static_cast<size_t>(full_group_num));
        }

        const auto full_attention_group_num = std::count_if(
            config.topology().groups().begin(), config.topology().groups().end(), [](const GroupBase& group) {
                return group.policy.group_type == CacheGroupType::FULL && group.spec
                       && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
            });
        RTP_LLM_CHECK_WITH_INFO(full_attention_group_num == 1,
                                "cache config requires exactly one FULL MHA/MLA cache group, got %zu",
                                static_cast<size_t>(full_attention_group_num));
    }
    config.disable_decode_first_malloc_device_reuse |= config.use_opaque_kv_cache_store;
    setupIndependentPoolSizes(config, is_mtp);
    return config;
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
    KVCacheConfig default_config;
    default_config.seq_size_per_block        = 0;
    default_config.kernel_seq_size_per_block = 0;
    return createConfigFromDescs(model_config, parallelism_config, default_config, is_mtp, gen_num_per_cycle);
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    return createConfigFromDescs(model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config = createConfigFromDescs(model_config, parallelism_config, kv_cache_config, false, 0);
    config.linear_step = std::max(1, kv_cache_config.linear_step);
    setupKernelSeqSize(config, kv_cache_config, "cache");

    uint32_t block_num = computeBlockNum(
        config, model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

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
    CacheConfig score_config =
        createConfigFromDescs(score_model_config, parallelism_config, kv_cache_config, false, sp_config.gen_num_per_cycle);
    CacheConfig propose_config =
        createConfigFromDescs(propose_model_config, parallelism_config, kv_cache_config, is_mtp, sp_config.gen_num_per_cycle);

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

    const auto         score_budget   = blockBudgetForConfig(score_config);
    const auto         propose_budget = blockBudgetForConfig(propose_config);
    KVCacheBlockBudget joint_budget   = score_budget;
    addBlockBudget(joint_budget, propose_budget, static_cast<size_t>(num_mtp_modules));
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
                score_budget.explicit_pool_reserve_bytes / 1024 / 1024,
                propose_budget.explicit_pool_reserve_bytes / 1024 / 1024,
                num_mtp_modules);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, joint_budget, joint_step);
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config   = score_config;
    config.linear_step   = joint_step;
    config.layer_all_num = score_config.layer_num;
    config.block_size_bytes = total_block_size_bytes;

    const uint32_t main_layer_num = score_config.layer_num;

    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg = config.mergeMTPModule(propose_config, m, main_layer_num);
        sub_cfg->finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    config.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);

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
