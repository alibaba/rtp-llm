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
#include "rtp_llm/cpp/config/RankLayout.h"
#include "rtp_llm/cpp/config/StageLayerRange.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

bool modelHasMultipleCacheGroups(const ModelConfig& model_config) {
    std::set<std::string> tags;
    for (const auto& layer_descs : model_config.kv_cache_spec_descs) {
        for (const auto& desc : layer_descs) {
            tags.insert(desc.tag);
            if (tags.size() > 1) {
                return true;
            }
        }
    }
    return false;
}

// HCA and FlashMLA compressed pools require a 128-token kernel quantum.
// Qwen4Exp's indexer_kv instead compresses groups of four tokens and supports
// the existing 64-token ATOM page size without a partial group.
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
            case CacheGroupType::REQUEST_STATE:
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
                const bool qwen4_atom_indexer =
                    (model_config.model_type == "atom_qwen4_exp" || model_config.model_type == "atom_qwen4_exp_mtp")
                    && desc.tag == "indexer_kv" && desc.compression_ratio == 4;
                if (desc.compression_ratio > 1 && !qwen4_atom_indexer) {
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

LayerKVCacheSpecDescs resolveLinearStateLayout(const ModelConfig&       model,
                                               const ParallelismConfig& parallel,
                                               const KVCacheConfig&     options,
                                               const SpecBuildContext&  ctx) {
    auto       descs = model.kv_cache_spec_descs;
    const bool supported_topology =
        parallel.get_attn_tp_size() == 2 && parallel.pp_size == 1 && !parallel.prefill_cp_config.is_enabled()
        && !parallel.prefill_cp_config.is_prefill_enabled() && !parallel.prefill_cp_config.kv_cache_sharded
        && parallel.prefill_cp_config.prefill_cp_size <= 1;
    RTP_LLM_CHECK_WITH_INFO(!(options.linear_ssm_double_buffer && options.linear_ssm_replay),
                            "linear_ssm_double_buffer and linear_ssm_replay are mutually exclusive");
    const bool     split       = options.linear_ssm_double_buffer || options.linear_ssm_replay;
    const uint32_t state_slots = options.linear_ssm_replay ? 1 : 2;
    if (split) {
        RTP_LLM_CHECK_WITH_INFO(parallel.role_type == RoleType::DECODE && supported_topology && parallel.dp_size == 1
                                    && !parallel.ffn_disaggregate_config.enable_ffn_disaggregate && !options.reuse_cache
                                    && !options.enable_remote_cache && ctx.gen_num_per_cycle > 0
                                    && ctx.gen_num_per_cycle <= 3,
                                "split linear SSM requires D, TP2/CP1/DP1, no FFN disaggregation, "
                                "reuse off and chained MTP "
                                "with 1 to 3 draft tokens");
        RTP_LLM_CHECK_WITH_INFO(options.linear_pool_blocks > state_slots,
                                "split linear SSM requires linear_pool_blocks > %u "
                                "(SSM slots per request plus the reserved null slot)",
                                state_slots);
        RTP_LLM_CHECK_WITH_INFO(ctx.seq_size_per_block >= ctx.gen_num_per_cycle + 1,
                                "split linear SSM requires a page span at least as large as the verify width");
    }
    bool declared = false;
    for (auto& layer : descs) {
        std::vector<KVCacheSpecDesc> resolved;
        for (auto desc : layer) {
            if (desc.linear_state_layout != LinearStateLayout::SPLIT_EXPORT) {
                RTP_LLM_CHECK_WITH_INFO(!split || desc.cache_type != KVCacheSpecType::LinearAttention,
                                        "model must declare the split linear-state layout on every linear layer");
                resolved.push_back(std::move(desc));
                continue;
            }
            RTP_LLM_CHECK_WITH_INFO(desc.cache_type == KVCacheSpecType::LinearAttention && desc.tag == "linear",
                                    "split linear state requires the model's linear attention descriptor");
            declared = true;
            if (!split) {
                if (!supported_topology) {
                    desc.linear_state_layout = LinearStateLayout::LEGACY;
                }
                resolved.push_back(std::move(desc));
                continue;
            }
            const auto  legacy = SpecBuilder::build(desc, ctx);
            const auto& linear = model.linear_attention_config;
            RTP_LLM_CHECK_WITH_INFO(linear.ssm_state_dtype == DataType::TYPE_FP32
                                        && linear.conv_state_dtype == DataType::TYPE_BF16
                                        && linear.linear_key_head_dim == 128 && linear.linear_value_head_dim == 128,
                                    "hybrid replay requires FP32 128x128 SSM and BF16 convolution state");
            const uint64_t requests = (options.linear_pool_blocks - 1) / state_slots;
            const uint64_t query    = static_cast<uint64_t>(ctx.gen_num_per_cycle) + 1;
            // Conv keeps two history slots and Q candidates. malloc is serialized
            // and cleans up each single-sequence request before allocating the
            // next, so one additional rollover block covers the transient peak.
            const uint64_t conv_blocks = 2 + requests * (query + 2);
            RTP_LLM_CHECK_WITH_INFO(conv_blocks <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                                    "linear convolution pool capacity overflows int");
            auto region = [&](const std::string& tag,
                              LinearStateLayout  layout,
                              DataType           dtype,
                              size_t             elems,
                              uint32_t           entries,
                              CacheGroupType     type,
                              uint32_t           slots,
                              uint32_t           blocks) {
                RTP_LLM_CHECK_WITH_INFO(elems > 0 && elems <= std::numeric_limits<uint32_t>::max(),
                                        "linear state entry width is out of range");
                KVCacheSpecDesc result;
                result.tag                  = tag;
                result.cache_type           = KVCacheSpecType::OpaqueState;
                result.is_state_cache       = true;
                result.dtype                = dtype;
                result.entry_dtype          = dtype;
                result.entry_elems          = static_cast<uint32_t>(elems);
                result.explicit_entry_count = entries;
                result.group_type           = type;
                result.linear_state_layout  = layout;
                result.request_state_slots  = slots;
                result.reuse                = CacheReusePolicyDesc{false};
                result.capacity             = CacheCapacityPolicyDesc{true, blocks, true};
                return result;
            };
            resolved.push_back(region("linear",
                                      LinearStateLayout::SPLIT_CONV,
                                      linear.conv_state_dtype,
                                      legacy->v_block_size_bytes() / getTypeSize(linear.conv_state_dtype),
                                      1,
                                      CacheGroupType::LINEAR,
                                      0,
                                      static_cast<uint32_t>(conv_blocks)));
            resolved.push_back(
                region("linear_ssm",
                       options.linear_ssm_replay ? LinearStateLayout::SINGLE_SSM : LinearStateLayout::DOUBLE_BUFFER_SSM,
                       linear.ssm_state_dtype,
                       legacy->k_block_size_bytes() / getTypeSize(linear.ssm_state_dtype),
                       1,
                       CacheGroupType::REQUEST_STATE,
                       state_slots,
                       options.linear_pool_blocks));
            const size_t record_elems =
                static_cast<size_t>(linear.linear_num_value_heads / 2) * 3 * linear.linear_value_head_dim;
            resolved.push_back(region("linear_replay",
                                      LinearStateLayout::SPLIT_REPLAY,
                                      DataType::TYPE_FP32,
                                      record_elems,
                                      static_cast<uint32_t>(query),
                                      CacheGroupType::REQUEST_STATE,
                                      1,
                                      static_cast<uint32_t>(requests + 1)));
        }
        layer = std::move(resolved);
    }
    RTP_LLM_CHECK_WITH_INFO(!split || declared, "model has not declared support for split linear SSM");
    return descs;
}

CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                  const ParallelismConfig& parallelism_config,
                                  const KVCacheConfig&     kv_cache_config,
                                  int                      gen_num_per_cycle,
                                  uint32_t                 global_layer_begin) {
    const auto [seq_size, kernel_seq_size] = resolveSeqSizes(model_config, kv_cache_config);
    validateDescs(model_config, kernel_seq_size);
    validateAttentionMetadata(model_config);
    const auto ctx =
        makeSpecBuildContext(model_config, parallelism_config, seq_size, kernel_seq_size, gen_num_per_cycle);
    const auto descs = resolveLinearStateLayout(model_config, parallelism_config, kv_cache_config, ctx);
    const auto specs = CacheConfigCreator::buildLayerSpecsFromDescs(descs, ctx, model_config.num_layers);

    CacheConfig config;
    config.dtype                   = ctx.dtype;
    config.layer_num               = static_cast<uint32_t>(model_config.num_layers);
    config.global_layer_begin      = global_layer_begin;
    config.cache_min_replay_tokens = std::max(1, model_config.cache_min_replay_tokens);
    config.cache_key_hash_seed     = model_config.cache_key_hash_seed;

    config.seq_size_per_block = seq_size;
    config.use_mla            = model_config.attn_config.use_mla;
    config.is_sparse          = model_config.attn_config.is_sparse;
    populateGroups(config, descs, specs, model_config);
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

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget) {
    if (budget.paged_block_bytes == 0 && budget.swa_block_bytes == 0) {
        return budget.explicit_pool_reserve_bytes <= total_budget_bytes ? 1u : 0u;
    }

    uint32_t low  = 0;
    uint32_t high = std::numeric_limits<uint32_t>::max();
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

ModelConfig CacheConfigCreator::stageScopedModelConfig(const ModelConfig&       model_config,
                                                       const ParallelismConfig& parallelism_config,
                                                       bool                     is_draft_model) {
    if (parallelism_config.pp_size <= 1 || is_draft_model) {
        return model_config;
    }

    const auto range = stageLayerRange(parallelism_config, parallelism_config.pp_rank, model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);

    ModelConfig stage_config = model_config;
    stage_config.num_layers  = range.size;
    stage_config.kv_cache_spec_descs.assign(model_config.kv_cache_spec_descs.begin() + range.begin,
                                            model_config.kv_cache_spec_descs.begin() + range.end());

    auto& attention_types = stage_config.hybrid_attention_config.hybrid_attention_types;
    if (!attention_types.empty()) {
        RTP_LLM_CHECK_WITH_INFO(attention_types.size() == static_cast<size_t>(model_config.num_layers),
                                "hybrid_attention_types size %zu != num_layers %ld",
                                attention_types.size(),
                                model_config.num_layers);
        attention_types.assign(model_config.hybrid_attention_config.hybrid_attention_types.begin() + range.begin,
                               model_config.hybrid_attention_config.hybrid_attention_types.begin() + range.end());
    }
    return stage_config;
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle,
                                                  uint32_t                 global_layer_begin) {
    auto config =
        createConfigFromDescs(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, global_layer_begin);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.reuse_cache_tail_blocks > 0, "reuse_cache_tail_blocks must be positive");
    config.reuse_cache_tail_blocks = kv_cache_config.reuse_cache_tail_blocks;
    auto groups                    = config.topology().groups();
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
    config.setTopology(std::move(groups), config.topology().layers());
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
                                                   int                      gen_num_per_cycle,
                                                   int                      request_batch_size) {
    RTP_LLM_CHECK_WITH_INFO(request_batch_size > 0, "warmup batch must be positive");
    const auto stage_model_config = stageScopedModelConfig(model_config, parallelism_config);
    const auto range = stageLayerRange(parallelism_config, parallelism_config.pp_rank, model_config.num_layers);
    auto       config =
        createBasicConfig(stage_model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, range.begin);
    config.model_has_multiple_cache_groups = modelHasMultipleCacheGroups(model_config);
    config.finalizeBlockNums(2, RuntimeConfig{});
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.block_num = 2;
        // Unlike paged KV, writable request state cannot alias block 0 across
        // batch rows. Reserve distinct physical slots for the warmup batch,
        // without retaining the production-sized explicit pool.
        if (group.policy.group_type == CacheGroupType::REQUEST_STATE) {
            const uint64_t count = 1 + static_cast<uint64_t>(request_batch_size) * group.policy.request_state_slots;
            RTP_LLM_CHECK_WITH_INFO(count <= group.policy.explicit_block_num,
                                    "warmup batch exceeds the configured request-state capacity");
            group.block_num = static_cast<uint32_t>(count);
        } else if (group.policy.linear_state_layout == LinearStateLayout::DOUBLE_BUFFER_CONV) {
            group.block_num = 1 + static_cast<uint32_t>(request_batch_size);
        }
    }
    config.setTopology(std::move(groups), config.topology().layers());
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
    if (kv_cache_config.linear_ssm_double_buffer || kv_cache_config.linear_ssm_replay) {
        RTP_LLM_CHECK_WITH_INFO(sp_config && sp_config->type == SP_TYPE_MTP && is_mtp && !is_eagle
                                    && sp_config->tree_decode_config.empty() && draft_model_config != nullptr,
                                "split linear SSM requires a chained MTP target/draft configuration");
    }
    const int  gen_num_per_cycle  = draft_model_config != nullptr ? sp_config->gen_num_per_cycle : 0;
    const auto stage_model_config = stageScopedModelConfig(model_config, parallelism_config);
    const auto range = stageLayerRange(parallelism_config, parallelism_config.pp_rank, model_config.num_layers);
    auto       config =
        createBasicConfig(stage_model_config, parallelism_config, kv_cache_config, gen_num_per_cycle, range.begin);
    config.model_has_multiple_cache_groups = modelHasMultipleCacheGroups(model_config);
    if (draft_model_config != nullptr) {
        const auto layout = RankLayout::fromParallelismConfig(parallelism_config);
        RTP_LLM_CHECK_WITH_INFO(layout.isLastStage(), "draft cache configuration is only valid on the last PP stage");
        auto draft_model = stageScopedModelConfig(*draft_model_config, parallelism_config, true);
        auto draft_options                     = kv_cache_config;
        draft_options.linear_ssm_double_buffer = false;
        draft_options.linear_ssm_replay        = false;
        auto draft       = createBasicConfig(
            draft_model, parallelism_config, draft_options, gen_num_per_cycle, /*global_layer_begin=*/0);
        if (is_mtp && !is_eagle && sp_config->type != SP_TYPE_DSPARK) {
            RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle > 0, "draft cache configuration requires at least one module");
        }
        // The executor and PD transfer use only draft module 0. Proposal width
        // still sizes speculative state; it does not count active draft models.
        config.mtp_sub_configs.push_back(config.mergeMTPModule(draft, /*module_index=*/0, config.layer_num));
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
        auto block_budget = blockBudgetForConfig(config);
        addBudgetBytes(block_budget.paged_block_bytes, model_config.auxiliary_paged_cache_bytes_per_block);
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

        return selectConfirmedBlockNum(block_num_ptr, group_size);
    }
    return candidate_block_num;
}

uint32_t CacheConfigCreator::selectConfirmedBlockNum(const int* candidates, size_t count) {
    RTP_LLM_CHECK_WITH_INFO(candidates != nullptr && count > 0, "cross-rank cache candidates must not be empty");
    const auto confirmed = *std::min_element(candidates, candidates + count);
    RTP_LLM_CHECK_WITH_INFO(confirmed > 0, "cross-rank cache block count must be positive");
    return static_cast<uint32_t>(confirmed);
}

}  // namespace rtp_llm
