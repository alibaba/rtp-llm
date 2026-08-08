#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

bool checkedAdd(size_t lhs, size_t rhs, size_t* result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool checkedMul(size_t lhs, size_t rhs, size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

void validateLinearStep(const KVCacheConfig& kv_cache_config) {
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.linear_step >= 1,
                            "KVCacheConfig linear_step must be at least 1, got %d",
                            kv_cache_config.linear_step);
}

void validateModelBlockGranularity(const ModelConfig& model_config) {
    const auto block_size  = model_config.attn_config.tokens_per_block;
    const auto kernel_size = model_config.attn_config.kernel_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(block_size > 0, "model tokens_per_block must be positive");
    RTP_LLM_CHECK_WITH_INFO(kernel_size > 0 && block_size >= kernel_size && block_size % kernel_size == 0,
                            "model tokens_per_block=%zu must be >= kernel_tokens_per_block=%zu and divisible by it",
                            block_size,
                            kernel_size);
}

void validateRuntimeBlockGranularity(const ModelConfig& model_config, const KVCacheConfig& kv_cache_config) {
    validateModelBlockGranularity(model_config);
    validateLinearStep(kv_cache_config);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config.seq_size_per_block > 0,
                            "KVCacheConfig seq_size_per_block must be positive, got %d",
                            kv_cache_config.seq_size_per_block);
    const auto runtime_seq_size    = static_cast<size_t>(kv_cache_config.seq_size_per_block);
    const auto runtime_kernel_size = kv_cache_config.kernel_seq_size_per_block > 0 ?
                                         static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block) :
                                         model_config.attn_config.kernel_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(runtime_kernel_size > 0,
                            "KVCacheConfig kernel_seq_size_per_block must resolve positive, got %zu",
                            runtime_kernel_size);
    RTP_LLM_CHECK_WITH_INFO(runtime_seq_size == model_config.attn_config.tokens_per_block,
                            "KVCacheConfig seq_size_per_block=%zu does not match projected model tokens_per_block=%zu",
                            runtime_seq_size,
                            model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(
        runtime_kernel_size == model_config.attn_config.kernel_tokens_per_block,
        "KVCacheConfig kernel_seq_size_per_block=%zu does not match projected model kernel_tokens_per_block=%zu",
        runtime_kernel_size,
        model_config.attn_config.kernel_tokens_per_block);
}

void validateDescriptors(const ModelConfig& model_config, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "cache desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(
        gen_num_per_cycle >= 0, "cache desc config requires non-negative gen_num_per_cycle, got %d", gen_num_per_cycle);
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "cache desc config layer %ld has no descs", layer_id);
        for (const auto& desc : descs) {
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED
                || desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "cache desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

LayerKVCacheSpecBuildResults
buildLayerSpecs(const ModelConfig& model_config, const ParallelismConfig& parallelism_config, int gen_num_per_cycle) {
    validateDescriptors(model_config, gen_num_per_cycle);
    SpecBuildContext ctx;
    ctx.dtype                   = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    ctx.seq_size_per_block      = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);

    LayerKVCacheSpecBuildResults layer_specs(model_config.kv_cache_spec_descs.size());
    for (size_t layer_id = 0; layer_id < model_config.kv_cache_spec_descs.size(); ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[layer_id];
        auto&       specs = layer_specs[layer_id];
        specs.reserve(descs.size());
        for (const auto& desc : descs) {
            specs.push_back(SpecBuilder::build(desc, ctx));
        }
    }
    return layer_specs;
}

std::pair<std::vector<GroupTopology>, std::vector<LayerTopology>>
buildTopology(const ModelConfig& model_config, const ParallelismConfig& parallelism_config, int gen_num_per_cycle) {
    validateModelBlockGranularity(model_config);
    const auto layer_specs = buildLayerSpecs(model_config, parallelism_config, gen_num_per_cycle);

    struct GroupBuildState {
        KVCacheSpecPtr   spec;
        std::string      fingerprint;
        CacheGroupPolicy policy;
        uint32_t         local_kv_head_num = 1;
        std::vector<int> layer_ids;
    };
    std::map<std::string, GroupBuildState> group_by_tag;
    std::vector<std::string>               ordered_tags;

    for (size_t layer_id = 0; layer_id < layer_specs.size(); ++layer_id) {
        std::set<std::string> layer_tags;
        for (const auto& [spec, policy] : layer_specs[layer_id]) {
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache layer %zu has null spec", layer_id);
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(spec->tag).second,
                                    "cache layer %zu has duplicate tag=%s",
                                    layer_id,
                                    spec->tag.c_str());
            const auto local_kv_head_num = resolveLocalKVHeadNum(
                spec->type, model_config.attn_config, model_config.linear_attention_config, parallelism_config);
            auto [it, inserted] = group_by_tag.emplace(spec->tag, GroupBuildState{});
            if (inserted) {
                it->second.spec              = spec;
                it->second.fingerprint       = spec->layoutFingerprint();
                it->second.policy            = policy;
                it->second.local_kv_head_num = local_kv_head_num;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(it->second.fingerprint == spec->layoutFingerprint(),
                                        "cache tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(it->second.policy, policy),
                                        "cache tag=%s has inconsistent policy",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(it->second.local_kv_head_num == local_kv_head_num,
                                        "cache tag=%s has inconsistent local_kv_head_num",
                                        spec->tag.c_str());
            }
            it->second.layer_ids.push_back(static_cast<int>(layer_id));
        }
    }

    std::vector<GroupTopology> groups;
    std::vector<LayerTopology> layers(layer_specs.size());
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto&   state = group_by_tag.at(tag);
        GroupTopology group;
        group.tag                   = tag;
        group.spec                  = state.spec;
        group.policy                = state.policy;
        group.layer_ids             = state.layer_ids;
        group.local_kv_head_num     = state.local_kv_head_num;
        group.kv_block_stride_bytes = state.spec->block_size_bytes();
        group.kv_scale_stride_bytes = state.spec->scale_block_size_bytes();
        groups.push_back(group);
        for (int layer_id : state.layer_ids) {
            layers[static_cast<size_t>(layer_id)].group_tags.push_back(tag);
        }
        if (group.policy.group_type == CacheGroupType::FULL) {
            RTP_LLM_CHECK_WITH_INFO(
                group.spec->kernel_seq_size_per_block
                    == static_cast<uint32_t>(model_config.attn_config.kernel_tokens_per_block),
                "FULL cache descriptor tag=%s kernel_seq_size_per_block=%u does not match model value=%zu",
                tag.c_str(),
                group.spec->kernel_seq_size_per_block,
                model_config.attn_config.kernel_tokens_per_block);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "cache config produced no cache specs");
    const bool has_linear_group         = std::any_of(groups.begin(), groups.end(), [](const GroupTopology& group) {
        return group.policy.group_type == CacheGroupType::LINEAR;
    });
    const bool has_full_attention_group = std::any_of(groups.begin(), groups.end(), [](const GroupTopology& group) {
        return group.policy.group_type == CacheGroupType::FULL && group.spec
               && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                   || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
    });
    RTP_LLM_CHECK_WITH_INFO(!has_linear_group || has_full_attention_group,
                            "linear cache config requires at least one FULL MHA/MLA cache group");
    return {std::move(groups), std::move(layers)};
}

const GroupTopology* findGroup(const std::vector<GroupTopology>& groups, const std::string& tag) {
    const auto it =
        std::find_if(groups.begin(), groups.end(), [&tag](const GroupTopology& group) { return group.tag == tag; });
    return it == groups.end() ? nullptr : &*it;
}

bool isFullAttentionSpec(KVCacheSpecType type) {
    return type == KVCacheSpecType::MultiHeadAttention || type == KVCacheSpecType::MultiHeadLatentAttention;
}

std::string cacheGroupPolicySummary(const CacheGroupPolicy& policy) {
    std::ostringstream stream;
    stream << "{group_type=" << cacheGroupTypeName(policy.group_type) << ", prefix_reuse=" << policy.enable_prefix_reuse
           << ", evict=" << static_cast<int>(policy.evict_policy) << ", reservable=" << policy.reservable
           << ", explicit_block_num=" << policy.explicit_block_num
           << ", active_tail_blocks=" << policy.active_tail_blocks
           << ", validate_tail_blocks=" << policy.validate_tail_blocks
           << ", cp_mapping=" << static_cast<int>(policy.cp_mapping)
           << ", cp_slice=" << static_cast<int>(policy.cp_slice) << '}';
    return stream.str();
}

std::string groupSummary(const std::vector<GroupTopology>& groups) {
    std::ostringstream stream;
    stream << '[';
    for (size_t gid = 0; gid < groups.size(); ++gid) {
        if (gid != 0) {
            stream << ", ";
        }
        const auto& group = groups[gid];
        stream << "{tag=" << group.tag << ", group_type=" << cacheGroupTypeName(group.policy.group_type);
        if (group.spec != nullptr) {
            stream << ", spec_type=" << static_cast<int>(group.spec->type)
                   << ", dtype=" << static_cast<int>(group.spec->memoryLayoutDType())
                   << ", seq_size_per_block=" << group.spec->seq_size_per_block
                   << ", kernel_seq_size_per_block=" << group.spec->kernel_seq_size_per_block
                   << ", block_size_bytes=" << group.spec->block_size_bytes()
                   << ", scale_block_size_bytes=" << group.spec->scale_block_size_bytes();
        } else {
            stream << ", spec=null";
        }
        stream << ", kv_block_stride_bytes=" << group.kv_block_stride_bytes
               << ", kv_scale_stride_bytes=" << group.kv_scale_stride_bytes
               << ", policy=" << cacheGroupPolicySummary(group.policy) << '}';
    }
    stream << ']';
    return stream.str();
}

std::optional<std::string> resolveDefaultMtpGroupAlias(const std::vector<GroupTopology>& target_groups,
                                                       const std::vector<GroupTopology>& propose_groups) {
    if (propose_groups.size() != 1 || propose_groups.front().tag != "default") {
        return std::nullopt;
    }
    if (findGroup(target_groups, "default") != nullptr) {
        return std::nullopt;  // Exact tag matching remains authoritative.
    }

    const auto& source_group = propose_groups.front();
    if (source_group.policy.group_type != CacheGroupType::FULL || source_group.spec == nullptr
        || !isFullAttentionSpec(source_group.spec->type)) {
        return std::nullopt;
    }

    std::vector<std::string> candidates;
    for (const auto& target_group : target_groups) {
        if (target_group.spec != nullptr && CacheConfig::samePolicy(target_group.policy, source_group.policy)
            && target_group.spec->layoutFingerprint() == source_group.spec->layoutFingerprint()
            && target_group.spec->seq_size_per_block == source_group.spec->seq_size_per_block
            && target_group.spec->kernel_seq_size_per_block == source_group.spec->kernel_seq_size_per_block
            && target_group.kv_block_stride_bytes == source_group.kv_block_stride_bytes
            && target_group.kv_scale_stride_bytes == source_group.kv_scale_stride_bytes) {
            candidates.push_back(target_group.tag);
        }
    }

    if (candidates.empty()) {
        const auto targets = groupSummary(target_groups);
        RTP_LLM_FAIL("no compatible target group for sole MTP draft tag=default: source_spec_type=%d "
                     "source_dtype=%d source_seq_size_per_block=%u source_kernel_seq_size_per_block=%u "
                     "source_block_size_bytes=%zu source_scale_block_size_bytes=%zu "
                     "source_kv_block_stride_bytes=%zu source_kv_scale_stride_bytes=%zu source_policy=%s "
                     "target_groups=%s",
                     static_cast<int>(source_group.spec->type),
                     static_cast<int>(source_group.spec->memoryLayoutDType()),
                     source_group.spec->seq_size_per_block,
                     source_group.spec->kernel_seq_size_per_block,
                     source_group.spec->block_size_bytes(),
                     source_group.spec->scale_block_size_bytes(),
                     source_group.kv_block_stride_bytes,
                     source_group.kv_scale_stride_bytes,
                     cacheGroupPolicySummary(source_group.policy).c_str(),
                     targets.c_str());
    }
    if (candidates.size() > 1) {
        const auto targets = groupSummary(target_groups);
        RTP_LLM_FAIL("ambiguous MTP default FULL group mapping: compatible target groups=%zu spec_type=%d "
                     "target_groups=%s",
                     candidates.size(),
                     static_cast<int>(source_group.spec->type),
                     targets.c_str());
    }
    return candidates.front();
}

std::string groupTags(const std::vector<GroupTopology>& groups) {
    std::ostringstream stream;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << groups[i].tag;
    }
    return stream.str();
}

std::pair<std::vector<GroupTopology>, std::vector<LayerTopology>>
mergeMtpModule(std::vector<GroupTopology>&       target_groups,
               std::vector<LayerTopology>&       target_layers,
               const std::vector<GroupTopology>& propose_groups,
               uint32_t                          mtp_layer_num,
               int                               module_index,
               uint32_t                          main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "invalid MTP module_index=%d", module_index);
    const size_t total_layers =
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
    target_layers.resize(total_layers);
    for (size_t layer_id = 0; layer_id < target_layers.size(); ++layer_id) {
        target_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    const auto default_alias_target = resolveDefaultMtpGroupAlias(target_groups, propose_groups);
    for (const auto& propose_group : propose_groups) {
        const bool has_exact_group    = findGroup(target_groups, propose_group.tag) != nullptr;
        const bool uses_default_alias = propose_group.tag == "default" && default_alias_target.has_value();
        RTP_LLM_CHECK_WITH_INFO(has_exact_group || uses_default_alias,
                                "MTP draft cache tag=%s has no exact or compatible main-model mapping; main tags=[%s]",
                                propose_group.tag.c_str(),
                                groupTags(target_groups).c_str());
    }

    std::vector<GroupTopology> sub_groups;
    std::vector<LayerTopology> sub_layers(mtp_layer_num);
    sub_groups.reserve(target_groups.size());
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        sub_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }

    for (auto& target_group : target_groups) {
        const GroupTopology* source_group = findGroup(propose_groups, target_group.tag);
        const bool           uses_default_alias =
            source_group == nullptr && default_alias_target.has_value() && target_group.tag == *default_alias_target;
        if (uses_default_alias) {
            source_group = &propose_groups.front();
        }
        GroupTopology sub_group = source_group == nullptr ? target_group : *source_group;
        sub_group.layer_ids.clear();
        if (source_group != nullptr) {
            RTP_LLM_CHECK_WITH_INFO(target_group.spec->layoutFingerprint() == source_group->spec->layoutFingerprint()
                                        && CacheConfig::samePolicy(target_group.policy, source_group->policy)
                                        && target_group.kv_block_stride_bytes == source_group->kv_block_stride_bytes
                                        && target_group.kv_scale_stride_bytes == source_group->kv_scale_stride_bytes,
                                    "MTP incompatible group tag=%s",
                                    target_group.tag.c_str());
            if (uses_default_alias) {
                auto aliased_spec = source_group->spec->clone();
                aliased_spec->tag = target_group.tag;
                sub_group.tag     = target_group.tag;
                sub_group.spec    = std::move(aliased_spec);
                RTP_LLM_LOG_INFO("aliases sole MTP draft tag=default to compatible main tag=%s for module=%d",
                                 target_group.tag.c_str(),
                                 module_index);
            }
            RTP_LLM_CHECK_WITH_INFO(source_group->layer_ids.size() == mtp_layer_num,
                                    "MTP group source_tag=%s target_tag=%s must cover every module layer, "
                                    "got=%zu expected=%u",
                                    source_group->tag.c_str(),
                                    target_group.tag.c_str(),
                                    source_group->layer_ids.size(),
                                    mtp_layer_num);
            // buildTopology emits ordered membership. A matched propose group that covers all module layers is
            // exactly [0, mtp_layer_num), and every prior module has appended one entry per local layer.
            for (size_t local = 0; local < source_group->layer_ids.size(); ++local) {
                const auto global =
                    CacheConfig::mtpGlobalLayerId(main_layer_num, module_index, mtp_layer_num, static_cast<int>(local));
                RTP_LLM_CHECK_WITH_INFO(global != std::numeric_limits<uint32_t>::max(), "MTP global layer id overflow");
                sub_group.layer_ids.push_back(static_cast<int>(local));
                sub_layers[local].group_tags.push_back(target_group.tag);
                target_group.layer_ids.push_back(static_cast<int>(global));
                target_layers[global].group_tags.push_back(target_group.tag);
            }
        }
        sub_groups.push_back(std::move(sub_group));
    }
    // propose_groups is non-empty, all propose tags belong to target_groups, and each matched group covers all
    // module layers, so every sub-layer receives at least one group tag above.
    return {std::move(sub_groups), std::move(sub_layers)};
}

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
    budget.explicit_pool_reserve_bytes = config.explicitlySizedPoolReserveBytes();
    for (const auto& group : config.topology().groups()) {
        if (group.policy.explicit_block_num > 0) {
            continue;
        }
        size_t stride     = 0;
        size_t slot_bytes = 0;
        RTP_LLM_CHECK_WITH_INFO(checkedAdd(group.kv_block_stride_bytes, group.kv_scale_stride_bytes, &stride)
                                    && checkedMul(group.layer_ids.size(), stride, &slot_bytes),
                                "kv cache slot bytes overflow for group tag=%s",
                                group.tag.c_str());
        size_t* destination =
            group.policy.group_type == CacheGroupType::SWA ? &budget.swa_block_bytes : &budget.paged_block_bytes;
        size_t next = 0;
        RTP_LLM_CHECK_WITH_INFO(checkedAdd(*destination, slot_bytes, &next),
                                "kv cache block budget overflow for group tag=%s",
                                group.tag.c_str());
        *destination = next;
    }
    return budget;
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");
    uint32_t low  = 0;
    uint32_t high = static_cast<uint32_t>(std::numeric_limits<BlockIdxType>::max());
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

uint32_t CacheConfigCreator::localBlockNum(const KVCacheBlockBudget& budget,
                                           size_t                    total_budget_bytes,
                                           int                       test_block_num,
                                           int                       linear_step,
                                           bool                      sentinel_only) {
    if (sentinel_only) {
        return 1;
    }
    if (test_block_num != 0) {
        RTP_LLM_CHECK_WITH_INFO(
            test_block_num >= 2, "KVCacheConfig test_block_num must be 0 or at least 2, got %d", test_block_num);
        return static_cast<uint32_t>(test_block_num);
    }
    const uint32_t block_num = maxKVCacheBlockNumForBudget(total_budget_bytes, budget, linear_step);
    RTP_LLM_CHECK_WITH_INFO(
        block_num >= 2,
        "kv cache budget is insufficient for 2 total slots: budget=%zu explicit=%zu paged=%zu SWA=%zu",
        total_budget_bytes,
        budget.explicit_pool_reserve_bytes,
        budget.paged_block_bytes,
        budget.swa_block_bytes);
    return block_num;
}

void CacheConfigCreator::publishLocalBlockNum(
    int* block_nums, size_t world_size, int64_t world_rank, uint32_t local_block_num, bool sentinel_only) {
    RTP_LLM_CHECK_WITH_INFO(block_nums != nullptr, "block num publish buffer is null");
    RTP_LLM_CHECK_WITH_INFO(world_rank >= 0 && static_cast<size_t>(world_rank) < world_size,
                            "invalid world_rank=%ld for world_size=%zu",
                            static_cast<long>(world_rank),
                            world_size);
    RTP_LLM_CHECK_WITH_INFO(local_block_num <= static_cast<uint32_t>(std::numeric_limits<BlockIdxType>::max()),
                            "local block num exceeds BlockIdxType: %u",
                            local_block_num);
    block_nums[world_rank] = sentinel_only ? std::numeric_limits<int>::max() : static_cast<int>(local_block_num);
}

uint32_t CacheConfigCreator::selectConvergedBlockNum(const int* block_nums,
                                                     size_t     world_size,
                                                     uint32_t   local_block_num,
                                                     bool       sentinel_only) {
    RTP_LLM_CHECK_WITH_INFO(block_nums != nullptr, "block num convergence buffer is null");
    RTP_LLM_CHECK_WITH_INFO(world_size > 0, "block num convergence requires at least one rank");
    if (sentinel_only) {
        return local_block_num;
    }
    const int converged_block_num = *std::min_element(block_nums, block_nums + world_size);
    RTP_LLM_CHECK_WITH_INFO(converged_block_num != std::numeric_limits<int>::max(),
                            "block num convergence received only sentinel ranks");
    RTP_LLM_CHECK_WITH_INFO(converged_block_num >= 0, "invalid converged block num=%d", converged_block_num);
    return static_cast<uint32_t>(converged_block_num);
}

uint32_t CacheConfigCreator::convergeBlockNum(uint32_t                 local_block_num,
                                              const ParallelismConfig& parallelism_config,
                                              bool                     sentinel_only) {
    const int64_t world_size = parallelism_config.world_size;
    const int64_t world_rank = parallelism_config.world_rank;
    RTP_LLM_CHECK_WITH_INFO(world_size > 0, "invalid world_size=%ld", static_cast<long>(world_size));
    RTP_LLM_CHECK_WITH_INFO(world_rank >= 0 && world_rank < world_size,
                            "invalid world_rank=%ld for world_size=%ld",
                            static_cast<long>(world_rank),
                            static_cast<long>(world_size));
    if (world_size == 1) {
        return local_block_num;
    }

    auto  block_nums = torch::full({world_size}, std::numeric_limits<int>::max(), torch::kInt32).pin_memory();
    auto* data       = block_nums.data_ptr<int>();
    // FFN-service ranks publish a sentinel so they participate without reducing attention-rank capacity.
    publishLocalBlockNum(data, static_cast<size_t>(world_size), world_rank, local_block_num, sentinel_only);
    execAllGather({{block_nums}, ParallelMode::DP_AND_TP});
    execSyncCommunication(false);
    cudaSyncAndCheck();
    return selectConvergedBlockNum(data, static_cast<size_t>(world_size), local_block_num, sentinel_only);
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool /*is_mtp*/,
                                                  int gen_num_per_cycle) {
    auto [groups, layers] = buildTopology(model_config, parallelism_config, gen_num_per_cycle);
    return CacheConfig(static_cast<uint32_t>(model_config.num_layers),
                       static_cast<uint32_t>(model_config.num_layers),
                       model_config.attn_config.use_mla,
                       model_config.attn_config.is_sparse,
                       model_config.hybrid_attention_config.enable_hybrid_attention,
                       static_cast<size_t>(model_config.attn_config.tokens_per_block),
                       std::move(groups),
                       std::move(layers));
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    validateRuntimeBlockGranularity(model_config, kv_cache_config);
    auto config                = createBasicConfig(model_config, parallelism_config, false, 0);
    config.linear_step         = kv_cache_config.linear_step;
    const bool   sentinel_only = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    const size_t budget =
        sentinel_only || kv_cache_config.test_block_num != 0 ?
            0 :
            MemoryEvaluationHelper::getKVCacheMemorySize(
                runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    const auto     block_budget = blockBudgetForConfig(config);
    const uint32_t local_block_num =
        localBlockNum(block_budget, budget, kv_cache_config.test_block_num, config.linear_step, sentinel_only);
    const uint32_t block_num = convergeBlockNum(local_block_num, parallelism_config, sentinel_only);
    RTP_LLM_CHECK_WITH_INFO(
        sentinel_only || block_num >= 2, "normal kv cache requires at least 2 total slots, got %u", block_num);
    const size_t usable_tokens = sentinel_only ? 0 : static_cast<size_t>(block_num - 1) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("kv cache plan: explicit=%zu paged=%zu swa=%zu step=%d block_num=%u usable_tokens=%zu",
                     block_budget.explicit_pool_reserve_bytes,
                     block_budget.paged_block_bytes,
                     block_budget.swa_block_bytes,
                     config.linear_step,
                     block_num,
                     usable_tokens);
    for (const auto& group : config.topology().groups()) {
        if (group.policy.explicit_block_num == 0) {
            continue;
        }
        RTP_LLM_LOG_INFO("kv cache plan: group=%s explicit_blocks=%u reserve_bytes=%zu",
                         group.tag.c_str(),
                         group.policy.explicit_block_num,
                         config.explicitReserveBytesForGroup(group.tag));
    }
    if (!sentinel_only && usable_tokens < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache has %u total slots and can store %zu usable tokens, less than max_seq_len %ld",
                            block_num,
                            usable_tokens,
                            model_config.max_seq_len);
    }
    if (sentinel_only) {
        config.publishSentinelOnlyBlockNum();
    } else {
        config.finalizeBlockNums(block_num, runtime_config);
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
    validateRuntimeBlockGranularity(score_model_config, kv_cache_config);
    validateRuntimeBlockGranularity(propose_model_config, kv_cache_config);
    RTP_LLM_CHECK_WITH_INFO(score_model_config.attn_config.tokens_per_block
                                == propose_model_config.attn_config.tokens_per_block,
                            "MTP global seq_size_per_block mismatch: main=%zu propose=%zu",
                            score_model_config.attn_config.tokens_per_block,
                            propose_model_config.attn_config.tokens_per_block);
    auto [score_groups, score_layers] =
        buildTopology(score_model_config, parallelism_config, sp_config.gen_num_per_cycle);
    auto propose_topology = buildTopology(propose_model_config, parallelism_config, sp_config.gen_num_per_cycle);
    int  module_num       = is_mtp ? sp_config.gen_num_per_cycle : 1;
    if (is_eagle) {
        module_num = 1;
    }
    RTP_LLM_CHECK_WITH_INFO(
        module_num > 0, "speculative cache requires at least one propose module, got %d", module_num);
    size_t     mtp_layers   = 0;
    size_t     total_layers = 0;
    const bool valid_mtp =
        checkedMul(static_cast<size_t>(module_num), static_cast<size_t>(propose_model_config.num_layers), &mtp_layers)
        && checkedAdd(static_cast<size_t>(score_model_config.num_layers), mtp_layers, &total_layers)
        && total_layers <= static_cast<size_t>(std::numeric_limits<int>::max());
    RTP_LLM_CHECK_WITH_INFO(valid_mtp,
                            "speculative cache layer count overflow: score=%ld propose=%ld modules=%d",
                            score_model_config.num_layers,
                            propose_model_config.num_layers,
                            module_num);
    std::vector<std::shared_ptr<CacheConfig>> sub_configs;
    sub_configs.reserve(static_cast<size_t>(module_num));
    for (int module = 0; module < module_num; ++module) {
        auto [sub_groups, sub_layers] = mergeMtpModule(score_groups,
                                                       score_layers,
                                                       propose_topology.first,
                                                       static_cast<uint32_t>(propose_model_config.num_layers),
                                                       module,
                                                       static_cast<uint32_t>(score_model_config.num_layers));
        sub_configs.push_back(std::shared_ptr<CacheConfig>(
            new CacheConfig(static_cast<uint32_t>(propose_model_config.num_layers),
                            static_cast<uint32_t>(propose_model_config.num_layers),
                            propose_model_config.attn_config.use_mla,
                            propose_model_config.attn_config.is_sparse,
                            propose_model_config.hybrid_attention_config.enable_hybrid_attention,
                            static_cast<size_t>(propose_model_config.attn_config.tokens_per_block),
                            std::move(sub_groups),
                            std::move(sub_layers))));
    }
    const auto layer_all_num     = static_cast<uint32_t>(total_layers);
    auto       score_config      = CacheConfig(static_cast<uint32_t>(score_model_config.num_layers),
                                    layer_all_num,
                                    score_model_config.attn_config.use_mla,
                                    score_model_config.attn_config.is_sparse,
                                    score_model_config.hybrid_attention_config.enable_hybrid_attention,
                                    static_cast<size_t>(score_model_config.attn_config.tokens_per_block),
                                    std::move(score_groups),
                                    std::move(score_layers));
    score_config.mtp_sub_configs = std::move(sub_configs);
    score_config.linear_step     = kv_cache_config.linear_step;
    for (auto& sub_config : score_config.mtp_sub_configs) {
        sub_config->linear_step = score_config.linear_step;
    }

    const bool   sentinel_only = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    const size_t budget =
        sentinel_only || kv_cache_config.test_block_num != 0 ?
            0 :
            MemoryEvaluationHelper::getKVCacheMemorySize(
                runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);
    const auto     block_budget = blockBudgetForConfig(score_config);
    const uint32_t local_block_num =
        localBlockNum(block_budget, budget, kv_cache_config.test_block_num, score_config.linear_step, sentinel_only);
    const uint32_t block_num = convergeBlockNum(local_block_num, parallelism_config, sentinel_only);
    RTP_LLM_CHECK_WITH_INFO(sentinel_only || block_num >= 2,
                            "normal speculative kv cache requires at least 2 total slots, got %u",
                            block_num);
    RTP_LLM_LOG_INFO("speculative kv cache plan: layers=%u explicit=%zu paged=%zu swa=%zu step=%d block_num=%u",
                     score_config.layer_all_num,
                     block_budget.explicit_pool_reserve_bytes,
                     block_budget.paged_block_bytes,
                     block_budget.swa_block_bytes,
                     score_config.linear_step,
                     block_num);
    for (const auto& group : score_config.topology().groups()) {
        if (group.policy.explicit_block_num == 0) {
            continue;
        }
        RTP_LLM_LOG_INFO("speculative kv cache plan: group=%s explicit_blocks=%u reserve_bytes=%zu",
                         group.tag.c_str(),
                         group.policy.explicit_block_num,
                         score_config.explicitReserveBytesForGroup(group.tag));
    }
    if (sentinel_only) {
        score_config.publishSentinelOnlyBlockNum();
    } else {
        score_config.finalizeBlockNums(block_num, runtime_config);
    }
    return score_config;
}

}  // namespace rtp_llm
