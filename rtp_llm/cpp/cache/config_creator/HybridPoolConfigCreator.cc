#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

uint32_t fixedRegionCpSize(const ParallelismConfig& parallelism_config) {
    if (!parallelism_config.prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    if (parallelism_config.role_type == RoleType::PREFILL && parallelism_config.tp_size > 1) {
        return static_cast<uint32_t>(parallelism_config.tp_size);
    }
    if (parallelism_config.role_type == RoleType::DECODE && parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        RTP_LLM_CHECK_WITH_INFO(
            parallelism_config.prefill_cp_config.prefill_cp_size > 1,
            "fixed/SWA CP sharding decode requires explicit prefill_cp_size when PREFILL_CP and kv_cache_sharded are enabled");
        return static_cast<uint32_t>(parallelism_config.prefill_cp_config.prefill_cp_size);
    }
    return 1;
}

bool isPrefillCpSliced(const ParallelismConfig& parallelism_config) {
    return parallelism_config.role_type == RoleType::PREFILL && fixedRegionCpSize(parallelism_config) > 1;
}

CacheGroupPolicy policyFromSpecDesc(const KVCacheSpecDesc& desc) {
    CacheGroupPolicy policy = defaultCacheGroupPolicy(SpecBuilder::groupType(desc));
    if (desc.is_state_cache) {
        policy.evict_policy = CacheEvictPolicy::INDEPENDENT;
    }
    if (desc.skip_prefix_reuse) {
        policy.reuse_policy         = CacheReusePolicy::NON_REUSABLE;
        policy.active_tail_blocks   = 1;
        policy.validate_tail_blocks = false;
    }
    if (desc.has_reuse_policy) {
        policy.reuse_policy = desc.reuse_policy;
    }
    if (desc.has_evict_policy) {
        policy.evict_policy = desc.evict_policy;
    }
    if (desc.has_active_tail_blocks) {
        policy.active_tail_blocks = desc.active_tail_blocks;
    }
    if (desc.has_validate_tail_blocks) {
        policy.validate_tail_blocks = desc.validate_tail_blocks;
    }
    policy.explicit_block_num        = desc.extra.explicit_block_num;
    policy.reserve_from_paged_budget = desc.extra.reserve_from_paged_budget;
    if (desc.has_prefix_reusable) {
        policy.prefix_reusable = desc.prefix_reusable;
    }
    policy.uses_pinned_cpu_backing = desc.uses_pinned_cpu_backing;
    if (desc.has_is_cp_shardable) {
        policy.is_cp_shardable = desc.is_cp_shardable;
    }
    if (desc.cache_type == CacheType::COMPRESSED_KV) {
        policy.has_sparse_slots = true;
    }
    if (desc.has_sparse_slots) {
        policy.has_sparse_slots = desc.sparse_slots;
    }
    if (desc.has_kernel_block_subdiv) {
        policy.has_kernel_block_subdiv = desc.kernel_block_subdiv;
    }
    if (desc.has_cp_compact_tail_blocks) {
        policy.cp_compact_tail_blocks = desc.cp_compact_tail_blocks;
    }
    if (desc.has_is_reservable) {
        policy.is_reservable = desc.is_reservable;
    }
    return policy;
}

void validateHybridPoolDescs(const ModelConfig& model_config,
                             uint32_t           kernel_tokens_per_block,
                             int                gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
                            model_config.kv_cache_spec_descs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "hybrid-pool desc config requires non-negative gen_num_per_cycle, got %d",
                            gen_num_per_cycle);

    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& layer_descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!layer_descs.empty(),
                                "hybrid-pool desc config layer %ld has no descs",
                                layer_id);
        for (const auto& desc : layer_descs) {
            RTP_LLM_CHECK_WITH_INFO(
                desc.cache_type != CacheType::MHA || desc.num_kv_heads > 0,
                "hybrid-pool MHA desc tag=%s missing num_kv_heads (must be set by Python model)",
                desc.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                desc.cache_type != CacheType::LINEAR || (desc.num_k_heads > 0 && desc.num_v_heads > 0),
                "hybrid-pool LINEAR desc tag=%s missing num_k_heads/num_v_heads (must be set by Python model)",
                desc.tag.c_str());
            if (desc.extra.derive_entries_from_kernel_block) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0,
                                        "desc tag=%s derives entries from kernel block but kernel_tokens_per_block is 0",
                                        desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_tokens_per_block);
            }
            if (desc.extra.state_ring_compression_ratio > 0) {
                RTP_LLM_CHECK_WITH_INFO(desc.extra.state_ring_compression_ratio > 0,
                                        "state ring desc tag=%s requires positive state_ring_compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

void populateGroupsFromLayerSpecs(CacheConfig&                  config,
                                  const LayerKVCacheSpecDescs& layer_descs,
                                  const LayerKVCacheSpecs&     layer_specs) {
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
        std::vector<int> layers;
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
            const auto policy   = policyFromSpecDesc(desc);
            const auto type     = SpecBuilder::groupType(desc);
            auto       group_it = group_by_tag.find(spec->tag);
            if (group_it == group_by_tag.end()) {
                GroupBuildState state;
                state.spec        = spec;
                state.fingerprint = spec->fingerprint();
                state.type        = type;
                state.policy      = policy;
                group_it          = group_by_tag.emplace(spec->tag, std::move(state)).first;
                ordered_tags.push_back(spec->tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(group_it->second.fingerprint == spec->fingerprint(),
                                        "hybrid-pool tag=%s has multiple physical prototypes",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group_it->second.type == type,
                                        "hybrid-pool tag=%s has inconsistent group type",
                                        spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group_it->second.policy, policy),
                                        "hybrid-pool tag=%s has inconsistent policy",
                                        spec->tag.c_str());
            }
            group_it->second.layers.push_back(static_cast<int>(layer));
        }
    }

    std::vector<GroupInfo> groups;
    std::vector<LayerInfo> layers(static_cast<size_t>(config.layer_num));
    groups.reserve(ordered_tags.size());
    for (const auto& tag : ordered_tags) {
        const auto& state = group_by_tag.at(tag);
        GroupInfo   group;
        group.spec      = state.spec;
        group.policy    = state.policy;
        group.layer_ids = state.layers;
        const int gid   = static_cast<int>(groups.size());
        groups.push_back(group);
        for (int layer_id : state.layers) {
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(gid);
            layer.tag_to_gid[tag] = gid;
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
}

size_t kernelBlocksPerKvBlockForGroup(const CacheConfig& config, size_t group_id) {
    const bool is_full = config.typeForGroup(group_id) == CacheGroupType::FULL;
    return is_full ? config.kernelBlocksPerKvBlock() : 1;
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto group_num               = static_cast<size_t>(config.groupNums());
    std::vector<uint32_t> group_block_nums(group_num, 0);
    std::vector<size_t> group_kv_block_stride_bytes(group_num, 0);
    std::vector<size_t> group_kv_scale_stride_bytes(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = config.specForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        config.setGroupSeqSizePerBlock(gid, spec->seq_size_per_block);
        const auto   layer_count                = static_cast<uint32_t>(config.layerIdsForGroup(gid).size());
        const size_t kernel_kv_stride           = spec->block_size_bytes();
        const auto   kernel_scale               = spec->scale_block_size_bytes();
        const size_t group_bpk                  = kernelBlocksPerKvBlockForGroup(config, gid);
        const size_t kv_stride                  = kernel_kv_stride * group_bpk;
        const size_t scale_stride               = kernel_scale * group_bpk;
        group_kv_block_stride_bytes[gid]        = kv_stride;
        group_kv_scale_stride_bytes[gid]        = scale_stride;
        const auto type     = config.typeForGroup(gid);
        const bool is_state = spec->is_state_cache;
        if (!is_state && type == CacheGroupType::FULL) {
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

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            const KVCacheConfig&     kv_cache_config,
                                            bool                     is_mtp,
                                            int                      gen_num_per_cycle) {
    const auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto physical_tokens_per_block =
        kv_cache_config.seq_size_per_block > 0 ? static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                                 static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block =
        kv_cache_config.kernel_seq_size_per_block > 0 ? static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                                        physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "hybrid-pool seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "hybrid-pool kernel_seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block >= kernel_tokens_per_block
                                && physical_tokens_per_block % kernel_tokens_per_block == 0,
                            "hybrid-pool seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = physical_tokens_per_block;
    config.kernel_seq_size_per_block = kernel_tokens_per_block;
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    if (!model_config.kv_cache_spec_descs.empty()) {
        validateHybridPoolDescs(model_config, kernel_tokens_per_block, gen_num_per_cycle);
        SpecBuildContext ctx;
        ctx.dtype                   = dtype;
        ctx.seq_size_per_block      = physical_tokens_per_block;
        ctx.attn_tp_size            = static_cast<uint32_t>(parallelism_config.get_attn_tp_size());
        ctx.kernel_tokens_per_block = kernel_tokens_per_block;
        ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
        ctx.cp_size                 = fixedRegionCpSize(parallelism_config);
        ctx.cp_prefill_sliced       = isPrefillCpSliced(parallelism_config);
        auto refreshed_specs = CacheConfigCreator::buildLayerSpecsFromDescs(
            model_config.kv_cache_spec_descs, ctx, model_config.num_layers);
        populateGroupsFromLayerSpecs(config, model_config.kv_cache_spec_descs, refreshed_specs);
        for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
            const auto& spec = config.specForGroup(gid);
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool desc config produced null spec gid=%zu", gid);
            config.setGroupSeqSizePerBlock(gid, spec->seq_size_per_block);
            config.use_typed_cache_regions =
                config.use_typed_cache_regions || spec->type == KVCacheSpecType::OpaqueKV
                || spec->type == KVCacheSpecType::OpaqueState;
            config.use_opaque_kv_cache_store =
                config.use_opaque_kv_cache_store || spec->type == KVCacheSpecType::OpaqueKV
                || spec->type == KVCacheSpecType::OpaqueState;
        }
        for (const auto& layer_descs : model_config.kv_cache_spec_descs) {
            for (const auto& desc : layer_descs) {
                config.is_sparse = config.is_sparse || desc.cache_type == CacheType::COMPRESSED_KV;
            }
        }
        config.disable_decode_first_malloc_device_reuse =
            config.disable_decode_first_malloc_device_reuse || config.use_opaque_kv_cache_store;
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "HybridPoolConfigCreator requires kv_cache_spec_descs");
    }

    RTP_LLM_CHECK_WITH_INFO(config.groupNums() > 0, "hybrid-pool config produced no cache specs");
    setupIndependentPoolSizes(config, is_mtp);
    return config;
}

}  // namespace

CacheConfig HybridPoolConfigCreator::createConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    return createHybridAttentionPoolConfig(model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
}

}  // namespace rtp_llm
