#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

void validateHybridPoolDescs(const ModelConfig& model_config, uint32_t kernel_tokens_per_block, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
        "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
        model_config.kv_cache_spec_descs.size(),
        model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "hybrid-pool desc config requires non-negative gen_num_per_cycle, got %d",
                            gen_num_per_cycle);

    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& layer_descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!layer_descs.empty(), "hybrid-pool desc config layer %ld has no descs", layer_id);
        for (const auto& desc : layer_descs) {
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

uint32_t localKvHeadNumForDesc(const KVCacheSpecDesc&   desc,
                               const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config) {
    switch (desc.cache_type) {
        case KVCacheSpecType::MultiHeadAttention:
            return mhaLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::LinearAttention:
            return linearLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::OpaqueKV:
        case KVCacheSpecType::OpaqueState:
            return 1;
        default:
            RTP_LLM_FAIL("unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    }
    return 1;
}

void populateGroupsFromLayerSpecs(CacheConfig&                 config,
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
            const auto local_kv_head_num = localKvHeadNumForDesc(desc, model_config, parallelism_config);
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
            auto& layer = layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(tag);
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

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t group_index = 0; group_index < group_num; ++group_index) {
        const auto& group = config.topology().groups().at(group_index);
        const auto& spec  = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec for tag=%s is null", group.tag.c_str());
        const auto   layer_count                 = static_cast<uint32_t>(group.layer_ids.size());
        const size_t kernel_kv_stride            = spec->block_size_bytes();
        const auto   kernel_scale                = spec->scale_block_size_bytes();
        const size_t group_bpk                   = config.kernelBlocksPerKvBlock(group.tag);
        const size_t kv_stride                   = kernel_kv_stride * group_bpk;
        const size_t scale_stride                = kernel_scale * group_bpk;
        group_kv_block_stride_bytes[group_index] = kv_stride;
        group_kv_scale_stride_bytes[group_index] = scale_stride;
        const auto type                          = group.policy.group_type;
        const bool is_paged_group                = type == CacheGroupType::FULL || type == CacheGroupType::LINEAR;
        if (is_paged_group && !config.usesExplicitIndependentBlocks(group.tag)) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);

        for (int layer_id : group.layer_ids) {
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
        validateHybridPoolDescs(model_config, kernel_tokens_per_block, gen_num_per_cycle);
        SpecBuildContext ctx;
        ctx.dtype                   = dtype;
        ctx.seq_size_per_block      = physical_tokens_per_block;
        ctx.attn_config             = &model_config.attn_config;
        ctx.linear_attention_config = &model_config.linear_attention_config;
        ctx.parallelism_config      = &parallelism_config;
        ctx.kernel_tokens_per_block = kernel_tokens_per_block;
        ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
        auto refreshed_specs        = CacheConfigCreator::buildLayerSpecsFromDescs(
            model_config.kv_cache_spec_descs, ctx, model_config.num_layers);
        populateGroupsFromLayerSpecs(
            config, model_config.kv_cache_spec_descs, refreshed_specs, model_config, parallelism_config);
        for (const auto& group : config.topology().groups()) {
            const auto& spec               = group.spec;
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
    return createHybridAttentionPoolConfig(
        model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
}

}  // namespace rtp_llm
