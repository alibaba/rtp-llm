#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

#include <limits>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

uint32_t effectiveCacheCpSize(const SpecBuildContext& ctx) {
    if (ctx.parallelism_config == nullptr || !ctx.parallelism_config->prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    const auto& parallelism_config = *ctx.parallelism_config;
    if (parallelism_config.role_type == RoleType::PREFILL && parallelism_config.tp_size > 1) {
        return static_cast<uint32_t>(parallelism_config.tp_size);
    }
    if (parallelism_config.role_type == RoleType::DECODE && parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        RTP_LLM_CHECK_WITH_INFO(
            parallelism_config.prefill_cp_config.prefill_cp_size > 1,
            "compact CP decode requires explicit prefill_cp_size when PREFILL_CP and kv_cache_sharded are enabled");
        return static_cast<uint32_t>(parallelism_config.prefill_cp_config.prefill_cp_size);
    }
    return 1;
}

namespace {

uint32_t physicalBlockSpan(const CacheGroupPolicy& policy, const SpecBuildContext& ctx) {
    return policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK ? effectiveCacheCpSize(ctx) : 1;
}

}  // namespace

BuiltLayerSpec SpecBuilder::build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    const auto policy                  = groupPolicy(desc);
    auto       finalized_ctx           = ctx;
    const auto base_seq_size_per_block = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
    const auto span                    = physicalBlockSpan(policy, ctx);
    RTP_LLM_CHECK_WITH_INFO(base_seq_size_per_block <= std::numeric_limits<uint32_t>::max() / span,
                            "KVCacheSpecDesc tag=%s physical seq size overflow: base=%u span=%u",
                            desc.tag.c_str(),
                            base_seq_size_per_block,
                            span);
    finalized_ctx.seq_size_per_block = base_seq_size_per_block * span;
    finalized_ctx.kernel_seq_size_per_block =
        ctx.kernel_seq_size_per_block == 0 ? base_seq_size_per_block : ctx.kernel_seq_size_per_block;
    auto spec = buildSpec(desc, finalized_ctx);
    return {desc.tag, std::move(spec), policy};
}

KVCacheSpecPtr SpecBuilder::buildSpec(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    RTP_LLM_CHECK_WITH_INFO(!desc.tag.empty(), "KVCacheSpecDesc tag must not be empty");
    switch (desc.cache_type) {
        case KVCacheSpecType::MultiHeadAttention:
            return MHAKVCacheSpec::build(desc, ctx);
        case KVCacheSpecType::MultiHeadLatentAttention:
            return MLAKVCacheSpec::build(desc, ctx);
        case KVCacheSpecType::LinearAttention:
            return LinearKVCacheSpec::build(desc, ctx);
        case KVCacheSpecType::OpaqueKV:
            return CompressedKVCacheSpec::build(desc, ctx);
        case KVCacheSpecType::OpaqueState:
            return FixedStateCacheSpec::build(desc, ctx);
    }
    RTP_LLM_CHECK_WITH_INFO(false, "unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    return nullptr;
}

CacheGroupType SpecBuilder::groupType(const KVCacheSpecDesc& desc) {
    if (desc.group_type.has_value()) {
        return *desc.group_type;
    }
    switch (desc.cache_type) {
        case KVCacheSpecType::LinearAttention:
            return CacheGroupType::LINEAR;
        case KVCacheSpecType::OpaqueState:
            return CacheGroupType::SWA;
        case KVCacheSpecType::MultiHeadAttention:
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::OpaqueKV:
            return CacheGroupType::FULL;
    }
    return CacheGroupType::FULL;
}

CacheGroupPolicy SpecBuilder::groupPolicy(const KVCacheSpecDesc& desc) {
    CacheGroupPolicy policy = defaultCacheGroupPolicy(groupType(desc));
    if (desc.is_state_cache) {
        policy.enable_prefix_reuse = true;
        policy.evict_policy        = CacheEvictPolicy::INDEPENDENT;
    }
    if (desc.reuse.has_value()) {
        if (desc.reuse->enable_prefix_reuse.has_value()) {
            policy.enable_prefix_reuse = *desc.reuse->enable_prefix_reuse;
        }
        if (desc.reuse->evict_policy.has_value()) {
            policy.evict_policy = *desc.reuse->evict_policy;
        }
    }
    if (desc.capacity.has_value()) {
        if (desc.capacity->reservable.has_value()) {
            policy.reservable = *desc.capacity->reservable;
        }
        if (desc.capacity->explicit_block_num.has_value()) {
            policy.explicit_block_num = *desc.capacity->explicit_block_num;
        }
    }
    if (desc.tail.has_value()) {
        if (desc.tail->active_tail_blocks.has_value()) {
            policy.active_tail_blocks = *desc.tail->active_tail_blocks;
        }
        if (desc.tail->validate_tail_blocks.has_value()) {
            policy.validate_tail_blocks = *desc.tail->validate_tail_blocks;
        }
    }
    if (desc.cp.has_value()) {
        if (desc.cp->mapping.has_value()) {
            policy.cp_mapping = *desc.cp->mapping;
        }
        if (desc.cp->slice.has_value()) {
            policy.cp_slice = *desc.cp->slice;
        }
    }
    return policy;
}

}  // namespace rtp_llm
