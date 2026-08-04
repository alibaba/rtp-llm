#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

#include <limits>

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

CacheGroupType defaultGroupType(KVCacheSpecType cache_type) {
    switch (cache_type) {
        case KVCacheSpecType::LinearAttention:
            return CacheGroupType::LINEAR;
        case KVCacheSpecType::OpaqueState:
            return CacheGroupType::SWA;
        case KVCacheSpecType::MultiHeadAttention:
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::OpaqueKV:
            return CacheGroupType::FULL;
    }
    RTP_LLM_CHECK_WITH_INFO(false, "unknown KVCacheSpecType=%d", static_cast<int>(cache_type));
    return CacheGroupType::FULL;
}

CpBlockMappingMode cpMapping(const KVCacheSpecDesc& desc) {
    const auto group_type = desc.group_type.value_or(defaultGroupType(desc.cache_type));
    auto       mapping    = defaultCacheGroupPolicy(group_type).cp_mapping;
    if (desc.cp.has_value() && desc.cp->mapping.has_value()) {
        mapping = *desc.cp->mapping;
    }
    return mapping;
}

}  // namespace

KVCacheSpecBuildResult SpecBuilder::build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    auto policy = buildPolicy(desc);
    auto spec   = buildSpec(desc, ctx);
    return {std::move(spec), std::move(policy)};
}

KVCacheSpecPtr SpecBuilder::buildSpec(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    RTP_LLM_CHECK_WITH_INFO(!desc.tag.empty(), "KVCacheSpecDesc tag must not be empty");
    const auto seq_size_per_block        = seqSizePerBlock(desc, ctx);
    const auto kernel_seq_size_per_block = kernelSeqSizePerBlock(desc, seq_size_per_block);

    switch (desc.cache_type) {
        case KVCacheSpecType::MultiHeadAttention:
            return MHAKVCacheSpec::build(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        case KVCacheSpecType::MultiHeadLatentAttention:
            return MLAKVCacheSpec::build(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        case KVCacheSpecType::LinearAttention:
            return LinearKVCacheSpec::build(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        case KVCacheSpecType::OpaqueKV:
            return CompressedKVCacheSpec::build(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        case KVCacheSpecType::OpaqueState:
            return FixedStateCacheSpec::build(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
    }
    RTP_LLM_CHECK_WITH_INFO(false, "unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    return nullptr;
}

CacheGroupType SpecBuilder::groupType(const KVCacheSpecDesc& desc) {
    if (desc.group_type.has_value()) {
        return *desc.group_type;
    }
    return defaultGroupType(desc.cache_type);
}

CacheGroupPolicy SpecBuilder::buildPolicy(const KVCacheSpecDesc& desc) {
    CacheGroupPolicy policy = defaultCacheGroupPolicy(groupType(desc));
    policy.cp_mapping       = cpMapping(desc);
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

    RTP_LLM_CHECK_WITH_INFO(policy.group_type == CacheGroupType::LINEAR || policy.group_type == CacheGroupType::FULL
                                || policy.group_type == CacheGroupType::SWA,
                            "KVCacheSpecDesc tag=%s has invalid group type=%d",
                            desc.tag.c_str(),
                            static_cast<int>(policy.group_type));
    RTP_LLM_CHECK_WITH_INFO(policy.cp_mapping == CpBlockMappingMode::NONE
                                || policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN
                                || policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK,
                            "KVCacheSpecDesc tag=%s has invalid CP mapping=%d",
                            desc.tag.c_str(),
                            static_cast<int>(policy.cp_mapping));
    RTP_LLM_CHECK_WITH_INFO(policy.cp_slice == CpBlockSliceMode::NONE
                                || policy.cp_slice == CpBlockSliceMode::EQUAL_BYTES
                                || policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES,
                            "KVCacheSpecDesc tag=%s has invalid CP slice=%d",
                            desc.tag.c_str(),
                            static_cast<int>(policy.cp_slice));
    RTP_LLM_CHECK_WITH_INFO(policy.cp_mapping != CpBlockMappingMode::COMPACT_LAST_RANK
                                || policy.group_type != CacheGroupType::FULL,
                            "FULL KVCacheSpecDesc tag=%s cannot use COMPACT_LAST_RANK mapping",
                            desc.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(policy.group_type != CacheGroupType::FULL || policy.cp_slice == CpBlockSliceMode::NONE,
                            "FULL KVCacheSpecDesc tag=%s cannot use CP byte slicing",
                            desc.tag.c_str());
    const auto prefill_slice_layout = desc.cp.has_value() ?
                                          desc.cp->prefill_slice_layout.value_or(CpPrefillSliceLayout::NONE) :
                                          CpPrefillSliceLayout::NONE;
    RTP_LLM_CHECK_WITH_INFO(prefill_slice_layout == CpPrefillSliceLayout::NONE
                                || prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD
                                || prefill_slice_layout == CpPrefillSliceLayout::BLOCK_STRIDE,
                            "KVCacheSpecDesc tag=%s has invalid CP prefill slice layout=%d",
                            desc.tag.c_str(),
                            static_cast<int>(prefill_slice_layout));
    RTP_LLM_CHECK_WITH_INFO(policy.group_type != CacheGroupType::FULL
                                || prefill_slice_layout == CpPrefillSliceLayout::NONE,
                            "FULL KVCacheSpecDesc tag=%s cannot use CP prefill byte slicing",
                            desc.tag.c_str());
    return policy;
}

uint32_t SpecBuilder::seqSizePerBlock(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    RTP_LLM_CHECK_WITH_INFO(ctx.seq_size_per_block > 0,
                            "KVCacheSpecDesc tag=%s requires positive SpecBuildContext.seq_size_per_block",
                            desc.tag.c_str());

    const auto mapping = cpMapping(desc);
    if (mapping != CpBlockMappingMode::COMPACT_LAST_RANK) {
        return ctx.seq_size_per_block;
    }

    RTP_LLM_CHECK_WITH_INFO(groupType(desc) != CacheGroupType::FULL,
                            "FULL KVCacheSpecDesc tag=%s cannot use COMPACT_LAST_RANK mapping",
                            desc.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr,
                            "COMPACT_LAST_RANK KVCacheSpecDesc tag=%s requires "
                            "SpecBuildContext.parallelism_config",
                            desc.tag.c_str());
    const auto& parallelism = *ctx.parallelism_config;
    uint32_t    cp_size     = 1;
    if (parallelism.prefill_cp_config.kv_cache_sharded) {
        if (parallelism.role_type == RoleType::PREFILL && parallelism.tp_size > 1) {
            RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(parallelism.tp_size) <= std::numeric_limits<uint32_t>::max(),
                                    "KVCacheSpecDesc tag=%s CP size overflows uint32: %ld",
                                    desc.tag.c_str(),
                                    parallelism.tp_size);
            cp_size = static_cast<uint32_t>(parallelism.tp_size);
        } else if (parallelism.role_type == RoleType::DECODE && parallelism.prefill_cp_config.is_prefill_enabled()) {
            RTP_LLM_CHECK_WITH_INFO(
                parallelism.prefill_cp_config.prefill_cp_size > 1,
                "compact CP decode tag=%s requires explicit prefill_cp_size when PREFILL_CP is enabled",
                desc.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(parallelism.prefill_cp_config.prefill_cp_size)
                                        <= std::numeric_limits<uint32_t>::max(),
                                    "KVCacheSpecDesc tag=%s prefill CP size overflows uint32: %ld",
                                    desc.tag.c_str(),
                                    parallelism.prefill_cp_config.prefill_cp_size);
            cp_size = static_cast<uint32_t>(parallelism.prefill_cp_config.prefill_cp_size);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(ctx.seq_size_per_block <= std::numeric_limits<uint32_t>::max() / cp_size,
                            "KVCacheSpecDesc tag=%s block span overflow: B=%u C_g=%u",
                            desc.tag.c_str(),
                            ctx.seq_size_per_block,
                            cp_size);
    return ctx.seq_size_per_block * cp_size;
}

uint32_t SpecBuilder::kernelSeqSizePerBlock(const KVCacheSpecDesc& desc, uint32_t seq_size_per_block) {
    if (groupType(desc) != CacheGroupType::FULL) {
        RTP_LLM_CHECK_WITH_INFO(!desc.kernel_seq_size_per_block.has_value(),
                                "non-FULL KVCacheSpecDesc tag=%s must not define kernel_seq_size_per_block",
                                desc.tag.c_str());
        return seq_size_per_block;
    }

    const auto kernel_seq_size = desc.kernel_seq_size_per_block.value_or(seq_size_per_block);
    RTP_LLM_CHECK_WITH_INFO(kernel_seq_size > 0 && seq_size_per_block >= kernel_seq_size
                                && seq_size_per_block % kernel_seq_size == 0,
                            "FULL KVCacheSpecDesc tag=%s requires S_g=%u to be divisible by K_g=%u",
                            desc.tag.c_str(),
                            seq_size_per_block,
                            kernel_seq_size);
    return kernel_seq_size;
}

}  // namespace rtp_llm
