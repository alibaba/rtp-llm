#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

KVCacheSpecPtr SpecBuilder::build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
    RTP_LLM_CHECK_WITH_INFO(!desc.tag.empty(), "KVCacheSpecDesc tag must not be empty");
    KVCacheSpecPtr spec;
    switch (desc.cache_type) {
        case KVCacheSpecType::MultiHeadAttention:
            spec = MHAKVCacheSpec::build(desc, ctx);
            break;
        case KVCacheSpecType::MultiHeadLatentAttention:
            spec = MLAKVCacheSpec::build(desc, ctx);
            break;
        case KVCacheSpecType::LinearAttention:
            spec = LinearKVCacheSpec::build(desc, ctx);
            break;
        case KVCacheSpecType::OpaqueKV:
            spec = CompressedKVCacheSpec::build(desc, ctx);
            break;
        case KVCacheSpecType::OpaqueState:
            spec = FixedStateCacheSpec::build(desc, ctx);
            break;
    }
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    const auto mode = desc.tp_transfer_mode;
    RTP_LLM_CHECK_WITH_INFO(
        mode == TpCacheTransferMode::LEGACY
            || (mode == TpCacheTransferMode::LINEAR_QKV && desc.cache_type == KVCacheSpecType::LinearAttention)
            || (mode == TpCacheTransferMode::MHA_HEADS && desc.cache_type == KVCacheSpecType::MultiHeadAttention)
            || (mode == TpCacheTransferMode::REPLICATED
                && (desc.cache_type == KVCacheSpecType::OpaqueKV || desc.cache_type == KVCacheSpecType::OpaqueState)),
        "invalid cache TP transfer mode: tag=%s cache_type=%d mode=%d",
        desc.tag.c_str(),
        static_cast<int>(desc.cache_type),
        static_cast<int>(mode));
    spec->tp_transfer.mode = mode;
    return spec;
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

uint32_t SpecBuilder::kernelSeqSizePerBlock(const KVCacheSpecDesc&  desc,
                                            const SpecBuildContext& ctx,
                                            uint32_t                physical_seq_size_per_block) {
    const auto requested = ctx.kernel_tokens_per_block == 0 ? physical_seq_size_per_block : ctx.kernel_tokens_per_block;
    if (groupType(desc) != CacheGroupType::FULL) {
        return physical_seq_size_per_block;
    }
    return std::min(requested, physical_seq_size_per_block);
}

CacheGroupPolicy SpecBuilder::groupPolicy(const KVCacheSpecDesc& desc) {
    CacheGroupPolicy policy = defaultCacheGroupPolicy(groupType(desc));
    if (desc.is_state_cache) {
        policy.reuse_mode = CacheReuseMode::STANDARD;
    }
    if (desc.reuse.has_value()) {
        const auto& reuse = *desc.reuse;
        RTP_LLM_CHECK_WITH_INFO(!(reuse.enable_prefix_reuse.value_or(false) && reuse.local_only_state.value_or(false)),
                                "tag=%s has conflicting prefix and precise-only reuse",
                                desc.tag.c_str());
        if (reuse.enable_prefix_reuse.has_value()) {
            policy.reuse_mode = *reuse.enable_prefix_reuse ? CacheReuseMode::STANDARD : CacheReuseMode::DISABLED;
        }
        if (reuse.local_only_state.value_or(false)) {
            policy.reuse_mode = CacheReuseMode::PRECISE_ONLY;
        }
        if (reuse.mode.has_value()) {
            RTP_LLM_CHECK_WITH_INFO(
                (!reuse.enable_prefix_reuse.has_value()
                 || *reuse.enable_prefix_reuse == (*reuse.mode == CacheReuseMode::STANDARD))
                    && (!reuse.local_only_state.has_value()
                        || *reuse.local_only_state == (*reuse.mode == CacheReuseMode::PRECISE_ONLY)),
                "tag=%s has conflicting legacy and explicit reuse modes",
                desc.tag.c_str());
            policy.reuse_mode = *reuse.mode;
        }
    }
    if (desc.capacity.has_value()) {
        if (desc.capacity->reservable.has_value()) {
            policy.reservable = *desc.capacity->reservable;
        }
        if (desc.capacity->explicit_block_num.has_value()) {
            policy.explicit_block_num = *desc.capacity->explicit_block_num;
        }
        if (desc.capacity->charge_to_paged_budget.has_value()) {
            policy.charge_to_paged_budget = *desc.capacity->charge_to_paged_budget;
        }
    }
    if (desc.memory.has_value() && desc.memory->placement.has_value()) {
        policy.memory_placement = *desc.memory->placement;
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
    policy.shared_chunk_state = policy.group_type == CacheGroupType::SWA
                                && desc.cache_type == KVCacheSpecType::OpaqueState
                                && desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING
                                && desc.state_ring_overlap == 0 && policy.active_tail_blocks == 1;
    return policy;
}

}  // namespace rtp_llm
