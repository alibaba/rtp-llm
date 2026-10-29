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
    auto& layout = spec->tp_transfer;
    layout.mode  = desc.tp_transfer_mode;
    if (layout.mode == TpCacheTransferMode::LINEAR_QKV) {
        RTP_LLM_CHECK_WITH_INFO(desc.cache_type == KVCacheSpecType::LinearAttention,
                                "LINEAR_QKV transfer requires a linear cache spec");
        const auto& linear    = *ctx.linear_attention_config;
        const auto  tp        = std::max<int64_t>(1, ctx.parallelism_config->get_attn_tp_size());
        const auto  item_size = getTypeSize(linear.conv_state_dtype);
        layout.key_heads      = linear.linear_num_key_heads / tp;
        layout.value_heads    = linear.linear_num_value_heads / tp;
        layout.q_row_bytes    = layout.key_heads * linear.linear_key_head_dim * item_size;
        layout.k_row_bytes    = layout.q_row_bytes;
        layout.v_row_bytes    = layout.value_heads * linear.linear_value_head_dim * item_size;
        layout.history_rows   = linear.linear_conv_kernel_dim - 1;
    } else if (layout.mode == TpCacheTransferMode::MHA_HEADS) {
        RTP_LLM_CHECK_WITH_INFO(desc.cache_type == KVCacheSpecType::MultiHeadAttention,
                                "MHA_HEADS transfer requires an MHA cache spec");
    } else {
        RTP_LLM_CHECK_WITH_INFO(layout.mode == TpCacheTransferMode::LEGACY
                                    || (layout.mode == TpCacheTransferMode::REPLICATED
                                        && (desc.cache_type == KVCacheSpecType::OpaqueKV
                                            || desc.cache_type == KVCacheSpecType::OpaqueState)),
                                "invalid cache TP transfer mode for tag=%s",
                                desc.tag.c_str());
    }
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
        policy.enable_prefix_reuse = true;
    }
    if (desc.reuse.has_value()) {
        if (desc.reuse->enable_prefix_reuse.has_value()) {
            policy.enable_prefix_reuse = *desc.reuse->enable_prefix_reuse;
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
    policy.shared_chunk_state =
        policy.group_type == CacheGroupType::SWA && desc.cache_type == KVCacheSpecType::OpaqueState
        && desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING && desc.state_ring_overlap == 0
        && !policy.enable_prefix_reuse && policy.active_tail_blocks == 1;
    return policy;
}

}  // namespace rtp_llm
