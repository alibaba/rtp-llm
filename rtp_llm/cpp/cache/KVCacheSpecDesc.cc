#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

KVCacheSpecPtr SpecBuilder::build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
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
    return policy;
}

}  // namespace rtp_llm
