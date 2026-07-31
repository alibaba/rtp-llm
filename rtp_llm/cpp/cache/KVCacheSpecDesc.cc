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
        case KVCacheSpecType::CompressedKVCache:
            spec = CompressedKVCacheSpec::build(desc, ctx);
            break;
        case KVCacheSpecType::SWAState:
            spec = SWAStateCacheSpec::build(desc, ctx);
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    }
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "failed to build KVCacheSpec tag=%s", desc.tag.c_str());
    spec->cp_slice_ = desc.cp.has_value() && desc.cp->slice.value_or(false);
    return spec;
}

CacheGroupType SpecBuilder::groupType(const KVCacheSpecDesc& desc) {
    if (desc.group_type.has_value()) {
        return *desc.group_type;
    }
    switch (desc.cache_type) {
        case KVCacheSpecType::LinearAttention:
            return CacheGroupType::LINEAR;
        case KVCacheSpecType::SWAState:
            return CacheGroupType::SWA;
        case KVCacheSpecType::MultiHeadAttention:
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::CompressedKVCache:
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
    }
    return policy;
}

}  // namespace rtp_llm
