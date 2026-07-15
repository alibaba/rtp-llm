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
    }
    RTP_LLM_CHECK_WITH_INFO(false, "unknown KVCacheSpecType=%d", static_cast<int>(desc.cache_type));
    return nullptr;
}

CacheGroupType SpecBuilder::groupType(const KVCacheSpecDesc& desc) {
    return desc.cache_type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

CacheGroupPolicy SpecBuilder::groupPolicy(const KVCacheSpecDesc& desc) {
    return defaultCacheGroupPolicy(groupType(desc));
}

}  // namespace rtp_llm
