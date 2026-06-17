#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SWA: sliding-window attention group (PD cache-store transfer keeps the last two blocks)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    SWA    = 2,
};

enum class CacheReusePolicy : int8_t {
    REUSABLE     = 0,
    NON_REUSABLE = 1,
};

struct CacheGroupPolicy {
    CacheReusePolicy reuse_policy              = CacheReusePolicy::REUSABLE;
    int              active_tail_blocks        = 2;
    bool             validate_tail_blocks      = true;
    uint32_t         explicit_block_num        = 0;
    bool             reserve_from_paged_budget = false;
};

inline const char* cacheGroupTypeName(CacheGroupType group_type) {
    switch (group_type) {
        case CacheGroupType::LINEAR:
            return "LINEAR";
        case CacheGroupType::FULL:
            return "FULL";
        case CacheGroupType::SWA:
            return "SWA";
    }
    return "UNKNOWN";
}

// Cache identity for models where one logical layer owns multiple cache entries.
// CacheGroupType describes allocation/reuse policy, while KVCacheRegionName
// describes which cache object a layer wants to access.
enum class KVCacheRegionName : int8_t {
    DEFAULT       = 0,
    CSA_KV        = 1,
    HCA_KV        = 2,
    INDEXER_KV    = 3,
    INDEXER_STATE = 4,
    CSA_STATE     = 5,
    HCA_STATE     = 6,
    SWA_KV        = 7,
    REGION_COUNT  = 8,
};

inline const char* cacheRegionName(KVCacheRegionName region_name) {
    switch (region_name) {
        case KVCacheRegionName::DEFAULT:
            return "DEFAULT";
        case KVCacheRegionName::CSA_KV:
            return "CSA_KV";
        case KVCacheRegionName::HCA_KV:
            return "HCA_KV";
        case KVCacheRegionName::INDEXER_KV:
            return "INDEXER_KV";
        case KVCacheRegionName::INDEXER_STATE:
            return "INDEXER_STATE";
        case KVCacheRegionName::CSA_STATE:
            return "CSA_STATE";
        case KVCacheRegionName::HCA_STATE:
            return "HCA_STATE";
        case KVCacheRegionName::SWA_KV:
            return "SWA_KV";
        case KVCacheRegionName::REGION_COUNT:
            return "REGION_COUNT";
    }
    return "UNKNOWN";
}

inline bool isStateRegion(KVCacheRegionName region_name) {
    return region_name == KVCacheRegionName::INDEXER_STATE || region_name == KVCacheRegionName::CSA_STATE
           || region_name == KVCacheRegionName::HCA_STATE;
}

inline bool isDsv4FixedRegion(KVCacheRegionName region_name) {
    return isStateRegion(region_name) || region_name == KVCacheRegionName::SWA_KV;
}

inline bool skipReuseCacheRegion(KVCacheRegionName region_name) {
    return region_name == KVCacheRegionName::HCA_STATE;
}

inline CacheGroupPolicy cacheGroupPolicyForLegacyRegion(CacheGroupType group_type, KVCacheRegionName region_name) {
    CacheGroupPolicy policy;
    policy.active_tail_blocks = group_type == CacheGroupType::SWA ? 2 : 0;
    if (region_name == KVCacheRegionName::HCA_STATE) {
        policy.reuse_policy         = CacheReusePolicy::NON_REUSABLE;
        policy.active_tail_blocks   = 1;
        policy.validate_tail_blocks = false;
    }
    return policy;
}

}  // namespace rtp_llm
