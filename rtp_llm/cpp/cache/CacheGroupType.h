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

enum class CacheEvictPolicy : int8_t {
    CHAIN       = 0,
    INDEPENDENT = 1,
    NONE        = 2,
};

struct CacheGroupPolicy {
    CacheReusePolicy reuse_policy              = CacheReusePolicy::REUSABLE;
    CacheEvictPolicy evict_policy              = CacheEvictPolicy::CHAIN;
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

inline const char* cacheEvictPolicyName(CacheEvictPolicy evict_policy) {
    switch (evict_policy) {
        case CacheEvictPolicy::CHAIN:
            return "chain";
        case CacheEvictPolicy::INDEPENDENT:
            return "independent";
        case CacheEvictPolicy::NONE:
            return "none";
    }
    return "unknown";
}

inline CacheGroupPolicy defaultCacheGroupPolicy(CacheGroupType group_type) {
    CacheGroupPolicy policy;
    policy.active_tail_blocks = group_type == CacheGroupType::LINEAR ? 1 : (group_type == CacheGroupType::SWA ? 2 : 0);
    return policy;
}

}  // namespace rtp_llm
