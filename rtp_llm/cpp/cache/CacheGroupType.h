#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
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
    CacheReusePolicy reuse_policy         = CacheReusePolicy::REUSABLE;
    CacheEvictPolicy evict_policy         = CacheEvictPolicy::CHAIN;
    bool             validate_tail_blocks = true;
    bool             prefix_reusable      = true;
    bool             is_reservable        = true;
    CacheGroupType   group_type           = CacheGroupType::FULL;
};

inline const char* cacheGroupTypeName(CacheGroupType group_type) {
    switch (group_type) {
        case CacheGroupType::LINEAR:
            return "LINEAR";
        case CacheGroupType::FULL:
            return "FULL";
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
    policy.group_type      = group_type;
    policy.prefix_reusable = group_type == CacheGroupType::FULL;
    return policy;
}

}  // namespace rtp_llm
