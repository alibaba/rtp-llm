#pragma once

#include <cstddef>
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

inline const char* metricCacheGroupTypeName(CacheGroupType group_type) {
    if (group_type == CacheGroupType::LINEAR) {
        return "linear";
    }
    if (group_type == CacheGroupType::FULL) {
        return "full";
    }
    if (group_type == CacheGroupType::SWA) {
        return "swa";
    }
    return "unknown";
}

enum class CacheMemoryPlacement : int8_t {
    DEVICE      = 0,
    HOST        = 1,
    HOST_PINNED = 2,
};

enum class CpBlockMappingMode : int8_t {
    NONE              = 0,
    BLOCK_ROUND_ROBIN = 1,
    COMPACT_LAST_RANK = 2,
};

enum class CpBlockSliceMode : int8_t {
    NONE          = 0,
    EQUAL_BYTES   = 1,
    PAYLOAD_BYTES = 2,
};

struct CacheGroupPolicy {
    CacheGroupType       group_type             = CacheGroupType::FULL;
    bool                 enable_prefix_reuse    = true;
    bool                 reservable             = true;
    uint32_t             explicit_block_num     = 0;
    bool                 charge_to_paged_budget = false;
    CacheMemoryPlacement memory_placement       = CacheMemoryPlacement::DEVICE;
    uint32_t             active_tail_blocks     = 0;
    bool                 validate_tail_blocks   = true;
    CpBlockMappingMode   cp_mapping             = CpBlockMappingMode::NONE;
    CpBlockSliceMode     cp_slice               = CpBlockSliceMode::NONE;
    // Model-derived SWA attention window in tokens. Zero means no bounded
    // window is configured and remains the default for non-SWA groups.
    int sliding_window_size = 0;
    // Share one state bank across logical blocks during chunked prefill.
    bool shared_chunk_state = false;
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

inline CacheGroupPolicy defaultCacheGroupPolicy(CacheGroupType group_type) {
    CacheGroupPolicy policy;
    policy.group_type          = group_type;
    policy.enable_prefix_reuse = group_type == CacheGroupType::FULL || group_type == CacheGroupType::LINEAR;
    policy.active_tail_blocks  = group_type == CacheGroupType::LINEAR ? 1 : (group_type == CacheGroupType::SWA ? 2 : 0);
    // CP mapping and slicing are supplied as resolved policy values by the caller.
    return policy;
}

// A group belongs to the KV cache event publication completeness set only when
// it participates in prefix reuse AND materializes every block position.
// Tail-sparse groups (active_tail_blocks > 0, i.e. LINEAR/SWA keep only their
// tail blocks) never hold full block chains, so counting them as required
// would suppress publication for almost every key.
inline bool cacheGroupPublishesPrefixChain(const CacheGroupPolicy& policy) {
    return policy.enable_prefix_reuse && policy.active_tail_blocks == 0;
}

}  // namespace rtp_llm
