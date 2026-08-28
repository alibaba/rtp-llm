#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

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

enum class CpBlockMappingMode : int8_t {
    NONE              = 0,
    BLOCK_ROUND_ROBIN = 1,
    COMPACT_LAST_RANK = 2,
};

struct CacheGroupPolicy {
    CacheGroupType     group_type             = CacheGroupType::FULL;
    bool               enable_prefix_reuse    = true;
    CacheEvictPolicy   evict_policy           = CacheEvictPolicy::CHAIN;
    bool               reservable             = true;
    uint32_t           explicit_block_num     = 0;
    bool               charge_to_paged_budget = false;
    uint32_t           active_tail_blocks     = 0;
    bool               validate_tail_blocks   = true;
    CpBlockMappingMode cp_mapping             = CpBlockMappingMode::NONE;
};

// One cache-store registration step: pair a cache key from the full logical
// namespace with a slot in the tag-local physical block table. Under CP,
// both round-robin FULL groups and compact STATE/SWA groups use local slots.
struct CacheStoreBlockMapping {
    int cache_key_index;
    int block_table_index;
};

namespace detail {

inline size_t
resolveTransferStart(size_t block_count, size_t reuse_block_size, size_t active_tail_blocks, bool use_hybrid) {
    if (!use_hybrid) {
        return std::min(reuse_block_size, block_count);
    }
    if (active_tail_blocks == 0) {
        return 0;
    }
    return block_count > active_tail_blocks ? block_count - active_tail_blocks : 0;
}

inline std::vector<CacheStoreBlockMapping>
buildNormalPlan(const CacheGroupPolicy& policy, size_t total_logical_blocks, size_t reuse_block_size, bool use_hybrid) {
    const size_t start =
        resolveTransferStart(total_logical_blocks, reuse_block_size, policy.active_tail_blocks, use_hybrid);

    std::vector<CacheStoreBlockMapping> plan;
    plan.reserve(total_logical_blocks - start);
    for (size_t logical_idx = start; logical_idx < total_logical_blocks; ++logical_idx) {
        const int block_idx = static_cast<int>(logical_idx);
        plan.push_back({block_idx, block_idx});
    }
    return plan;
}

inline std::vector<CacheStoreBlockMapping> buildBlockRoundRobinPlan(const CacheGroupPolicy& policy,
                                                                    size_t                  total_logical_blocks,
                                                                    size_t                  reuse_block_size,
                                                                    bool                    use_hybrid,
                                                                    int                     cp_rank,
                                                                    int                     cp_size) {
    const size_t start =
        resolveTransferStart(total_logical_blocks, reuse_block_size, policy.active_tail_blocks, use_hybrid);
    const size_t cp_size_t = static_cast<size_t>(cp_size);
    const size_t cp_rank_t = static_cast<size_t>(cp_rank);

    std::vector<CacheStoreBlockMapping> plan;
    plan.reserve(total_logical_blocks - start);
    for (size_t logical_idx = start; logical_idx < total_logical_blocks; ++logical_idx) {
        if (logical_idx % cp_size_t != cp_rank_t) {
            continue;
        }
        plan.push_back({static_cast<int>(logical_idx), static_cast<int>(logical_idx / cp_size_t)});
    }
    return plan;
}

inline std::vector<CacheStoreBlockMapping> buildCompactLastRankPlan(const CacheGroupPolicy& policy,
                                                                    size_t                  total_logical_blocks,
                                                                    size_t                  reuse_block_size,
                                                                    bool                    use_hybrid,
                                                                    int                     cp_size) {
    const size_t cp_size_t        = static_cast<size_t>(cp_size);
    const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
    const size_t tail_blocks      = std::max<size_t>(1, policy.active_tail_blocks);
    const size_t start            = resolveTransferStart(canonical_blocks, reuse_block_size, tail_blocks, use_hybrid);

    std::vector<CacheStoreBlockMapping> plan;
    plan.reserve(canonical_blocks - start);
    for (size_t local_idx = start; local_idx < canonical_blocks; ++local_idx) {
        const size_t logical_idx = std::min((local_idx + 1) * cp_size_t - 1, total_logical_blocks - 1);
        plan.push_back({static_cast<int>(logical_idx), static_cast<int>(local_idx)});
    }
    return plan;
}

}  // namespace detail

// Keep cache-store projection header-only so bindings that consume ExecOps.cc
// as a source file do not need to link the full CPSlotMapper implementation.
inline std::vector<CacheStoreBlockMapping> buildCacheStorePlan(const CacheGroupPolicy& policy,
                                                               size_t                  total_logical_blocks,
                                                               size_t                  reuse_block_size,
                                                               bool                    use_hybrid,
                                                               int                     cp_rank,
                                                               int                     cp_size) {
    if (total_logical_blocks == 0) {
        return {};
    }

    if (cp_size <= 1) {
        return detail::buildNormalPlan(policy, total_logical_blocks, reuse_block_size, use_hybrid);
    }

    switch (policy.cp_mapping) {
        case CpBlockMappingMode::NONE:
            return detail::buildNormalPlan(policy, total_logical_blocks, reuse_block_size, use_hybrid);
        case CpBlockMappingMode::BLOCK_ROUND_ROBIN:
            return detail::buildBlockRoundRobinPlan(
                policy, total_logical_blocks, reuse_block_size, use_hybrid, cp_rank, cp_size);
        case CpBlockMappingMode::COMPACT_LAST_RANK:
            return detail::buildCompactLastRankPlan(
                policy, total_logical_blocks, reuse_block_size, use_hybrid, cp_size);
    }
    RTP_LLM_CHECK_WITH_INFO(false,
                            "unhandled CpBlockMappingMode=%d in buildCacheStorePlan",
                            static_cast<int>(policy.cp_mapping));
    return {};
}

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
    policy.group_type          = group_type;
    policy.enable_prefix_reuse = group_type == CacheGroupType::FULL || group_type == CacheGroupType::LINEAR;
    policy.active_tail_blocks  = group_type == CacheGroupType::LINEAR ? 1 : (group_type == CacheGroupType::SWA ? 2 : 0);
    if (group_type == CacheGroupType::FULL) {
        policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    } else if (group_type == CacheGroupType::SWA) {
        policy.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    }
    return policy;
}

}  // namespace rtp_llm
