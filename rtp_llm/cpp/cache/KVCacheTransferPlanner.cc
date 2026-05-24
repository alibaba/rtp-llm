#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// ----------------------------------------------------------------------------
// M04-PR1: canonical block-major planner.
//
// Outer loop: B ∈ [0, total_cache_keys). Inner loop: per-pool include
// predicate that replicates the legacy per-group transfer policy (LINEAR
// last-only, SWA last-2, FULL all-non-reused). CP page-RR re-pairing is
// FULL-pool-only and emits ``offset_index = B / cp_size``; non-FULL pools
// are replicated and only owned by ``shard_owner_rank == 0``.
//
// Under bps[p] ≡ 1 the output is byte-equivalent to the legacy concatenation
// of per-group plans; the legacy ``blockPositionsForCacheTransfer`` and
// ``buildCacheStoreBlockPlan`` free functions are now 1-line wrappers around
// this planner.
// ----------------------------------------------------------------------------
std::vector<UnifiedTransferItem>
buildUnifiedTransferPlan(int                                model_id,
                         size_t                             total_logical_blocks,
                         size_t                             total_cache_keys,
                         size_t                             reuse_block_size,
                         const std::vector<PoolDescriptor>& pools,
                         const std::vector<size_t>&         pool_block_counts,
                         int                                shard_owner_rank,
                         int                                cp_size,
                         bool                               use_hybrid,
                         bool                               hybrid_full_from_begin,
                         PlannerRole                        role) {
    (void)model_id;             // reserved for future MTP namespacing checks (F06 12-6)
    (void)role;                 // arithmetic identical between prefill/decode roles
    (void)total_logical_blocks; // currently bounded by total_cache_keys (A08 16-2)

    RTP_LLM_CHECK(pool_block_counts.size() == pools.size());
    // Planner output blocks are indexed by both the unified logical id and the
    // post-cp-remap cache_keys array; emit only B values that have a
    // corresponding cache_keys entry. Hard CHECK catches caller miscount
    // (e.g. forgot the cp projection). A08 16-2.
    RTP_LLM_CHECK(total_cache_keys <= total_logical_blocks);

    // Non-hybrid callers MUST run pure FULL semantics; otherwise SWA/LINEAR
    // dispatch silently falls through to FULL with wrong block counts.
    // A08 16-4.
    if (!use_hybrid) {
        for (const auto& p : pools) {
            RTP_LLM_CHECK(p.group_type == CacheGroupType::FULL);
        }
    }

    std::vector<UnifiedTransferItem> plan;
    plan.reserve(total_cache_keys * pools.size());

    for (size_t B = 0; B < total_cache_keys; ++B) {
        for (const auto& pool : pools) {
            const size_t pool_blocks = pool_block_counts[static_cast<size_t>(pool.pool_id)];

            // Step 1: per-pool inclusion predicate = today's
            // blockPositionsForCacheTransfer per group.
            switch (pool.group_type) {
                case CacheGroupType::FULL: {
                    const size_t start = (use_hybrid && hybrid_full_from_begin) ? 0u : reuse_block_size;
                    if (B < start || B >= pool_blocks) {
                        continue;
                    }
                    break;
                }
                case CacheGroupType::SWA: {
                    // Risk 9.1: SWA tail is computed off PER-POOL block count.
                    const size_t swa_start = pool_blocks > 2u ? pool_blocks - 2u : 0u;
                    if (B < swa_start || B >= pool_blocks) {
                        continue;
                    }
                    break;
                }
                case CacheGroupType::LINEAR: {
                    if (pool_blocks == 0u || B != pool_blocks - 1u) {
                        continue;
                    }
                    break;
                }
            }

            // Step 2: per-pool drop-last-partial gate. REQ-D3: applies to
            // FULL pools only. Legacy SWA path unconditionally emits the
            // last 2 blocks regardless of last_partial; LINEAR emits exactly
            // one tail block and ignores last_partial. Dropping the partial
            // SWA tail here would produce a silent wire-semantic delta vs
            // legacy peers.
            if (pool.last_partial && pool.group_type == CacheGroupType::FULL && B + 1u == total_cache_keys) {
                continue;
            }

            // Step 3: CP page-RR sharding — FULL pools only. Identical
            // arithmetic to legacy buildCacheStoreBlockPlan. Non-FULL pools
            // are replicated and only owned by shard_owner_rank == 0.
            int        key_index    = static_cast<int>(B);  // == block_pos
            int        offset_index = static_cast<int>(B);
            const bool sharded      = (cp_size > 1) && (pool.group_type == CacheGroupType::FULL);
            if (sharded) {
                if (static_cast<int>(B) % cp_size != shard_owner_rank) {
                    continue;
                }
                offset_index = static_cast<int>(B) / cp_size;
            } else if (cp_size > 1 && shard_owner_rank != 0) {
                // Non-FULL pools are replicated to rank 0 only; F06 26-1.
                continue;
            }

            plan.push_back({static_cast<int64_t>(B), pool.pool_id, key_index, offset_index});
        }
    }
    return plan;
}

// ----------------------------------------------------------------------------
// Legacy compat shims: now 1-line wrappers over the unified planner.
// External callers (DecodeRpcServer, ExecOps, internal smoke harnesses)
// keep compiling unchanged. PR-4 will retire these once all consumers
// migrate to the unified planner.
// ----------------------------------------------------------------------------
std::vector<size_t> blockPositionsForCacheTransfer(size_t         block_num,
                                                   size_t         reuse_block_size,
                                                   bool           use_hybrid,
                                                   CacheGroupType group_type,
                                                   bool           hybrid_full_from_begin) {
    // A08 16-4: non-hybrid callers MUST be FULL.
    RTP_LLM_CHECK(use_hybrid || group_type == CacheGroupType::FULL);

    PoolDescriptor pool{
        /*pool_id=*/0,
        /*layer_id=*/-1,
        /*region_name=*/KVCacheRegionName::DEFAULT,
        /*group_type=*/use_hybrid ? group_type : CacheGroupType::FULL,
        /*stride_bytes=*/0,
        /*last_partial=*/false,
    };
    const auto plan = buildUnifiedTransferPlan(/*model_id=*/0,
                                               /*total_logical_blocks=*/block_num,
                                               /*total_cache_keys=*/block_num,
                                               reuse_block_size,
                                               {pool},
                                               /*pool_block_counts=*/{block_num},
                                               /*shard_owner_rank=*/0,
                                               /*cp_size=*/1,
                                               use_hybrid,
                                               hybrid_full_from_begin,
                                               PlannerRole::kPrefillWrite);
    std::vector<size_t> out;
    out.reserve(plan.size());
    for (const auto& it : plan) {
        out.push_back(static_cast<size_t>(it.block_pos));
    }
    return out;
}

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         reuse_block_size,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size) {
    PoolDescriptor pool{
        /*pool_id=*/0,
        /*layer_id=*/-1,
        /*region_name=*/KVCacheRegionName::DEFAULT,
        /*group_type=*/use_hybrid ? group_type : CacheGroupType::FULL,
        /*stride_bytes=*/0,
        /*last_partial=*/false,
    };
    const auto plan = buildUnifiedTransferPlan(/*model_id=*/0,
                                               total_logical_blocks,
                                               /*total_cache_keys=*/total_logical_blocks,
                                               reuse_block_size,
                                               {pool},
                                               /*pool_block_counts=*/{total_logical_blocks},
                                               /*shard_owner_rank=*/cp_rank,
                                               cp_size,
                                               use_hybrid,
                                               /*hybrid_full_from_begin=*/true,
                                               PlannerRole::kPrefillWrite);
    std::vector<CacheStoreBlockPair> out;
    out.reserve(plan.size());
    for (const auto& it : plan) {
        out.push_back({it.key_index, it.offset_index});
    }
    return out;
}

std::string layerRegionCacheTransferKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (region_name != KVCacheRegionName::DEFAULT) {
        key += "-" + std::to_string(static_cast<int>(region_name));
    }
    return key;
}

}  // namespace rtp_llm
