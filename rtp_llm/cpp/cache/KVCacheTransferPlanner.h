#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);

std::string layerRegionCacheTransferKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name);

// One iteration step of cache_store registration: pair the cache_key at
// ``key_index`` (FULL-length namespace) with the kv_cache_offset slot at
// ``offset_index`` (rank-local namespace). Outside CP-page-RR sharding the
// two are equal; under sharding they diverge for FULL groups (see below).
struct CacheStoreBlockPair {
    int key_index;
    int offset_index;
};

// Build the per-prefill-write iteration plan for cache_store registration.
//
// Background: ``cache_keys`` is always the FULL logical-block hash sequence
// (length = total_logical_blocks). ``kv_cache_offset`` is per-group and
// per-rank: for non-FULL groups every rank holds the full block list (length
// = total_logical_blocks), for FULL groups under CP-page-RR sharding each
// rank holds only the 1/cp_size logical blocks it owns, **compactly**, in
// the order they appear within the rank — i.e. local index ``i`` ↔ logical
// position ``cp_rank + i*cp_size``.
//
// To register the right key with the right buffer the planner emits:
//   * (pos, pos)                              — non-CP / non-FULL groups
//   * (cp_rank + i*cp_size, i) for owned i    — CP-sharded FULL groups
//
// Without this re-pairing the prefill side advertises ``cache_keys[i]``
// (== key for logical position i) attached to data from logical position
// ``cp_rank + i*cp_size`` — decode then receives content shifted by
// ``cp_rank`` slots and produces coherent-but-wrong output (DSV4 PD reuse
// regression seen 2026-05-12).
std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         reuse_block_size,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size);

// ============================================================================
// M04-PR1: unified block-major planner.
//
// The unified planner replaces the group-major outer loop with a block-major
// outer loop. The per-group transfer policy (LINEAR last-only, SWA last-2,
// FULL all-non-reused) is reinterpreted as a per-pool include predicate so a
// single planner walk emits a flat list of
// ``(block_pos, pool_id, key_index, offset_index)`` items.
//
// Under ``bps[p] ≡ 1`` (F02) the unified output is byte-equivalent to the
// legacy concatenation of per-group plans. Both legacy free functions above
// (`blockPositionsForCacheTransfer` / `buildCacheStoreBlockPlan`) become
// 1-line wrappers that synthesize a single-pool ``PoolDescriptor`` and
// forward to this planner.
// ============================================================================

struct PoolDescriptor {
    int               pool_id;          // contiguous 0..P-1
    int               layer_id;         // owning layer; -1 if cross-layer
    KVCacheRegionName region_name;      // for makeCacheKey
    CacheGroupType    group_type;       // policy carrier
    size_t            stride_bytes;     // per-pool kv_block_stride; 0 = legacy single-stride
    bool              last_partial;     // per-pool drop-last gate; FULL pools only (REQ-D3)
};

// Disambiguates the ``shard_owner_rank`` parameter:
//   * kPrefillWrite — shard owner == local rank emitting the plan
//   * kDecodeRead   — shard owner == peer being read from
// The arithmetic is identical either way; the enum keeps the cross-doc
// reading honest (F06 26-2).
enum class PlannerRole {
    kPrefillWrite,
    kDecodeRead,
};

struct UnifiedTransferItem {
    int64_t block_pos;     // unified logical block id, [0, total_logical_blocks)
    int     pool_id;       // index into pools[]
    int     key_index;     // index into cache_keys[] (== block_pos under unified namespace)
    int     offset_index;  // index into per-pool kv_cache_offset[]
};

// Build the unified transfer plan. See struct comments above for invariants.
//
// Inputs:
//   * ``total_logical_blocks``  — N, length of the unified super-block id list.
//   * ``total_cache_keys``      — M, upper bound on ``B``. Must be <= N. The
//                                  planner only emits B values < M. Caller is
//                                  responsible for the cp-projection step
//                                  (so M = source.localCacheKeys(cp_size-1,
//                                  cp_size).size() under CP+FULL).
//   * ``pools``                 — per-pool descriptors (length P).
//   * ``pool_block_counts``     — per-pool block counts (length P). Under
//                                  ``bps[p] ≡ 1`` typically uniform == N.
//   * ``shard_owner_rank``      — semantics depend on ``role``.
//   * ``cp_size``               — CP world size; 1 disables sharding.
//   * ``use_hybrid``            — when false, every pool MUST be FULL.
//   * ``hybrid_full_from_begin``— FULL group start position (true => 0,
//                                  false => reuse_block_size).
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
                         PlannerRole                        role = PlannerRole::kPrefillWrite);

// Formal "is this (pool, B) owned by ``peer_idx``" predicate. Derived from
// the same arithmetic as Step 3 inside the planner; exposed so the decode
// caller (DecodeRpcServer) can collapse the two legacy
// ``shouldLoad{Group,Block}FromPeer`` lambdas into a single call. F06 26-1.
inline bool ownedByPeer(CacheGroupType gtype, int64_t B, int peer_idx, int cp_size) {
    if (cp_size <= 1) {
        return peer_idx == 0;
    }
    if (gtype != CacheGroupType::FULL) {
        return peer_idx == 0;
    }
    return (B % cp_size) == peer_idx;
}

}  // namespace rtp_llm
