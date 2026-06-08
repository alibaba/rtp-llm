#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);

size_t cacheVisibleBlockNumForCacheLoad(size_t         allocated_block_num,
                                        size_t         cache_key_count,
                                        bool           use_hybrid,
                                        CacheGroupType group_type,
                                        int            cp_size,
                                        bool           compact_blocks_by_cp);

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
//   * ((i+1)*cp_size-1, i)                    — CP-compact SWA/fixed groups, only when
//                                                compact_swa_by_cp is true
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
                                                          int            cp_size,
                                                          bool           compact_swa_by_cp);

}  // namespace rtp_llm
