#pragma once

#include <cstddef>
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

// Half-open absolute logical-page range. A zero window retains full history.
// Page rounding retains the partially covered first and last pages.
struct CacheTransferPageRange {
    size_t begin;
    size_t end;
};

CacheTransferPageRange cacheTransferPageRange(size_t context_tokens, size_t window_tokens, size_t tokens_per_block);

// Transport-only window for native FULL/page-RR caches, NOT the compact SWA
// layout below. Keys and offsets stay absolute, even when earlier pages are
// omitted. Callers must negotiate identical P/D window semantics first.
std::vector<CacheStoreBlockPair> buildFullCacheStoreBlockPlanForWindow(
    size_t context_tokens, size_t window_tokens, size_t tokens_per_block, int cp_rank, int cp_size);

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
//   * ((i+1)*cp_size-1, i)                    — CP-compact SWA/fixed groups
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

}  // namespace rtp_llm
