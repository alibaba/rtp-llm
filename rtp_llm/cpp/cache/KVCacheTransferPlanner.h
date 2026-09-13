#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

std::vector<size_t>
blockPositionsForCacheTransfer(size_t block_num, size_t first_full_block, bool use_hybrid, CacheGroupType group_type);

std::string layerRegionCacheTransferKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name);

// A non-FULL group uses virtual-block cache layout when one physical row spans
// an entire page-RR stripe: V = physical_page_tokens * shard_size. This is a
// storage-layout predicate only; it does not imply Query Context Parallelism.
bool usesVirtualBlockCacheLayout(CacheGroupType group_type,
                                 size_t         physical_page_tokens,
                                 size_t         group_block_tokens,
                                 int            shard_size);

// One iteration step of cache_store registration: pair the cache_key at
// ``key_index`` (FULL-length namespace) with the kv_cache_offset slot at
// ``offset_index`` (rank-local namespace). They differ for sharded FULL pages
// and compact LINEAR/SWA checkpoints (see below).
struct CacheStoreBlockPair {
    int key_index;
    int offset_index;
};

struct CacheStorePublishRange {
    size_t begin_block = 0;
    size_t end_block   = 0;
    bool   terminal    = false;
};

// Build the per-prefill-write iteration plan for cache_store registration.
//
// Background: ``cache_keys`` is always the FULL logical-block hash sequence
// (length = total_logical_blocks). ``kv_cache_offset`` is per-group and
// per-rank. Ordinary non-FULL groups keep the full block list. FULL groups
// under CP page-RR hold only their 1/cp_size owned blocks, **compactly**, in
// appearance order: local index ``i`` ↔ logical position
// ``cp_rank + i*cp_size``. Non-FULL groups with virtual-block cache layout
// instead hold one local slot per D logical pages, shared by every rank's
// distinct head shard.
//
// To register the right key with the right buffer the planner emits:
//   * (pos, pos)                              — non-CP / non-FULL groups
//   * (cp_rank + i*cp_size, i) for owned i    — CP-sharded FULL groups
//   * ((i+1)*cp_size-1, i)                    — virtual-block SWA groups
//   * (total-1, ceil(total/cp_size)-1)         — virtual-block LINEAR terminal
//
// ``virtual_block_cache_layout`` opts a LINEAR or SWA group into the latter
// rules. Callers derive it from group geometry with
// ``usesVirtualBlockCacheLayout``.
//
// Without this re-pairing the prefill side advertises ``cache_keys[i]``
// (== key for logical position i) attached to data from logical position
// ``cp_rank + i*cp_size`` — decode then receives content shifted by
// ``cp_rank`` slots and produces coherent-but-wrong output (DSV4 PD reuse
// regression seen 2026-05-12).
std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         first_full_block,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size,
                                                          bool           virtual_block_cache_layout = false);

// Restrict the regular cache-store plan to one incremental publication.
// FULL groups publish only the supplied half-open logical-block range;
// LINEAR groups publish their existing final-state entry only on a terminal
// publication. Other hybrid group policies are deliberately unsupported.
std::vector<CacheStoreBlockPair> buildIncrementalCacheStoreBlockPlan(size_t                        total_logical_blocks,
                                                                     size_t                        reuse_block_size,
                                                                     bool                          use_hybrid,
                                                                     CacheGroupType                group_type,
                                                                     int                           cp_rank,
                                                                     int                           cp_size,
                                                                     const CacheStorePublishRange& publish_range,
                                                                     bool virtual_block_cache_layout = false);

}  // namespace rtp_llm
