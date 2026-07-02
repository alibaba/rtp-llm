#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);
std::vector<size_t> blockPositionsForCacheTransfer(size_t                             block_num,
                                                   size_t                             reuse_block_size,
                                                   bool                               use_hybrid,
                                                   bool                               transfer_tail_blocks,
                                                   size_t                             tail_block_count,
                                                   bool                               hybrid_full_from_begin);

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag);

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
// the order they appear within the rank.
std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         reuse_block_size,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size);
std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t                      total_logical_blocks,
                                                          size_t                      reuse_block_size,
                                                          bool                        use_hybrid,
                                                          bool                        cp_shardable,
                                                          bool                        cp_compact_tail_blocks,
                                                          size_t                      tail_block_count,
                                                          int                         cp_rank,
                                                          int                         cp_size);

}  // namespace rtp_llm
