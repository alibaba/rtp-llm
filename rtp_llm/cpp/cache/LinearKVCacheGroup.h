#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

class LinearKVCacheGroup: public KVCacheGroup {
public:
    LinearKVCacheGroup(const LayerIdsType&          layer_ids,
                       std::shared_ptr<KVCacheSpec> kvcache_spec,
                       BlockPoolPtr                 block_pool,
                       int                          group_id,
                       int                          linear_step = 0,
                       int                          fixed_cap   = 0):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id), linear_step_(linear_step), fixed_cap_(fixed_cap) {}

    MatchResult match(const CacheKeysType& cache_keys) override;
    // Match a single cache key (used by Hybrid allocator to do right-to-left joint matching).
    MatchResult matchSingleKey(CacheKeyType cache_key) const;
    bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;

    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void free(const BlockIndicesType& block_indices) override;
    void reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) override;
    int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const override;
    NeedBlocksInfo getNeedBlocks(int  common_seq_len,
                                 int  seq_len,
                                 int  reserve_step,
                                 int  reuse_blocks_len,
                                 bool reuse_enabled = false) const override;
    bool           shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const;

    // Record that an external caller (e.g. HybridPool::reuseCache) pre-populated
    // `block_ids` to reflect a prefix of `reuse_slots` logical block slots.
    // Subsequent `malloc` calls will use this as the baseline to compute how
    // many block boundaries have been crossed since last call.
    // No-op when fixed_cap_ == 0.
    void recordReuse(const BlockIds& block_ids, int reuse_slots);

    int fixedCap() const {
        return fixed_cap_;
    }

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;

    // Total logical blocks occupied by `seq_len` tokens plus any reserve_step extension.
    int logicalSlots(int seq_len, int reserve_step) const;

private:
    // NOTE: linear attention cache can be sparsified; current implementation is conservative:
    // - always keep the last 2 blocks (decode edge case: read block i, write block i+1)
    // - other blocks can be freed (set to NULL_BLOCK_IDX)
    int linear_step_ = 0;

    // Ring-buffer capacity. 0 = legacy unbounded behavior (existing tests rely on this).
    // >0 = block_ids maintains at most `fixed_cap_` entries; malloc rotates old blocks
    // back to the pool when sequence advances past the cap.
    int fixed_cap_ = 0;

    // Track block→key mapping so we can remove stale cache entries when
    // block content changes (e.g. SWA ring buffer update, state update).
    std::unordered_map<BlockIdxType, CacheKeyType> block_to_key_;

    // Per-request logical seq_slots tracking for ring-buffer rotation detection.
    // Keyed by BlockIds pointer identity. Only used when fixed_cap_ > 0.
    // Lazy cleanup: when the same BlockIds pointer is reused for a new request
    // (block_ids.blocksNum()==0 but map has entry>0), reset it.
    std::unordered_map<const BlockIds*, int> last_seq_slots_;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
