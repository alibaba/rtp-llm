#pragma once

#include <memory>
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
                       int                          linear_step = 0):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id), linear_step_(linear_step) {}

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

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    // Zero only the bytes this LINEAR group actually wrote (ssm_state +
    // conv_state, ~2 MB per layer per block) before returning blocks to the
    // pool. This breaks the chain "LINEAR fp32 SSM -> recycled by FULL ATTN
    // -> XQA paged-load reads stale fp32 bytes as bf16 K/V -> softmax NaN".
    // Cheaper than zeroing the full kv_block_stride (8 MB) in BlockPool::malloc:
    // we touch only the dirty region (1/4 the bandwidth) and only on the
    // LINEAR free path (not every malloc). FULL group's K/V residue is bf16
    // already so reusing it does not produce NaN, just bf16-magnitude noise
    // on padding positions, so we don't need a symmetric clean for FULL frees.
    void zeroLinearWriteRegion(const BlockIndicesType& block_ids) const;

private:
    // NOTE: linear attention cache can be sparsified; current implementation is conservative:
    // - always keep the last 2 blocks (decode edge case: read block i, write block i+1)
    // - other blocks can be freed (set to NULL_BLOCK_IDX)
    int linear_step_ = 0;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
