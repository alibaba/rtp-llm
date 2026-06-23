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
                       int                          linear_step      = 0,
                       SharedBlockCache*            shared_cache     = nullptr,
                       const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr,
                       CacheGroupPolicy policy = defaultCacheGroupPolicy(CacheGroupType::LINEAR)):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, policy, shared_cache, metrics_reporter),
        linear_step_(linear_step) {}

    // Match a single cache key (used by Hybrid allocator to do right-to-left joint matching).
    MatchResult matchSingleKey(CacheKeyType cache_key) const override;
    bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) override;

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

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;

private:
    // NOTE: linear attention cache can be sparsified; current implementation is conservative:
    // - always keep the last 2 blocks (decode edge case: read block i, write block i+1)
    // - other blocks can be freed (set to NULL_BLOCK_IDX)
    int linear_step_ = 0;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
