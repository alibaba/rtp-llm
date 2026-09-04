#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

class LinearKVCacheGroup: public KVCacheGroup {
public:
    LinearKVCacheGroup(const LayerIdsType&                 layer_ids,
                       std::shared_ptr<KVCacheSpec>        kvcache_spec,
                       BlockPoolPtr                        block_pool,
                       int                                 group_id,
                       SharedBlockCache*                   shared_cache     = nullptr,
                       const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, shared_cache, metrics_reporter) {}

    // Legacy sparse-checkpoint constructor. Request-cache mode ignores these
    // values, but keeping this path preserves behavior for models that have not
    // opted into ENABLE_LINEAR_ATTN_REQUEST_CACHE.
    LinearKVCacheGroup(const LayerIdsType&                 layer_ids,
                       std::shared_ptr<KVCacheSpec>        kvcache_spec,
                       BlockPoolPtr                        block_pool,
                       int                                 group_id,
                       int                                 unused_linear_step,
                       SharedBlockCache*                   shared_cache            = nullptr,
                       const kmonitor::MetricsReporterPtr& metrics_reporter        = nullptr,
                       int                                 unused_linear_fixed_cap = 0):
        LinearKVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, shared_cache, metrics_reporter) {
        linear_step_      = unused_linear_step;
        linear_fixed_cap_ = unused_linear_fixed_cap;
    }

    void setRequestCacheMode(bool enabled) {
        request_cache_mode_ = enabled;
    }

    void setRequestCacheAlignmentBlocks(int blocks) {
        request_cache_alignment_blocks_ = std::max(blocks, 1);
    }

    MatchResult match(const CacheKeysType& cache_keys) override;
    // Match a single cache key (used by Hybrid allocator to do right-to-left joint matching).
    MatchResult matchSingleKey(CacheKeyType cache_key) const;
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
    // A live request retains its read state and next write state. Prefill may
    // additionally pin the latest request-cache candidate until insertion;
    // older candidates are released as soon as the next aligned boundary wins.
    static constexpr int kResidentBlocksPerRequest       = 2;
    bool                 request_cache_mode_             = false;
    int                  request_cache_alignment_blocks_ = 1;
    int                  linear_step_                    = 1;
    int                  linear_fixed_cap_               = 0;
};

using LinearKVCacheGroupPtr = std::shared_ptr<LinearKVCacheGroup>;

}  // namespace rtp_llm
