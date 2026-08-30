#pragma once

#include <memory>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"

namespace rtp_llm {

class LinearCacheManager: public SingleTypeCacheManager {
public:
    LinearCacheManager(const CacheGroup&                   cache_group,
                       std::vector<int>                    layer_ids,
                       BlockPoolPtr                        block_pool,
                       int                                 linear_step      = 0,
                       SharedBlockCache*                   shared_cache     = nullptr,
                       const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        SingleTypeCacheManager(
            cache_group, std::move(layer_ids), std::move(block_pool), shared_cache, metrics_reporter),
        linear_step_(linear_step) {}

    LinearCacheManager(const CacheGroup&                   cache_group,
                       BlockPoolPtr                        block_pool,
                       int                                 linear_step      = 0,
                       SharedBlockCache*                   shared_cache     = nullptr,
                       const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        LinearCacheManager(cache_group, {0}, std::move(block_pool), linear_step, shared_cache, metrics_reporter) {}

    MatchResult matchSingleKey(CacheKeyType cache_key) const override;
    bool        malloc(BlockIds&            block_ids,
                       int                  seq_len,
                       bool                 enable_reuse_cache   = false,
                       int                  reserve_step         = 0,
                       std::vector<size_t>* backfilled_positions = nullptr) override;

    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void free(const BlockIndicesType& block_indices) override;
    void reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) override;
    int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const override;
    int  estimatePeakNeedBlocks(int                     seq_len,
                                const BlockIndicesType& current_block_indices,
                                int                     remaining_tokens,
                                int                     reserve_step,
                                bool                    enable_reuse_cache) const override;
    int  estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                            int  common_seq_len,
                                            int  remaining_tokens,
                                            int  reserve_step,
                                            bool enable_reuse_cache,
                                            int  target_batch_size) const override;
    NeedBlocksInfo getNeedBlocks(int  common_seq_len,
                                 int  seq_len,
                                 int  reserve_step,
                                 int  reuse_blocks_len,
                                 bool reuse_enabled = false) const override;
    bool           shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const;

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    int  materializedTailBlockCount() const;
    int  retainedTailBlockCount() const;

private:
    // NOTE: linear attention cache can be sparsified; current implementation is conservative:
    // - materialize at least one policy tail block during allocation
    // - retain at least two tail blocks across decode cleanup
    // - other blocks can be freed (set to NULL_BLOCK_IDX)
    int linear_step_ = 0;
};

using LinearCacheManagerPtr = std::shared_ptr<LinearCacheManager>;

}  // namespace rtp_llm
