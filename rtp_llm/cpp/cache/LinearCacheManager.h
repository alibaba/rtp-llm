#pragma once

#include <memory>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"

namespace rtp_llm {

class LinearCacheManager: public SingleTypeCacheManager {
public:
    LinearCacheManager(GroupBase cache_group, DeviceBlockPoolPtr block_pool, int group_id, int linear_step = 0):
        SingleTypeCacheManager(std::move(cache_group), std::move(block_pool), group_id), linear_step_(linear_step) {}

    // Transition-only overload.
    LinearCacheManager(const LayerIdsType&          layer_ids,
                       std::shared_ptr<KVCacheSpec> kvcache_spec,
                       DeviceBlockPoolPtr           block_pool,
                       int                          group_id,
                       int                          linear_step = 0,
                       CacheGroupPolicy             policy      = defaultCacheGroupPolicy(CacheGroupType::LINEAR)):
        SingleTypeCacheManager(layer_ids, kvcache_spec, block_pool, group_id, policy), linear_step_(linear_step) {}

    bool malloc(BlockIds&                block_ids,
                int                      seq_len,
                bool                     enable_reuse_cache   = false,
                int                      reserve_step         = 0,
                std::vector<size_t>*     backfilled_positions = nullptr,
                const RequiredPositions& required_positions   = {}) override;

    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void           removeSkippedBlocksBefore(BlockIds& block_ids, int prefix_len, bool enable_reuse_cache);
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
    NeedBlocksInfo getNeedBlocks(int                      common_seq_len,
                                 int                      seq_len,
                                 int                      reserve_step,
                                 int                      reuse_blocks_len,
                                 bool                     reuse_enabled      = false,
                                 const RequiredPositions& required_positions = {}) const override;
    bool           shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const;

private:
    void removeSkippedBlocksThrough(BlockIds& block_ids, int last_position, bool enable_reuse_cache);
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
