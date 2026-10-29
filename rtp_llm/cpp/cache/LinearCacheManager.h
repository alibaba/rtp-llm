#pragma once

#include <memory>
#include <limits>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"

namespace rtp_llm {

class LinearCacheManager: public SingleTypeCacheManager {
public:
    LinearCacheManager(GroupBase          cache_group,
                       DeviceBlockPoolPtr block_pool,
                       int                group_id,
                       int                tail_blocks = std::numeric_limits<int>::max(),
                       bool               enable_linear_cache_reclaim = false):
        SingleTypeCacheManager(std::move(cache_group), std::move(block_pool), group_id),
        tail_blocks_(tail_blocks),
        enable_linear_cache_reclaim_(enable_linear_cache_reclaim) {
        RTP_LLM_CHECK_WITH_INFO(tail_blocks > 0, "linear cache tail blocks must be positive");
    }

    // Transition-only overload.
    LinearCacheManager(const LayerIdsType&          layer_ids,
                       std::shared_ptr<KVCacheSpec> kvcache_spec,
                       DeviceBlockPoolPtr           block_pool,
                       int                          group_id,
                       int                          tail_blocks = std::numeric_limits<int>::max(),
                       CacheGroupPolicy             policy      = defaultCacheGroupPolicy(CacheGroupType::LINEAR)):
        SingleTypeCacheManager(layer_ids, kvcache_spec, block_pool, group_id, policy), tail_blocks_(tail_blocks) {
        RTP_LLM_CHECK_WITH_INFO(tail_blocks > 0, "linear cache tail blocks must be positive");
    }

    bool malloc(BlockIds&                block_ids,
                int                      seq_len,
                bool                     enable_reuse_cache   = false,
                int                      reserve_step         = 0,
                std::vector<size_t>*     backfilled_positions = nullptr,
                const RequiredPositions& required_positions   = {}) override;

    bool mallocForDecode(BlockIds&            block_ids,
                         int                  seq_len,
                         bool                 enable_reuse_cache   = false,
                         int                  reserve_step         = 0,
                         std::vector<size_t>* backfilled_positions = nullptr) override;

    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void releaseRequestStateToDecodeTail(BlockIds& block_ids, int reserve_step) override;
    size_t         historyReclaimEnd(int committed_tokens) const override;
    int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const override;
    int            estimatePeakNeedBlocks(int                     seq_len,
                                          const BlockIndicesType& current_block_indices,
                                          int                     remaining_tokens,
                                          int                     reserve_step,
                                          bool                    enable_reuse_cache,
                                          bool                    reclaim_history = false) const override;
    int            estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                      int  common_seq_len,
                                                      int  remaining_tokens,
                                                      int  reserve_step,
                                                      bool enable_reuse_cache,
                                                      int  target_batch_size,
                                                      bool reclaim_history = false) const override;
    NeedBlocksInfo getNeedBlocks(int                      common_seq_len,
                                 int                      seq_len,
                                 int                      reserve_step,
                                 int                      reuse_blocks_len,
                                 bool                     reuse_enabled           = false,
                                 const RequiredPositions& required_positions      = {},
                                 size_t                   reclaimed_prefix_blocks = 0) const override;
    bool           shouldMaterializeBlock(int pos, int seq_len, int reserve_step, bool enable_reuse_cache) const;

private:
    // History-reclaim estimation (opt-in via enable_linear_cache_reclaim):
    // simulates committed-slot retirement to bound the physical-block peak.
    int  estimateHistoryReclaimPeak(std::vector<int> block_costs,
                                    int              seq_len,
                                    int              remaining_tokens,
                                    int              reserve_step,
                                    bool             enable_reuse_cache,
                                    bool             initialized) const;
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    int  materializedTailBlockCount() const;
    int  retainedTailBlockCount() const;
    int  retainedDecodeBlocks(int reserve_step) const;
    bool mallocImpl(BlockIds&                block_ids,
                    int                      seq_len,
                    int                      reserve_step,
                    bool                     is_prefill,
                    bool                     enable_reuse_cache,
                    std::vector<size_t>*     backfilled_positions,
                    const RequiredPositions& required_positions);

private:
    int  tail_blocks_;
    bool enable_linear_cache_reclaim_ = false;
};

using LinearCacheManagerPtr = std::shared_ptr<LinearCacheManager>;

}  // namespace rtp_llm
