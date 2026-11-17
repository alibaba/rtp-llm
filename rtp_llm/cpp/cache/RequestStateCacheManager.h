#pragma once

#include "rtp_llm/cpp/cache/FullCacheManager.h"

namespace rtp_llm {

// Ownership is exactly the ordinary BlockIds/DeviceBlockPool ownership. Columns
// describe state roles (committed/scratch, or replay), never sequence pages.
// All allocation and admission entry points use the same fixed requirement.
class RequestStateCacheManager: public FullCacheManager {
public:
    RequestStateCacheManager(GroupBase group, DeviceBlockPoolPtr pool, int group_id):
        FullCacheManager(std::move(group), std::move(pool), group_id) {
        RTP_LLM_CHECK_WITH_INFO(policy().group_type == CacheGroupType::REQUEST_STATE && policy().request_state_slots > 0
                                    && !prefixReuseEnabled(),
                                "invalid request state cache policy");
    }

    bool           malloc(BlockIds&                blocks,
                          int                      seq_len,
                          bool                     enable_reuse_cache   = false,
                          int                      reserve_step         = 0,
                          std::vector<size_t>*     backfilled_positions = nullptr,
                          const RequiredPositions& required_positions   = {}) override;
    int            needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const override;
    int            estimatePeakNeedBlocks(int                     seq_len,
                                          const BlockIndicesType& current_blocks,
                                          int                     remaining_tokens,
                                          int                     reserve_step,
                                          bool                    enable_reuse_cache) const override;
    int            estimateInitialBatchPeakNeedBlocks(int  seq_len,
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

    bool hasSparseSlots() const override {
        return false;
    }
    bool hasKernelBlockSubdiv() const override {
        return false;
    }
};

}  // namespace rtp_llm
