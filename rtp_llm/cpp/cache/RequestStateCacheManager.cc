#include "rtp_llm/cpp/cache/RequestStateCacheManager.h"

#include <algorithm>

namespace rtp_llm {

bool RequestStateCacheManager::malloc(BlockIds&                blocks,
                                      int                      seq_len,
                                      bool                     enable_reuse_cache,
                                      int                      reserve_step,
                                      std::vector<size_t>*     backfilled_positions,
                                      const RequiredPositions& required_positions) {
    RTP_LLM_CHECK_WITH_INFO(!enable_reuse_cache && required_positions.empty(),
                            "request state does not support prefix reuse or chunked prefill");
    RTP_LLM_CHECK_WITH_INFO(blocks.blocksNum() <= policy().request_state_slots,
                            "request state row exceeds its declared slot count");
    return FullCacheManager::malloc(blocks, seq_len, false, reserve_step, backfilled_positions);
}

int RequestStateCacheManager::needBlocksNum(int seq_len, int current_blocks, int /*reserve_step*/) const {
    return std::max((seq_len > 0 ? static_cast<int>(policy().request_state_slots) : 0) - current_blocks, 0);
}

int RequestStateCacheManager::estimatePeakNeedBlocks(int                     seq_len,
                                                     const BlockIndicesType& current_blocks,
                                                     int                     remaining_tokens,
                                                     int /*reserve_step*/,
                                                     bool /*enable_reuse_cache*/) const {
    const auto allocated = std::count_if(
        current_blocks.begin(), current_blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); });
    return needBlocksNum(std::max(seq_len, remaining_tokens), static_cast<int>(allocated));
}

int RequestStateCacheManager::estimateInitialBatchPeakNeedBlocks(int seq_len,
                                                                 int /*common_seq_len*/,
                                                                 int remaining_tokens,
                                                                 int /*reserve_step*/,
                                                                 bool /*enable_reuse_cache*/,
                                                                 int target_batch_size) const {
    return needBlocksNum(std::max(seq_len, remaining_tokens), 0) * std::max(target_batch_size, 1);
}

NeedBlocksInfo RequestStateCacheManager::getNeedBlocks(int common_seq_len,
                                                       int seq_len,
                                                       int /*reserve_step*/,
                                                       int /*reuse_blocks_len*/,
                                                       bool /*reuse_enabled*/,
                                                       const RequiredPositions& /*required_positions*/) const {
    // initMalloc first allocates the common prefix; this first implementation
    // admits one sequence per request, so no writable state is shared by beams.
    const int common = needBlocksNum(common_seq_len, 0);
    return {common, std::max(needBlocksNum(seq_len, 0) - common, 0)};
}

}  // namespace rtp_llm
