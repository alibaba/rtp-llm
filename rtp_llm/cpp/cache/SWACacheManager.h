#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"

namespace rtp_llm {

class SWACacheManager: public SingleTypeCacheManager {
public:
    // Half-open physical slot interval containing the restored prefix history
    // needed by the first resumed query. It is validated, never backfilled.
    struct HistorySlots {
        size_t begin = 0;
        size_t end   = 0;
    };
    SWACacheManager(GroupBase cache_group, DeviceBlockPoolPtr block_pool, int group_id, int linear_step = 0):
        SingleTypeCacheManager(std::move(cache_group), std::move(block_pool), group_id), linear_step_(linear_step) {}

    // Transition-only overload.
    SWACacheManager(const LayerIdsType&          layer_ids,
                    std::shared_ptr<KVCacheSpec> kvcache_spec,
                    DeviceBlockPoolPtr           block_pool,
                    int                          group_id,
                    int                          linear_step = 0,
                    CacheGroupPolicy             policy      = defaultCacheGroupPolicy(CacheGroupType::SWA)):
        SingleTypeCacheManager(layer_ids, kvcache_spec, block_pool, group_id, policy), linear_step_(linear_step) {}

    bool malloc(BlockIds&                block_ids,
                int                      seq_len,
                bool                     enable_reuse_cache   = false,
                int                      reserve_step         = 0,
                std::vector<size_t>*     backfilled_positions = nullptr,
                const RequiredPositions& required_positions   = {}) override;
    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    bool mallocWithHistory(BlockIds&                block_ids,
                           int                      seq_len,
                           bool                     enable_reuse_cache,
                           int                      reserve_step,
                           std::vector<size_t>*     backfilled_positions,
                           const RequiredPositions& required_positions,
                           HistorySlots             history);
    void removeSkippedBlocksWithHistory(
        BlockIds& block_ids, bool enable_reuse_cache, int reserve_step, HistorySlots history);
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

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    int  activeTailBlockCount() const;
    bool effectiveReuseCacheForAllocation(bool enable_reuse_cache) const;
    bool shouldCheckSWATailBlockIds() const;
    void checkSWATailBlockIds(const BlockIds& block_ids, const char* caller) const;
    void checkHistoryBlockIds(const BlockIds& block_ids,
                              HistorySlots    history,
                              const char*     caller,
                              bool            check_tail = true) const;
    bool mallocImpl(BlockIds&                      block_ids,
                    int                            seq_len,
                    bool                           enable_reuse_cache,
                    int                            reserve_step,
                    std::vector<size_t>*           backfilled_positions,
                    const RequiredPositions&       required_positions,
                    std::optional<HistorySlots>    history);
    void removeSkippedBlocksImpl(
        BlockIds& block_ids, bool enable_reuse_cache, int reserve_step, std::optional<HistorySlots> history);

    int linear_step_ = 0;
};

using SWACacheManagerPtr = std::shared_ptr<SWACacheManager>;

}  // namespace rtp_llm
