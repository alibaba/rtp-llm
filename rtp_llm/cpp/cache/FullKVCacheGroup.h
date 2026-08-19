#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(GroupBase cache_group, DeviceBlockPoolPtr block_pool, int group_id):
        KVCacheGroup(std::move(cache_group), std::move(block_pool), group_id) {}

    // Transition-only overload.
    FullKVCacheGroup(const LayerIdsType&          layer_ids,
                     std::shared_ptr<KVCacheSpec> kvcache_spec,
                     DeviceBlockPoolPtr           block_pool,
                     int                          group_id,
                     CacheGroupPolicy policy = defaultCacheGroupPolicy(CacheGroupType::FULL)):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, policy) {}

    bool        malloc(BlockIds&                  block_indices,
                       int                        seq_len,
                       bool                       enable_reuse_cache   = false,
                       int                        reserve_step         = 0,
                       std::vector<size_t>*       backfilled_positions = nullptr,
                       const RequiredPositions&  required_positions = {}) override;
    void release(const BlockIndicesType& block_indices, BlockRefType ref_type = BlockRefType::REQUEST) override;
    void free(const BlockIndicesType& block_indices) override;
    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    int  needBlocksNum(int seq_len, int current_blocks = 0, int reserve_step = 0) const override;
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
                                 bool reuse_enabled = false,
                                 const RequiredPositions& required_positions = {}) const override;
    void           reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) override;

private:
};

}  // namespace rtp_llm
