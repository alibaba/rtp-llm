#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(const LayerIdsType&          layer_ids,
                     std::shared_ptr<KVCacheSpec> kvcache_spec,
                     BlockPoolPtr                 block_pool,
                     int                          group_id):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id) {}

    bool        malloc(BlockIndicesType& block_indices,
                       int               seq_len,
                       bool              enable_reuse_cache = false,
                       int               reserve_step       = 0) override;
    MatchResult match(const CacheKeysType& cache_keys, const std::vector<std::vector<int>>& mm_intervals) override;
    void        free(const BlockIndicesType& block_indices) override;
    void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) override;
    void           removeSkippedBlocks(BlockIndicesType& block_indices,
                                       bool              enable_reuse_cache = false,
                                       int               reserve_step       = 0) override;
    int            needBlocksNum(int seq_len, int current_blocks = 0, int reserve_step = 0) const override;
    NeedBlocksInfo getNeedBlocks(int  common_seq_len,
                                 int  seq_len,
                                 int  reserve_step,
                                 int  reuse_blocks_len,
                                 bool reuse_enabled = false) const override;
    void           reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices) override;

private:
};

}  // namespace rtp_llm
