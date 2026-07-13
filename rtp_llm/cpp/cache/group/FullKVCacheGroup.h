#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/group/KVCacheGroup.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(const LayerIdsType&                 layer_ids,
                     std::shared_ptr<KVCacheSpec>        kvcache_spec,
                     DeviceBlockPoolPtr                  block_pool,
                     int                                 group_id,
                     const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr,
                     CacheGroupPolicy                    policy           = CacheGroupPolicy{}):
        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, policy, metrics_reporter) {}

    bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) override;
    MatchResult matchPrefix(const CacheKeysType& cache_keys) const override;
    void        free(const BlockIndicesType& block_indices) override;
    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    int  needBlocksNum(int seq_len, int current_blocks = 0, int reserve_step = 0) const override;
    NeedBlocksInfo getNeedBlocks(int  common_seq_len,
                                 int  seq_len,
                                 int  reserve_step,
                                 int  reuse_blocks_len,
                                 bool reuse_enabled = false) const override;
    void           reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) override;

private:
};

}  // namespace rtp_llm
