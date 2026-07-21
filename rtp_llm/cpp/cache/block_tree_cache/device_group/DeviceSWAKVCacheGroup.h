#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceKVCacheGroup.h"

namespace rtp_llm {

class DeviceSWAKVCacheGroup: public DeviceKVCacheGroup {
public:
    DeviceSWAKVCacheGroup(const LayerIdsType&                 layer_ids,
                          std::shared_ptr<KVCacheSpec>        kvcache_spec,
                          DeviceBlockPoolPtr                  block_pool,
                          int                                 group_id,
                          int                                 linear_step      = 0,
                          const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr,
                          CacheGroupPolicy                    policy = defaultCacheGroupPolicy(CacheGroupType::SWA)):
        DeviceKVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, policy, metrics_reporter),
        linear_step_(linear_step) {}

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

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    int  activeTailBlockCount() const;
    bool effectiveReuseCacheForAllocation(bool enable_reuse_cache) const;
    bool shouldCheckSWATailBlockIds() const;
    void checkSWATailBlockIds(const BlockIds& block_ids, const char* caller) const;

    int linear_step_ = 0;
};

using DeviceSWAKVCacheGroupPtr = std::shared_ptr<DeviceSWAKVCacheGroup>;

}  // namespace rtp_llm
