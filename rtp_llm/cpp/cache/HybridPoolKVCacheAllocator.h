#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0);

    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int               layer_id,
                                                KVCacheRegionName region_name,
                                                int               block_id,
                                                int               partition_count,
                                                int               partition_id) const override;

    CacheLayerLayout allLayerCacheBase() const override;

    size_t                  freeBlocksNum() const override;
    size_t                  availableBlocksNum() const override;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free) override;
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override;
    size_t                  requestRefBlocksNum() const override;
    size_t                  connectorRefBlocksNum() const override;
    size_t                  blockCacheRefBlocksNum() const override;
    size_t                  notInUseBlocksNum() const override;
    size_t                  availableTokensNum() const override;
    size_t                  totalBlocksNum() const override;
    size_t                  maxAvailableTokensNum() const override;
    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t                 getMrCostTimeMs() const override;

private:
    bool doInit() override;

    void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const override;
    void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) override;

    int groupIdForLayerRegion(int layer_id, KVCacheRegionName region_name) const;
    int defaultGroupIdForLayer(int layer_id) const;

    std::vector<BlockPoolPtr> group_block_pools_;
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
