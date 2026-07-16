#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

namespace rtp_llm {

class HybridTypeKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridTypeKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0);

    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    CacheLayerLayout allLayerCacheBase() const override;

private:
    bool doInit() override;

    void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const override;
    void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) override;

    // global layer id -> group id
    std::vector<int> layer_to_group_id_;
    // global layer id -> local layer id
    std::vector<int> global_layer_to_local_id_;
};

using HybridTypeKVCacheAllocatorPtr = std::shared_ptr<HybridTypeKVCacheAllocator>;

}  // namespace rtp_llm
