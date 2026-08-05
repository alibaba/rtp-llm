#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

namespace rtp_llm {

class HybridTypeKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridTypeKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0);

    BlockAddrInfo          convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override;
    GroupedCacheLayerLayout allLayerCacheBase() const override;

private:
    bool doInit() override;

    void referenceBlocksInGroup(std::string_view        tag,
                                const BlockIndicesType& blocks,
                                bool                    is_connector = false) const override;
    void freeBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) override;

    std::vector<BlockInfo> logicalGroupBlockBuffers(std::string_view tag, std::vector<BlockInfo> buffers) const;

    // tag -> (global layer id -> local layer id)
    std::unordered_map<std::string, std::vector<int>> group_global_layer_to_local_id_;
};

using HybridTypeKVCacheAllocatorPtr = std::shared_ptr<HybridTypeKVCacheAllocator>;

}  // namespace rtp_llm
