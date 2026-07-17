#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/allocator/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0,
                               RoleType                           role_type           = RoleType::PDFUSION);

    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo          convertIndexToAddr(int layer_id, int group_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int group_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, int group_id, int block_id, int partition_count, int partition_id) const override;
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) override;

    CacheLayerLayout allLayerCacheBase() const override;

    size_t                  freeBlocksNum() const override;
    size_t                  activeTreeCachedBlocksNum() const override;
    size_t                  availableTokensNum() const override;
    size_t                  totalTokensNum() const override;
    size_t                  totalBlocksNum() const override;
    size_t                  maxAvailableTokensNum() const override;
    KVCacheTokenCapacity    tokenCapacity(size_t default_seq_size_per_block) const override;
    std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const override;
    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t                 getMrCostTimeMs() const override;

    // Per-pool access for diagnostics / per-pool metrics reporting.
    const std::vector<DeviceBlockPoolPtr>& groupBlockPools() const override {
        return group_block_pools_;
    }

private:
    bool doInit() override;

    void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const override;
    void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) override;
    bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const override;

    int validateGroupIdForLayer(int layer_id, int group_id) const;
    int defaultGroupIdForLayer(int layer_id) const;
    size_t minTokenCapacity(bool use_free_blocks, bool full_groups_only) const;
    size_t totalReservableFreeBlocks() const;
    size_t reserveBlocksForPool(size_t gid, size_t reserve_blocks, size_t total_reservable_free_blocks) const;

    std::vector<DeviceBlockPoolPtr> group_block_pools_;
    RoleType                        role_type_{RoleType::PDFUSION};
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
