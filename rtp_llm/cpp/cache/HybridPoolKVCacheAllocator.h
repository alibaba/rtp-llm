#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0,
                               RoleType                           role_type           = RoleType::PDFUSION);

    BlockAddrInfo          convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override;
    void blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) override;

    GroupedCacheLayerLayout allLayerCacheBase() const override;

    size_t                  freeBlocksNum() const override;
    size_t                  availableBlocksNum() const override;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free) override;
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override;
    size_t                  requestRefBlocksNum() const override;
    size_t                  connectorRefBlocksNum() const override;
    size_t                  blockCacheRefBlocksNum() const override;
    size_t                  notInUseBlocksNum() const override;
    size_t                  availableTokensNum() const override;
    size_t                  totalTokensNum() const override;
    size_t                  totalBlocksNum() const override;
    size_t                  maxAvailableTokensNum() const override;
    KVCacheTokenCapacity    tokenCapacity() const override;
    std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const override;
    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t getMrCostTimeMs() const override;

    BlockPoolPtr blockPool(std::string_view tag) const override {
        return kv_cache_groups_.at(std::string(tag))->blockPool();
    }

    size_t poolCount() const {
        return kv_cache_groups_.size();
    }

private:
    bool   doInit() override;
    size_t reservableAvailableBlocksNum() const override;

    void referenceBlocksInGroup(std::string_view        tag,
                                const BlockIndicesType& blocks,
                                bool                    is_connector = false) const override;
    void freeBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) override;
    bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const override;

    size_t minTokenCapacity(bool use_available_blocks) const;
    size_t logicalCoverageUnitTokens() const;
    size_t totalReservableAvailableBlocks() const;
    size_t
    reserveBlocksForPool(std::string_view tag, size_t reserve_blocks, size_t total_reservable_available_blocks) const;

    RoleType role_type_{RoleType::PDFUSION};
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
