#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator:
    public KVCacheAllocator,
    public std::enable_shared_from_this<HybridPoolKVCacheAllocator> {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0,
                               RoleType                           role_type           = RoleType::PDFUSION);

    void free(const FreeInfo& free_info) override;
    void insertIntoCache(const InsertInfo& insert_info) override;

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;

    bool updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                       const std::vector<int>&         block_src_batch,
                       bool                            copy_last_block,
                       std::vector<TaggedBlockIdPair>& block_update_mapping) override;

    int              seqSizePerBlock() const override;
    int              singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           int                            seq_len,
                                           int                            reserve_step) const override;
    std::vector<int> independentEvictionGroupIds() const override;

    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo          convertIndexToAddr(int layer_id, int group_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int group_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, int group_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo          convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBufferByTag(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override;
    using KVCacheAllocator::blockBatchCopy;
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) override;
    void blockBatchCopyByTag(const std::vector<TaggedBlockIdPair>& copy_mapping) override;

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
    KVCacheTokenCapacity    tokenCapacity(size_t default_seq_size_per_block) const override;
    std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const override;
    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t getMrCostTimeMs() const override;

    // Per-pool access for diagnostics / per-pool metrics reporting.
    const std::vector<BlockPoolPtr>& groupBlockPools() const {
        return group_block_pools_;
    }
    BlockPoolPtr soleGroupBlockPool() const;

private:
    bool   doInit() override;
    size_t reservableAvailableBlocksNum() const override;

    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          getNeedBlocks(const MallocInfo& malloc_info) const override;
    int          estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                        int                    seq_len,
                                        int                    remaining_tokens,
                                        int                    reserve_step,
                                        bool                   enable_reuse_cache) const override;
    int          estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                    int  common_seq_len,
                                                    int  remaining_tokens,
                                                    int  reserve_step,
                                                    bool enable_reuse_cache,
                                                    int  target_batch_size) const override;
    void         checkCPShardedMallocResult(const MallocInfo& malloc_info) const override;
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

    // Per-pool capacity classification: with one BlockPool per group, a shortfall
    // in a single pool decides the verdict, so the aggregate base-class view is
    // not usable here.
    MallocStatus
    evaluateInitCapacity(const MallocInfo& malloc_info, size_t reserve_blocks, InitCapacityMode mode) const override;

    int        reuseCache(const CacheKeysType&                 cache_keys,
                          BatchKVCacheResource&                kv_resource,
                          const std::shared_ptr<CPSlotMapper>& cp_mapper);
    void       referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const;
    void       freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false);
    bool       hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    void       logMallocFailure(const MallocInfo& malloc_info,
                                const char*       phase,
                                int               failed_batch,
                                int               failed_group,
                                bool              incremental,
                                int               failed_need_blocks) const;
    bool       skipReuseCacheGroup(int gid) const;
    bool       cpCompactSwaGroup(int gid, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void       rollbackBlockIdsToSize(int gid, BlockIds& block_ids, size_t original_size);
    void       rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                  const std::vector<BlockIndicesType>& referenced_blocks,
                                  const std::vector<size_t>&           original_sizes);
    void       rollbackIncrMalloc(BatchKVCacheResource&                   kv_resource,
                                  const std::vector<std::vector<size_t>>& original_sizes,
                                  int                                     failed_batch);
    void       copyBlockMappingForGroup(int gid, const std::vector<BlockIdPair>& block_update_mapping) const;
    MemoryType memoryTypeForGroup(int gid) const;

    int    validateGroupIdForLayer(int layer_id, int group_id) const;
    int    defaultGroupIdForLayer(int layer_id) const;
    size_t minTokenCapacity(bool use_available_blocks, bool full_groups_only) const;
    size_t totalReservableAvailableBlocks() const;
    size_t reserveBlocksForPool(size_t gid, size_t reserve_blocks, size_t total_reservable_available_blocks) const;

    std::vector<BlockPoolPtr>    group_block_pools_;
    std::vector<KVCacheGroupPtr> kv_cache_groups_;
    std::vector<int>             full_group_ids_;
    std::vector<int>             linear_group_ids_;
    std::vector<int>             swa_group_ids_;
    RoleType                     role_type_{RoleType::PDFUSION};
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
