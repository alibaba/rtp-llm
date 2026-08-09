#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
                               int64_t                            reserve_block_ratio = 0);

    void free(const FreeInfo& free_info) override;
    void insertIntoCache(const InsertInfo& insert_info) override;

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<GroupBlockIdPair>& block_update_mapping) override;

    int                      seqSizePerBlock() const override;
    int                      singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const override;
    std::vector<std::string> independentEvictionTags() const override;

    BlockAddrInfo          convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override;
    void blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) override;

    GroupedCacheLayerLayout allLayerCacheBase() const override;

    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free) override;
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override;
    // Capacity is reported in the finalized global token coordinate system.
    // Dynamic FULL attention pools provide the live availability signal.
    size_t                                  totalTokensNum() const override;
    size_t                                  availableTokensNum() const override;
    size_t                                  maxSequenceLength() const override;
    std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const override;

    // Per-pool access for diagnostics / per-pool metrics reporting.
    const std::unordered_map<std::string, BlockPoolPtr>& groupBlockPools() const {
        return group_block_pools_;
    }

    BlockPoolPtr getBlockPool(std::string_view tag) const override {
        const auto it = group_block_pools_.find(std::string(tag));
        return it == group_block_pools_.end() ? nullptr : it->second;
    }

private:
    bool   doInit() override;
    size_t reservableAvailableBlocksNum() const override;

    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
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
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

    int reuseCache(const CacheKeysType&                 cache_keys,
                   BatchKVCacheResource&                kv_resource,
                   const std::shared_ptr<CPSlotMapper>& cp_mapper);

    void referenceBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) const;
    void freeBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false);
    bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;

    bool                   skipReuseCacheGroup(std::string_view tag) const;
    bool                   cpCompactSwaGroup(std::string_view tag, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void                   rollbackGroupMalloc(std::string_view           tag,
                                               BlockIds&                  block_ids,
                                               size_t                     original_size,
                                               const std::vector<size_t>& filled_positions);
    void                   rollbackInitMalloc(BatchKVCacheResource&                                       kv_resource,
                                              const std::unordered_map<std::string, BlockIndicesType>&    referenced_blocks,
                                              const std::unordered_map<std::string, size_t>&              original_sizes,
                                              const std::unordered_map<std::string, std::vector<size_t>>& backfilled_positions);
    const KVCacheGroupPtr& cacheGroupForTag(std::string_view tag, const char* context) const;
    const BlockPoolPtr&    blockPoolForTag(std::string_view tag, const char* context) const;

    size_t                   globalUsableTokens() const;
    std::vector<std::string> dynamicFullAttentionTags() const;
    size_t                   minPoolTokens(bool use_available_blocks) const;
    size_t                   totalReservableAvailableBlocks() const;
    size_t
    reserveBlocksForPool(std::string_view tag, size_t reserve_blocks, size_t total_reservable_available_blocks) const;

    std::unordered_map<std::string, KVCacheGroupPtr> kv_cache_groups_;
    std::vector<std::string>                         full_group_tags_;
    std::vector<std::string>                         linear_group_tags_;
    std::vector<std::string>                         swa_group_tags_;
    std::unordered_map<std::string, BlockPoolPtr>    group_block_pools_;
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
