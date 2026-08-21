#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

namespace rtp_llm {

class LoadAsyncContext;

class HybridKVCacheAllocator: public KVCacheAllocator, public std::enable_shared_from_this<HybridKVCacheAllocator> {
public:
    HybridKVCacheAllocator(const CacheConfig&                 config,
                           AllocationType                     allocation_type     = AllocationType::DEVICE,
                           const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                           int64_t                            reserve_block_ratio = 0);

    void free(const FreeInfo& free_info) override;
    void insertIntoCache(const InsertInfo& insert_info) override;

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;

    bool updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                       const std::vector<int>&         block_src_batch,
                       bool                            copy_last_block,
                       std::vector<TaggedBlockIdPair>& block_update_mapping) override;

    int                          seqSizePerBlock() const override;
    int                          singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                       int                            seq_len,
                                                       int                            reserve_step) const override;
    std::vector<KVCacheGroupPtr> cacheGroups() const override {
        return kv_cache_groups_;
    }

protected:
    struct PreparedKVCache {
        size_t                            matched_device_blocks = 0;
        size_t                            total_logical_blocks  = 0;
        std::vector<RequiredPositions>    required_positions;
        std::vector<BlockIndicesType>     referenced_blocks;
        std::vector<size_t>               original_sizes;
        MallocStatus                      materialize_status = MallocStatus::NONE;
    };

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

    std::shared_ptr<LoadAsyncContext> prepareKVCache(const CacheKeysType&                 cache_keys,
                        BatchKVCacheResource&                kv_resource,
                        const std::shared_ptr<CPSlotMapper>& cp_mapper,
                        PreparedKVCache&                     prepared);
    bool                              materializeInitialBlocks(const MallocInfo& malloc_info,
                                                               PreparedKVCache&  prepared,
                                                               LoadAsyncContext* context,
                                                               size_t            matched_blocks);
    bool                              finishDeferredMalloc(const MallocInfo& malloc_info,
                                                           PreparedKVCache&  prepared,
                                                           LoadAsyncContext& context,
                                                           size_t            matched_blocks);

    std::vector<BlockRefTransition>
    freeBlocksInGroup(int group_id, const BlockIndicesType& blocks, BlockRefType ref_type);
    virtual MallocStatus evaluatePreparedInitCapacity(const MallocInfo&       malloc_info,
                                                      size_t                  reserve_blocks,
                                                      const PreparedKVCache& prepared,
                                                      bool                   has_load_context) const;
    virtual bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    virtual void logMallocFailure(const MallocInfo& malloc_info,
                                  const char*       phase,
                                  int               failed_batch,
                                  int               failed_group,
                                  bool              incremental,
                                  int               failed_need_blocks) const;
    size_t       loadTargetPosition(size_t                               path_index,
                                    size_t                               group_id,
                                    const std::shared_ptr<CPSlotMapper>& mapper,
                                    int                                  cp_scale) const;
    bool         cpCompactSwaGroup(size_t group_id, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void         rollbackBlockIdsToSize(int group_id, BlockIds& block_ids, size_t original_size);
    void         rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                    const std::vector<BlockIndicesType>& referenced_blocks,
                                    const std::vector<size_t>&           original_sizes,
                                    BlockReleaseBatch&                   releases);
    virtual void copyBlockMappingForGroup(int group_id, const std::vector<BlockIdPair>& block_update_mapping) const;
    virtual MemoryType memoryTypeForGroup(int group_id) const;

    std::vector<KVCacheGroupPtr> kv_cache_groups_;
    std::vector<int>             full_group_ids_;
    std::vector<int>             linear_group_ids_;
    std::vector<int>             swa_group_ids_;
};

using HybridKVCacheAllocatorPtr = std::shared_ptr<HybridKVCacheAllocator>;

}  // namespace rtp_llm
