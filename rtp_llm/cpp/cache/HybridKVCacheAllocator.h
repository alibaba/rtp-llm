#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

namespace rtp_llm {

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

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    int seqSizePerBlock() const override;
    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                              int                            seq_len,
                              int                            reserve_step) const override;

protected:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          getNeedBlocks(const MallocInfo& malloc_info) const override;
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

    int reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource);

    virtual void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const = 0;
    virtual void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false)            = 0;
    virtual bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    void         rollbackBlockIdsToSize(int gid, BlockIds& block_ids, size_t original_size);
    void         rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                    const std::vector<BlockIndicesType>& referenced_blocks,
                                    const std::vector<size_t>&           original_sizes);
    void         rollbackIncrMalloc(BatchKVCacheResource&                   kv_resource,
                                    const std::vector<std::vector<size_t>>& original_sizes,
                                    int                                     failed_batch);

    std::vector<KVCacheGroupPtr> kv_cache_groups_;
    std::vector<int>             full_group_ids_;
    std::vector<int>             linear_group_ids_;
    std::vector<int>             swa_group_ids_;
};

using HybridKVCacheAllocatorPtr = std::shared_ptr<HybridKVCacheAllocator>;

}  // namespace rtp_llm
