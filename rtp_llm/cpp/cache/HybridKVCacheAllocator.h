#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

namespace rtp_llm {

class HybridLayerKVCacheAllocator: public KVCacheAllocator {
public:
    HybridLayerKVCacheAllocator(const CacheConfig&                 config,
                                rtp_llm::DeviceBase*               device,
                                AllocationType                     allocation_type  = AllocationType::DEVICE,
                                const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    bool               init() override;
    void               free(const FreeInfo& free_info) override;
    void               insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const override;
    std::shared_ptr<KVCacheResource> incrKVCacheRef(KVCacheResource&     kvcache_resource,
                                                    const CacheKeysType& cache_keys) override;
    void                             decrKVCacheRef(KVCacheResource& kvcache_resource) override;
    CacheLayerLayout                 allLayerCacheBase() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    int seqSizePerBlock() const override;
    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len) const override;

private:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          getNeedBlocks(const MallocInfo& malloc_info) const override;

    // Joint match across groups. Returns reuse_blocks decided by full groups + linear groups.
    int  reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource);
    void referenceValidBlocks(const BlockIndicesType& blocks) const;

private:
    std::vector<KVCacheGroupPtr> kv_cache_groups_;

    std::vector<int> full_group_ids_;
    std::vector<int> linear_group_ids_;

    // global layer id -> group id
    std::vector<int> layer_to_group_id_;
};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocator>;

}  // namespace rtp_llm
