#pragma once

#include <memory>
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

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
    std::shared_ptr<KVCacheResourceV1> incrKVCacheRef(KVCacheResourceV1&   kvcache_resource,
                                                      const CacheKeysType& cache_keys) override;
    void                               decrKVCacheRef(KVCacheResourceV1& kvcache_resource) override;
    CacheLayerLayout                   layerCacheBase() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    int seqSizePerBlock() const override;

private:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          reuseCache(const CacheKeysType& cache_keys, KVCacheResourceV1& cache_resource);

private:
    std::shared_ptr<FullKVCacheGroup>                full_kv_cache_group_;
    std::vector<std::shared_ptr<LinearKVCacheGroup>> linear_kv_cache_groups_;

    std::vector<std::shared_ptr<KVCacheGroup>> all_kv_cache_groups_;
};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocator>;

}  // namespace rtp_llm
