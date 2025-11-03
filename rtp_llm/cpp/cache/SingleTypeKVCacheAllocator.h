#pragma once

#include <memory>
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"

namespace rtp_llm {

// SingleTypedKVCacheAllocator is used for model with full attentions only
class SingleTypeKVCacheAllocator:
    public KVCacheAllocator,
    public std::enable_shared_from_this<SingleTypeKVCacheAllocator> {
public:
    SingleTypeKVCacheAllocator(const CacheConfig&                 config,
                               rtp_llm::DeviceBase*               device,
                               AllocationType                     allocation_type  = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    bool                   init() override;
    void                   free(const FreeInfo& free_info) override;
    void                   insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;
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
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

private:
    std::shared_ptr<FullKVCacheGroup> full_kv_cache_group_;
};

using SingleTypeKVCacheAllocatorPtr = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
