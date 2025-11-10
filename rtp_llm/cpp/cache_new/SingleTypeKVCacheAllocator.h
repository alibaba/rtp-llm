#pragma once

#include <memory>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"

namespace rtp_llm {

// SingleTypedKVCacheAllocator is used for model with full attentions only
class SingleTypeKVCacheAllocator: public KVCacheAllocator {
public:
    SingleTypeKVCacheAllocator(const CacheConfig&   config,
                               rtp_llm::DeviceBase* device,
                               AllocationType       allocation_type = AllocationType::DEVICE);

    bool               init() override;
    void               free(const FreeInfo& free_info) override;
    void               insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    CacheLayerLayout   layerCacheBase() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    int  seqSizePerBlock() const override;
    void clearCache() override;

private:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;

private:
    std::shared_ptr<FullKVCacheGroup> full_kv_cache_group_;
};

using SingleTypeKVCacheAllocatorPtr = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
