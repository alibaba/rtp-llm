#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"

namespace rtp_llm {

// SingleTypedKVCacheAllocator is used for model with full attentions only
class SingleTypeKVCacheAllocator: public KVCacheAllocator {
public:
    SingleTypeKVCacheAllocator(const CacheConfig&   config,
                               rtp_llm::DeviceBase* device,
                               AllocationType       atype = AllocationType::DEVICE);

    bool               init() override;
    FreeResult         free(const FreeInfo& free_info) override;
    InsertResult       insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    CacheLayerLayout   layerCacheBase() const override;

    void regUserMr(size_t model_id) override;

    size_t freeBlocksNum() const override;
    size_t availableBlocksNum() const override;
    size_t totalBlocksNum() const override;
    size_t maxSeqLen() const override;

    KVCacheBuffer kvCacheBuffer() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

private:
    MallocResult incrMalloc(const MallocInfo& malloc_info);
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info);

private:
    BlockPoolPtr                      block_pool_;
    std::shared_ptr<FullKVCacheGroup> full_kv_cache_group_;
};

using SingleTypeKVCacheAllocatorPtr = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
