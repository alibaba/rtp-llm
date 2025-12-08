#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"

namespace rtp_llm {

class HybridLayerKVCacheAllocator: public KVCacheAllocator {
public:
    HybridLayerKVCacheAllocator(const CacheConfig&   config,
                                rtp_llm::DeviceBase* device,
                                AllocationType       atype = AllocationType::DEVICE);

    bool               init() override;
    void               free(const FreeInfo& free_info) override;
    InsertResult       insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    CacheLayerLayout   layerCacheBase() const override;

    void regUserMr(size_t model_id) override;

    size_t freeBlocksNum() const override;
    size_t availableBlocksNum() const override;
    size_t availableTokensNum() const override;
    size_t totalBlocksNum() const override;
    size_t maxSeqLen() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    KVCacheBuffer kvCacheBuffer() const override;

    std::vector<std::pair<rtp_llm::BufferPtr, size_t>> getAllBuffers() const override;

    BlockPoolPtr getBlockPool() const {
        return block_pool_;
    }

    // TODO, friend class test
public:
    MallocResult initMalloc(const MallocInfo& malloc_info);
    MallocResult incrMalloc(const MallocInfo& malloc_info);
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info);

    int reuseCache(const CacheKeysType& cache_keys, KVCacheResourceV1& cache_resource);

    BlockPoolPtr                                     block_pool_;
    std::shared_ptr<FullKVCacheGroup>                full_kv_cache_group_;
    std::vector<std::shared_ptr<LinearKVCacheGroup>> linear_kv_cache_groups_;

    std::vector<std::shared_ptr<KVCacheGroup>> all_kv_cache_groups_;
};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocator>;

}  // namespace rtp_llm
