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

    bool             init() override;
    MallocResult     malloc(const MallocInfo& malloc_info) override;
    FreeResult       free(const FreeInfo& free_info) override;
    InsertResult     insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo    convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo  convertIndexToBuffer(int layer_id, int block_id) const override;
    CacheLayerLayout layerCacheBase() const override;

    void regUserMr(size_t model_id) override;

    size_t freeBlocksNums() const override;
    size_t availableBlocksNums() const override;
    size_t totalBlocksNums() const override;
    size_t maxSeqLen() const override;

    KVCacheBuffer kvCacheBuffer() const override;

private:
    int reuseCache(const CacheKeysType& cache_keys, GroupBlockIds& group_block_ids);

    BlockPoolPtr                                     block_pool_;
    std::shared_ptr<FullKVCacheGroup>                full_kv_cache_group_;
    std::vector<std::shared_ptr<LinearKVCacheGroup>> linear_kv_cache_groups_;

    std::vector<std::shared_ptr<KVCacheGroup>> all_kv_cache_groups_;
};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocator>;

}  // namespace rtp_llm
