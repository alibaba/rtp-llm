#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"

namespace rtp_llm {


// SingleTypedKVCacheAllocator is used for model with full attentions only
class SingleTypeKVCacheAllocator : public KVCacheAllocator {
public: 
    SingleTypeKVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);
    
    bool init() override;
    MallocResult malloc(const MallocInfo& malloc_info) override;
    FreeResult free(const FreeInfo& free_info) override;
    InsertResult insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    CacheLayerLayout layerCacheBase() const override;

    size_t freeBlocksNums() const override;
    size_t availableBlocksNums() const override;
    size_t totalBlocksNums() const override;
    size_t maxSeqLen() const override;

    void blockCopy(int src_block_index, int dest_block_index) override;    
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) override;
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) override;
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping) override;
    
private:
    BlockPoolPtr block_pool_;
    std::shared_ptr<FullKVCacheGroup> full_kv_cache_group_;
};

using SingleTypeKVCacheAllocatorPtr = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
