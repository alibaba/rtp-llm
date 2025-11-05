#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"

namespace rtp_llm {



class KVCacheAllocator {
public:
    KVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE)
        : config_(config), device_(device), atype_(atype) {}
    
    virtual ~KVCacheAllocator() = default;

    virtual bool init() = 0;
    virtual MallocResult malloc(const MallocInfo& malloc_info) = 0;
    virtual FreeResult free(const FreeInfo& free_info) = 0;
    virtual InsertResult insertIntoCache(const InsertInfo& insert_info) = 0;
    virtual BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const = 0;
    virtual BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const = 0;
    virtual CacheLayerLayout layerCacheBase() const = 0;

    virtual size_t freeBlocksNums() const = 0;
    virtual size_t availableBlocksNums() const = 0;
    virtual size_t totalBlocksNums() const = 0;
    virtual size_t maxSeqLen() const = 0;

    virtual void blockCopy(int src_block_index, int dest_block_index) = 0;
    virtual void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) = 0;
    virtual void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) = 0;
    virtual void blockBatchCopy(const rtp_llm::Buffer& copy_mapping) = 0;

protected:
    CacheConfig config_;
    rtp_llm::DeviceBase* device_;
    AllocationType atype_;
    // std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
