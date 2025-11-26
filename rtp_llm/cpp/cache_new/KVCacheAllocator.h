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
    KVCacheAllocator(const CacheConfig&   config,
                     rtp_llm::DeviceBase* device,
                     AllocationType       atype = AllocationType::DEVICE):
        config_(config), device_(device), atype_(atype) {}

    virtual ~KVCacheAllocator() = default;

    virtual bool               init()                                                 = 0;
    virtual FreeResult         free(const FreeInfo& free_info)                        = 0;
    virtual InsertResult       insertIntoCache(const InsertInfo& insert_info)         = 0;
    virtual BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const   = 0;
    virtual BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const = 0;
    virtual CacheLayerLayout   layerCacheBase() const                                 = 0;

    virtual void regUserMr(size_t model_id) = 0;

    virtual size_t freeBlocksNum() const      = 0;
    virtual size_t availableBlocksNum() const = 0;
    virtual size_t availableTokensNum() const = 0;
    virtual size_t totalBlocksNum() const     = 0;
    virtual size_t maxSeqLen() const          = 0;

    MallocResult malloc(const MallocInfo& malloc_info);
    void         blockCopy(int src_block_index, int dest_block_index);
    void         blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void         blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);
    void         blockBatchCopy(const rtp_llm::Buffer& copy_mapping);

    virtual KVCacheBuffer kvCacheBuffer() const = 0;

    virtual bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                               const std::vector<int>&        block_src_batch,
                               bool                           copy_last_block,
                               std::vector<BlockIdPair>&      block_update_mapping) = 0;

protected:
    MallocResult         initMalloc(const MallocInfo& malloc_info);
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info)             = 0;
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) = 0;

    CacheConfig          config_;
    rtp_llm::DeviceBase* device_;
    AllocationType       atype_;
    // std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
