#pragma once

#include <memory>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class KVCacheAllocator {
public:
    KVCacheAllocator(const CacheConfig&                 config,
                     rtp_llm::DeviceBase*               device,
                     AllocationType                     allocation_type  = AllocationType::DEVICE,
                     const kmonitor::MetricsReporterPtr metrics_reporter = nullptr):
        config_(config), device_(device), allocation_type_(allocation_type), metrics_reporter_(metrics_reporter) {}

    virtual ~KVCacheAllocator() = default;

    virtual bool                   init()                                                 = 0;
    virtual void                   free(const FreeInfo& free_info)                        = 0;
    virtual void                   insertIntoCache(const InsertInfo& insert_info)         = 0;
    virtual BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const   = 0;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const = 0;
    virtual std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const = 0;
    virtual std::shared_ptr<KVCacheResource> incrKVCacheRef(KVCacheResource&     kvcache_resource,
                                                            const CacheKeysType& cache_keys)      = 0;
    virtual void                             decrKVCacheRef(KVCacheResource& kvcache_resource)    = 0;

    virtual CacheLayerLayout allLayerCacheBase() const                                                           = 0;
    virtual bool             updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           const std::vector<int>&        block_src_batch,
                                           bool                           copy_last_block,
                                           std::vector<BlockIdPair>&      block_update_mapping)                       = 0;
    virtual int              seqSizePerBlock() const                                                             = 0;
    virtual int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len) const = 0;

    virtual std::vector<std::pair<BufferPtr, size_t>> getAllBuffers() const;
    MallocResult                                      malloc(const MallocInfo& malloc_info);
    void                                              blockCopy(int src_block_index, int dest_block_index);
    void                                              blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping);

    BlockPoolPtr getBlockPool() const {
        return block_pool_;
    }

    // Reserve some blocks for already-running streams' future allocations.
    // Only applied to "init malloc" requests where batch_kv_cache_resource has no blocks yet.
    void setReserveBlockNum(size_t reserve_block_num) {
        reserve_block_num_ = reserve_block_num;
    }
    size_t reserveBlockNum() const {
        return reserve_block_num_;
    }

    void          regUserMr(size_t model_id);
    int64_t       getMrCostTimeMs() const;
    size_t        freeBlocksNum() const;
    size_t        availableBlocksNum() const;
    size_t        availableTokensNum() const;
    size_t        totalBlocksNum() const;
    size_t        maxAvailableTokensNum() const;
    KVCacheBuffer kvCacheBuffer() const;
    KVCacheBuffer getMTPModuleKVCacheBuffer(int mtp_module_id) const;

protected:
    MallocResult         initMalloc(const MallocInfo& malloc_info);
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info)             = 0;
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) = 0;
    virtual int          getNeedBlocks(const MallocInfo& malloc_info) const    = 0;

    CacheConfig                        config_;
    rtp_llm::DeviceBase*               device_;
    AllocationType                     allocation_type_;
    BlockPoolPtr                       block_pool_;
    const kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;

    size_t reserve_block_num_{0};
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
