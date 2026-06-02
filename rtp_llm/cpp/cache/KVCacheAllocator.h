#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class KVCacheAllocator {
public:
    KVCacheAllocator(const CacheConfig&                 config,
                     AllocationType                     allocation_type     = AllocationType::DEVICE,
                     const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                     int64_t                            reserve_block_ratio = 0):
        config_(config),
        allocation_type_(allocation_type),
        metrics_reporter_(metrics_reporter),
        reserve_block_ratio_(reserve_block_ratio) {}

    virtual ~KVCacheAllocator() = default;

    bool                           init();
    virtual void                   free(const FreeInfo& free_info)                        = 0;
    virtual void                   insertIntoCache(const InsertInfo& insert_info)         = 0;
    virtual BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const   = 0;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const = 0;
    virtual std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const = 0;
    virtual std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                            const CacheKeysType&   cache_keys,
                                                            bool                   is_connector = false)            = 0;

    virtual CacheLayerLayout allLayerCacheBase() const                                     = 0;
    virtual bool             updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           const std::vector<int>&        block_src_batch,
                                           bool                           copy_last_block,
                                           std::vector<BlockIdPair>&      block_update_mapping) = 0;
    virtual int              seqSizePerBlock() const                                       = 0;
    virtual int              singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const                 = 0;

    MallocResult malloc(const MallocInfo& malloc_info);
    void         blockCopy(int src_block_index, int dest_block_index);
    void         blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void         blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);
    void         blockBatchCopy(const torch::Tensor& copy_mapping);

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

    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    int64_t                 getMrCostTimeMs() const;
    size_t                  freeBlocksNum() const;
    size_t                  availableBlocksNum() const;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free);
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource);
    size_t                  requestRefBlocksNum() const;
    size_t                  connectorRefBlocksNum() const;
    size_t                  blockCacheRefBlocksNum() const;
    size_t                  notInUseBlocksNum() const;
    size_t                  availableTokensNum() const;
    size_t                  totalBlocksNum() const;
    size_t                  maxAvailableTokensNum() const;
    /// Returns global layer id; std::numeric_limits<uint32_t>::max() indicates invalid (caller must check).
    uint32_t convertToGlobalLayerId(size_t model_id, int local_layer_id) const;

protected:
    virtual bool         doInit() = 0;
    MallocResult         initMalloc(const MallocInfo& malloc_info);
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info)                                          = 0;
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info)                              = 0;
    virtual int          getNeedBlocks(const MallocInfo& malloc_info) const                                 = 0;
    virtual void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) = 0;

    CacheConfig                        config_;
    AllocationType                     allocation_type_;
    BlockPoolPtr                       block_pool_;
    const kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;

    size_t  reserve_block_num_{0};
    int64_t reserve_block_ratio_{0};
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
