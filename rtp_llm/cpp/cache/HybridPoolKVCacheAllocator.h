#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator:
    public KVCacheAllocator,
    public std::enable_shared_from_this<HybridPoolKVCacheAllocator> {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0);

    void                   free(const FreeInfo& free_info) override;
    void                   insertIntoCache(const InsertInfo& insert_info) override;
    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, KVCacheAttnType attn_type, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, KVCacheAttnType attn_type, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, KVCacheAttnType attn_type, int block_id, int partition_count, int partition_id) const override;

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;
    CacheLayerLayout                 allLayerCacheBase() const override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override;

    int seqSizePerBlock() const override;
    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                              int                            seq_len,
                              int                            reserve_step) const override;

    size_t                  freeBlocksNum() const override;
    size_t                  availableBlocksNum() const override;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free) override;
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override;
    size_t                  requestRefBlocksNum() const override;
    size_t                  connectorRefBlocksNum() const override;
    size_t                  blockCacheRefBlocksNum() const override;
    size_t                  notInUseBlocksNum() const override;
    size_t                  availableTokensNum() const override;
    size_t                  totalBlocksNum() const override;
    size_t                  maxAvailableTokensNum() const override;
    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t                 getMrCostTimeMs() const override;

private:
    bool         doInit() override;
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          getNeedBlocks(const MallocInfo& malloc_info) const override;
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

    int  groupIdForLayerAttn(int layer_id, KVCacheAttnType attn_type) const;
    int  defaultGroupIdForLayer(int layer_id) const;
    int  reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource);
    void referenceValidBlocks(int gid, const BlockIndicesType& blocks, bool is_connector = false) const;

private:
    std::vector<BlockPoolPtr>   group_block_pools_;
    std::vector<KVCacheGroupPtr> kv_cache_groups_;
    std::vector<int>             full_group_ids_;
    std::vector<int>             linear_group_ids_;
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
