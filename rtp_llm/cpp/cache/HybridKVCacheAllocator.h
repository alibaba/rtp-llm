#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"

namespace rtp_llm {

class HybridKVCacheAllocator: public KVCacheAllocator, public std::enable_shared_from_this<HybridKVCacheAllocator> {
public:
    HybridKVCacheAllocator(const CacheConfig&                 config,
                           AllocationType                     allocation_type     = AllocationType::DEVICE,
                           const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                           int64_t                            reserve_block_ratio = 0);

    void free(const FreeInfo& free_info) override;
    void insertIntoCache(const InsertInfo& insert_info) override;

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource&  kvcache_resource,
                                                    const CacheKeysByGroup& cache_keys_by_group,
                                                    bool                    is_connector = false) override;

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<GroupBlockIdPair>& block_update_mapping) override;

    int                      seqSizePerBlock() const override;
    int                      singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const override;
    std::vector<std::string> independentEvictionTags() const override;

protected:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override;
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override;
    int          getNeedBlocks(const MallocInfo& malloc_info) const override;
    int          estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                        int                    seq_len,
                                        int                    remaining_tokens,
                                        int                    reserve_step,
                                        bool                   enable_reuse_cache) const override;
    int          estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                    int  common_seq_len,
                                                    int  remaining_tokens,
                                                    int  reserve_step,
                                                    bool enable_reuse_cache,
                                                    int  target_batch_size) const override;
    void         checkCPShardedMallocResult(const MallocInfo& malloc_info) const override;
    void         decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override;

    int reuseTokens(BatchKVCacheResource& kv_resource, size_t max_reuse_tokens);

    virtual void
    referenceBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) const   = 0;
    virtual void freeBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) = 0;
    virtual bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    bool         skipReuseCacheGroup(std::string_view tag) const;
    bool         cpCompactSwaGroup(std::string_view tag, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void         rollbackBlockIdsToSize(std::string_view tag, BlockIds& block_ids, size_t original_size);
    void         rollbackInitMalloc(BatchKVCacheResource&                                    kv_resource,
                                    const std::unordered_map<std::string, BlockIndicesType>& referenced_blocks,
                                    const std::unordered_map<std::string, size_t>&           original_sizes);
    void         rollbackIncrMalloc(BatchKVCacheResource&                                       kv_resource,
                                    const std::vector<std::unordered_map<std::string, size_t>>& original_sizes,
                                    int                                                         failed_batch);
    virtual void copyBlockMappingForGroup(std::string_view                tag,
                                          const std::vector<BlockIdPair>& block_update_mapping) const;
    virtual MemoryType memoryTypeForGroup(std::string_view tag) const;

    std::unordered_map<std::string, KVCacheGroupPtr> kv_cache_groups_;
    std::vector<std::string>                         full_group_tags_;
    std::vector<std::string>                         linear_group_tags_;
    std::vector<std::string>                         swa_group_tags_;
};

using HybridKVCacheAllocatorPtr = std::shared_ptr<HybridKVCacheAllocator>;

}  // namespace rtp_llm
