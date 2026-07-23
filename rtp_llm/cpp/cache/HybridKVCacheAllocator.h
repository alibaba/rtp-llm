#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
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

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override;

    bool updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                       const std::vector<int>&         block_src_batch,
                       bool                            copy_last_block,
                       std::vector<TaggedBlockIdPair>& block_update_mapping) override;

    int                      seqSizePerBlock() const override;
    int                      singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const override;
    std::vector<std::string> independentEvictionGroupTags() const override;

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

    int reuseCache(const CacheKeysType&                 cache_keys,
                   BatchKVCacheResource&                kv_resource,
                   const std::shared_ptr<CPSlotMapper>& cp_mapper);

    virtual void
    referenceBlocksInGroup(int group_index, const BlockIndicesType& blocks, bool is_connector = false) const   = 0;
    virtual void freeBlocksInGroup(int group_index, const BlockIndicesType& blocks, bool is_connector = false) = 0;
    virtual bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    bool         skipReuseCacheGroup(int group_index) const;
    bool         cpCompactSwaGroup(int group_index, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void         rollbackBlockIdsToSize(int group_index, BlockIds& block_ids, size_t original_size);
    void         rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                    const std::vector<BlockIndicesType>& referenced_blocks,
                                    const std::vector<size_t>&           original_sizes);
    void         rollbackIncrMalloc(BatchKVCacheResource&                   kv_resource,
                                    const std::vector<std::vector<size_t>>& original_sizes,
                                    int                                     failed_batch);
    virtual void copyBlockMappingForGroup(int group_index, const std::vector<BlockIdPair>& block_update_mapping) const;
    virtual MemoryType memoryTypeForGroup(int group_index) const;
    const std::string& groupTag(int group_index) const {
        RTP_LLM_CHECK_WITH_INFO(group_index >= 0 && static_cast<size_t>(group_index) < kv_cache_groups_.size(),
                                "invalid group_index=%d",
                                group_index);
        return kv_cache_groups_[static_cast<size_t>(group_index)]->tag();
    }
    size_t groupIndex(std::string_view tag) const {
        const size_t group_index = config_.topology().groupIndex(tag);
        RTP_LLM_CHECK_WITH_INFO(group_index < kv_cache_groups_.size(),
                                "cache group index=%zu out of range=%zu for tag=%s",
                                group_index,
                                kv_cache_groups_.size(),
                                std::string(tag).c_str());
        return group_index;
    }
    const KVCacheGroupPtr& cacheGroup(std::string_view tag) const {
        return kv_cache_groups_.at(groupIndex(tag));
    }

    std::vector<KVCacheGroupPtr> kv_cache_groups_;
    std::vector<int>             full_group_indices_;
    std::vector<int>             linear_group_indices_;
    std::vector<int>             swa_group_indices_;
};

using HybridKVCacheAllocatorPtr = std::shared_ptr<HybridKVCacheAllocator>;

}  // namespace rtp_llm
