#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class CPSlotMapper;
class BlockTreeCache;
using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;
class KVCacheGroup;
using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;
struct KVCacheTokenCapacity {
    size_t total_tokens     = 0;
    size_t available_tokens = 0;
};

struct KVCachePoolMetricsSnapshot {
    size_t      pool_index                = 0;
    std::string pool_name                 = "unnamed";
    size_t      block_size_bytes          = 0;
    size_t      free_blocks               = 0;
    size_t      used_blocks               = 0;
    size_t      active_blocks             = 0;
    size_t      total_blocks              = 0;
    size_t      reserve_blocks            = 0;
    size_t      request_ref_blocks        = 0;
    size_t      block_cache_ref_blocks    = 0;
    size_t      load_ref_blocks           = 0;
    size_t      eviction_ref_blocks       = 0;
    size_t      store_ref_blocks          = 0;
    float       used_ratio                = 0.0f;
};

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
    virtual BlockAddrInfo          convertIndexToAddr(int layer_id, int group_id, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int group_id, int block_id) const;
    virtual std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int group_id, int block_id, int partition_count, int partition_id) const;
    virtual BlockAddrInfo          convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBufferByTag(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const;
    virtual std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                            const CacheKeysType&   cache_keys,
                                                            bool                   is_connector = false) = 0;

    virtual GroupedCacheLayerLayout allLayerCacheBase() const                                           = 0;
    virtual bool                    updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                                  const std::vector<int>&         block_src_batch,
                                                  bool                            copy_last_block,
                                                  std::vector<TaggedBlockIdPair>& block_update_mapping) = 0;
    virtual int                     seqSizePerBlock() const                                             = 0;
    virtual int                     singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                          int                            seq_len,
                                                          int                            reserve_step) const                       = 0;
    // Common-prefix growth is charged once; non-common growth is charged once per target sequence.
    int estimateBatchPeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                    int                            seq_len,
                                    int                            common_seq_len,
                                    int                            remaining_tokens,
                                    int                            reserve_step,
                                    bool                           enable_reuse_cache,
                                    int                            target_batch_size) const;

    MallocResult malloc(const MallocInfo& malloc_info);
    bool         abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    virtual void blockCopy(int src_block_index, int dest_block_index);
    virtual void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    virtual void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);
    virtual void blockBatchCopy(const torch::Tensor& copy_mapping);
    virtual void blockBatchCopyByTag(const std::vector<TaggedBlockIdPair>& copy_mapping);

    DeviceBlockPoolPtr getDeviceBlockPool() const {
        return block_pool_;
    }

    virtual const std::vector<DeviceBlockPoolPtr>& groupBlockPools() const {
        static const std::vector<DeviceBlockPoolPtr> empty;
        return empty;
    }

    virtual std::vector<KVCacheGroupPtr> cacheGroups() const {
        return {};
    }

    void attachBlockTreeCache(BlockTreeCachePtr block_tree_cache);

    BlockTreeCachePtr blockTreeCache() const {
        return block_tree_cache_;
    }

    void setUseCudaMallocBlockPool(bool use_cuda_malloc_block_pool) {
        use_cuda_malloc_block_pool_ = use_cuda_malloc_block_pool;
    }

    void setCPSlotMapper(std::shared_ptr<CPSlotMapper> cp_slot_mapper) {
        cp_slot_mapper_ = std::move(cp_slot_mapper);
    }

    std::shared_ptr<CPSlotMapper> cpSlotMapper() const {
        return cp_slot_mapper_;
    }

    void setReserveBlocksNum(size_t reserve_block_num) {
        reserve_block_num_ = reserve_block_num;
    }

    size_t reserveBlocksNum() const {
        return reserve_block_num_;
    }

    virtual void                 regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    virtual int64_t              getMrCostTimeMs() const;
    virtual size_t               freeBlocksNum() const;
    virtual size_t               activeTreeCachedBlocksNum() const;
    virtual size_t               availableBlocksNum() const;
    virtual size_t               availableTokensNum() const;
    virtual size_t               totalTokensNum() const;
    virtual size_t               totalBlocksNum() const;
    virtual size_t               maxAvailableTokensNum() const;
    virtual KVCacheTokenCapacity tokenCapacity(size_t default_seq_size_per_block) const;
    virtual std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const;
    /// Returns global layer id; std::numeric_limits<uint32_t>::max() indicates invalid (caller must check).
    uint32_t convertToGlobalLayerId(size_t model_id, int local_layer_id) const;

protected:
    // Which capacity snapshots evaluateInitCapacity() is allowed to consult.
    // TOTAL_ONLY answers "can this request ever fit"; TOTAL_AND_AVAILABLE also
    // answers "can it fit right now".
    enum class InitCapacityMode {
        TOTAL_ONLY,
        TOTAL_AND_AVAILABLE,
    };

    virtual bool         doInit() = 0;
    virtual size_t       reserveBlocksForPoolMetrics(size_t pool_index) const;
    virtual size_t       reservableFreeBlocksNum() const;
    MallocResult         initMalloc(const MallocInfo& malloc_info);
    // Classifies an init-malloc shortfall: a total-capacity shortfall is
    // PERMANENT (the request can never fit), an available-capacity shortfall is
    // RETRYABLE (the pools are momentarily full) so the stream stays WAITING
    // instead of being errored out under cache pressure.
    virtual MallocStatus
    evaluateInitCapacity(const MallocInfo& malloc_info, size_t reserve_blocks, InitCapacityMode mode) const;
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info)             = 0;
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) = 0;
    virtual int          getNeedBlocks(const MallocInfo& malloc_info) const    = 0;
    struct InitBlockDemand {
        // Added to the planner's result when checking the request's complete
        // footprint against pool total.
        size_t retained_blocks{0};
        // Compared with currently available capacity.
        size_t additional_blocks{0};
    };
    // Count unique valid physical blocks already held by this request. A
    // negative group_id counts all groups that share the allocator's pool.
    static size_t heldRequestBlocks(const MallocInfo& malloc_info, int group_id = -1);
    // Reuse-aware interpretation of planner output: reuse planners report
    // additional demand; no-reuse planners report the full footprint.
    static InitBlockDemand initBlockDemand(const MallocInfo& malloc_info,
                                           size_t            planned_blocks,
                                           int               group_id = -1);
    // Estimate peak additional blocks for one sequence resource.
    virtual int   estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                         int                    seq_len,
                                         int                    remaining_tokens,
                                         int                    reserve_step,
                                         bool                   enable_reuse_cache) const           = 0;
    virtual int   estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                     int  common_seq_len,
                                                     int  remaining_tokens,
                                                     int  reserve_step,
                                                     bool enable_reuse_cache,
                                                     int  target_batch_size) const = 0;
    virtual void  checkCPShardedMallocResult(const MallocInfo&) const {}
    virtual void  decrKVCacheRef(const KVCacheResource& kvcache_resource) = 0;
    bool          cpShardThisGroupForCapacity(size_t gid) const;
    size_t        logicalSeqSizePerBlockForCapacity(size_t gid) const;
    int           cpEffectiveSeqLenForAlloc(size_t gid, int seq_len) const;
    int           deviceCacheMetricTokensPerBlock() const;
    static size_t maxReusableMatchKeys(int seq_len, int reuse_unit_tokens) {
        if (seq_len <= 1 || reuse_unit_tokens <= 0) {
            return 0;
        }
        return static_cast<size_t>(seq_len - 1) / static_cast<size_t>(reuse_unit_tokens);
    }

    CacheConfig                        config_;
    AllocationType                     allocation_type_;
    DeviceBlockPoolPtr                 block_pool_;
    BlockTreeCachePtr                  block_tree_cache_;
    std::shared_ptr<CPSlotMapper>      cp_slot_mapper_;
    const kmonitor::MetricsReporterPtr metrics_reporter_           = nullptr;
    bool                               use_cuda_malloc_block_pool_ = false;

    size_t  reserve_block_num_{0};
    int64_t reserve_block_ratio_{0};
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
