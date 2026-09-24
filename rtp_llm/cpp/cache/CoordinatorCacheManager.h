#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/FullCacheManager.h"
#include "rtp_llm/cpp/cache/LinearCacheManager.h"
#include "rtp_llm/cpp/cache/SWACacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class CPSlotMapper;
class LoadAsyncContext;
class BlockTreeCache;
using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;
class SingleTypeCacheManager;
using SingleTypeCacheManagerPtr = std::shared_ptr<SingleTypeCacheManager>;
struct KVCacheTokenCapacity {
    size_t total_tokens     = 0;
    size_t available_tokens = 0;
};

struct KVCachePoolMetricsSnapshot {
    size_t      pool_index                 = 0;
    std::string pool_name                  = "unnamed";
    size_t      block_size_bytes           = 0;
    size_t      free_blocks                = 0;
    size_t      used_blocks                = 0;
    size_t      active_blocks              = 0;
    size_t      available_blocks           = 0;
    size_t      total_blocks               = 0;
    size_t      reserve_blocks             = 0;
    size_t      request_ref_blocks         = 0;
    size_t      block_cache_ref_blocks     = 0;
    size_t      load_ref_blocks            = 0;
    size_t      eviction_target_ref_blocks = 0;
    size_t      store_ref_blocks           = 0;
    float       used_ratio                 = 0.0f;
};

class CoordinatorCacheManager: public std::enable_shared_from_this<CoordinatorCacheManager> {
public:
    CoordinatorCacheManager(const CacheConfig&                 config,
                            AllocationType                     allocation_type     = AllocationType::DEVICE,
                            const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                            int64_t                            reserve_block_ratio = 0,
                            RoleType                           role_type           = RoleType::PDFUSION):
        config_(config),
        allocation_type_(allocation_type),
        metrics_reporter_(metrics_reporter),
        reserve_block_ratio_(reserve_block_ratio),
        role_type_(role_type) {}

    virtual ~CoordinatorCacheManager() = default;

    bool                           init();
    virtual void                   free(const FreeInfo& free_info);
    virtual void                   insertIntoCache(const InsertInfo& insert_info, size_t& resident_prefix_length);
    virtual BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;
    virtual std::vector<BlockInfo>
                          convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;
    virtual BlockAddrInfo convertIndexToAddr(int layer_id, const std::string& group_tag, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& group_tag, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& group_tag, int block_id, int partition_count, int partition_id) const;
    virtual std::shared_ptr<KVCacheResource>
    incrKVCacheRef(const KVCacheResource& kvcache_resource, const CacheKeysType& cache_keys, bool is_connector = false);
    virtual GroupedCacheLayerLayout allLayerCacheBase() const;
    virtual bool                    updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                                  const std::vector<int>&         block_src_batch,
                                                  bool                            copy_last_block,
                                                  std::vector<TaggedBlockIdPair>& block_update_mapping);
    virtual int                     seqSizePerBlock() const;
    virtual int
    singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len, int reserve_step) const;
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
    virtual void blockBatchCopyByGroup(const std::vector<TaggedBlockIdPair>& copy_mapping);

    virtual const std::vector<DeviceBlockPoolPtr>& groupBlockPools() const {
        return group_block_pools_;
    }

    virtual std::vector<SingleTypeCacheManagerPtr> cacheGroups() const {
        return kv_cache_groups_;
    }

    void attachBlockTreeCache(BlockTreeCachePtr block_tree_cache);

    BlockTreeCachePtr blockTreeCache() const {
        return block_tree_cache_;
    }

    void setUseDeviceMallocBlockPool(bool use_device_malloc_block_pool) {
        use_device_malloc_block_pool_ = use_device_malloc_block_pool;
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

    virtual bool   doInit();
    virtual size_t reserveBlocksForPoolMetrics(size_t pool_index) const;
    virtual size_t reservableFreeBlocksNum() const;
    MallocResult   initMalloc(const MallocInfo& malloc_info);
    // Classifies an init-malloc shortfall: a total-capacity shortfall is
    // PERMANENT (the request can never fit), an available-capacity shortfall is
    // RETRYABLE (the pools are momentarily full) so the stream stays WAITING
    // instead of being errored out under cache pressure.
    virtual MallocStatus
    evaluateInitCapacity(const MallocInfo& malloc_info, size_t reserve_blocks, InitCapacityMode mode) const;
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info);
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info);
    virtual int          getNeedBlocks(const MallocInfo& malloc_info) const;
    struct InitBlockDemand {
        // Added to the planner's result when checking the request's complete
        // footprint against pool total.
        size_t retained_blocks{0};
        // Compared with currently available capacity.
        size_t additional_blocks{0};
    };
    // Count unique valid physical blocks held by this request in one independent pool.
    static size_t heldRequestBlocks(const MallocInfo& malloc_info, std::string_view tag);
    // Reuse-aware interpretation of planner output: reuse planners report
    // additional demand; no-reuse planners report the full footprint.
    static InitBlockDemand initBlockDemand(const MallocInfo& malloc_info, size_t planned_blocks, std::string_view tag);
    // Estimate peak additional blocks for one sequence resource.
    virtual int   estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                         int                    seq_len,
                                         int                    remaining_tokens,
                                         int                    reserve_step,
                                         bool                   enable_reuse_cache) const;
    virtual int   estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                     int  common_seq_len,
                                                     int  remaining_tokens,
                                                     int  reserve_step,
                                                     bool enable_reuse_cache,
                                                     int  target_batch_size) const;
    virtual void  checkCPShardedMallocResult(const MallocInfo&) const;
    virtual void  decrKVCacheRef(const KVCacheResource& kvcache_resource);
    size_t        logicalSeqSizePerBlockForCapacity(const std::string& tag) const;
    int           deviceCacheMetricTokensPerBlock() const;
    static size_t maxReusableMatchKeys(int seq_len, int reuse_unit_tokens) {
        if (seq_len <= 1 || reuse_unit_tokens <= 0) {
            return 0;
        }
        return static_cast<size_t>(seq_len - 1) / static_cast<size_t>(reuse_unit_tokens);
    }

    // Own the immutable configuration binding for all dense manager/pool rows.
    const CacheConfig                  config_;
    AllocationType                     allocation_type_;
    BlockTreeCachePtr                  block_tree_cache_;
    std::shared_ptr<CPSlotMapper>      cp_slot_mapper_;
    const kmonitor::MetricsReporterPtr metrics_reporter_             = nullptr;
    bool                               use_device_malloc_block_pool_ = false;

    size_t  reserve_block_num_{0};
    int64_t reserve_block_ratio_{0};

    // One allocation spans capacity preflight, optional cache matching and one
    // or more BlockPool allocations.  BlockPool makes each individual
    // operation thread-safe, but without this transaction lock concurrent
    // init-malloc callers can all pass the same reserve check and collectively
    // consume the forward-progress reserve before any one of them allocates.
    std::mutex malloc_mutex_;

    struct PreparedKVCache {
        size_t                         matched_device_blocks = 0;
        size_t                         total_logical_blocks  = 0;
        std::vector<RequiredPositions> required_positions;
        std::vector<BlockIndicesType>  referenced_blocks;
        std::vector<size_t>            original_sizes;
        MallocStatus                   materialize_status = MallocStatus::NONE;
    };

    std::shared_ptr<LoadAsyncContext> prepareKVCache(const CacheKeysType&                 cache_keys,
                                                     BatchKVCacheResource&                kv_resource,
                                                     const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                                     PreparedKVCache&                     prepared);
    bool                              materializeInitialBlocks(const MallocInfo& malloc_info,
                                                               PreparedKVCache&  prepared,
                                                               LoadAsyncContext* context,
                                                               size_t            matched_blocks);
    bool                              finishDeferredMalloc(const MallocInfo& malloc_info,
                                                           PreparedKVCache&  prepared,
                                                           LoadAsyncContext& context,
                                                           size_t            matched_blocks);

    virtual MallocStatus evaluatePreparedInitCapacity(const MallocInfo&      malloc_info,
                                                      size_t                 reserve_blocks,
                                                      const PreparedKVCache& prepared,
                                                      bool                   has_load_context) const;
    virtual bool         hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const;
    virtual void         logMallocFailure(const MallocInfo& malloc_info,
                                          const char*       phase,
                                          int               failed_batch,
                                          int               failed_group,
                                          bool              incremental,
                                          int               failed_need_blocks) const;
    size_t               loadTargetPosition(size_t                               path_index,
                                            const std::string&                   tag,
                                            const std::shared_ptr<CPSlotMapper>& mapper,
                                            int                                  cp_scale) const;
    bool                 cpCompactSwaGroup(const std::string& tag, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void                 rollbackBlockIdsToSize(int group_id, BlockIds& block_ids, size_t original_size);
    void                 rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                            const std::vector<BlockIndicesType>& referenced_blocks,
                                            const std::vector<size_t>&           original_sizes);
    virtual MemoryType   memoryTypeForGroup(int group_id) const;

    std::vector<SingleTypeCacheManagerPtr> kv_cache_groups_;
    std::vector<int>                       full_group_ids_;
    std::vector<int>                       linear_group_ids_;
    std::vector<int>                       swa_group_ids_;
    MallocStatus                           evaluateInitCapacityImpl(const MallocInfo&                     malloc_info,
                                                                    size_t                                reserve_blocks,
                                                                    InitCapacityMode                      mode,
                                                                    const std::vector<RequiredPositions>* required_positions) const;
    int                                    validateGroupIdForLayer(int layer_id, int group_id) const;
    int                                    defaultGroupIdForLayer(int layer_id) const;
    size_t                                 minTokenCapacity(bool use_available_blocks, bool full_groups_only) const;
    size_t                                 totalReservableFreeBlocks() const;
    size_t                                 reserveBlocksForPool(size_t group_id) const;
    std::vector<DeviceBlockPoolPtr>        group_block_pools_;
    RoleType                               role_type_{RoleType::PDFUSION};

private:
    size_t groupIdForTag(std::string_view tag) const;
};

using CoordinatorCacheManagerPtr = std::shared_ptr<CoordinatorCacheManager>;

}  // namespace rtp_llm
