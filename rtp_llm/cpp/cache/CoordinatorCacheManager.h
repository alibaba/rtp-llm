#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/FullCacheManager.h"
#include "rtp_llm/cpp/cache/LinearCacheManager.h"
#include "rtp_llm/cpp/cache/SWACacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class CPSlotMapper;
struct KVCacheTokenCapacity {
    size_t total_tokens     = 0;
    size_t available_tokens = 0;
};

struct KVCachePoolMetricsSnapshot {
    std::string tag;
    std::string pool_name            = "unnamed";
    size_t      free_blocks          = 0;
    size_t      available_blocks     = 0;
    size_t      request_ref_blocks   = 0;
    size_t      connector_ref_blocks = 0;
    size_t      total_blocks         = 0;
    size_t      reserve_blocks       = 0;
    float       used_ratio           = 0.0f;
};

class CoordinatorCacheManager: public std::enable_shared_from_this<CoordinatorCacheManager> {
public:
    CoordinatorCacheManager(const CacheConfig&                 config,
                            AllocationType                     allocation_type     = AllocationType::DEVICE,
                            const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                            int64_t                            reserve_block_ratio = 0,
                            RoleType                           role_type           = RoleType::PDFUSION);

    virtual ~CoordinatorCacheManager() = default;

    bool                           init();
    virtual void                   free(const FreeInfo& free_info);
    virtual void                   insertIntoCache(const InsertInfo& insert_info);
    virtual BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;
    virtual std::vector<BlockInfo>
                          convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;
    virtual BlockAddrInfo convertIndexToAddr(int layer_id, std::string_view tag, int block_id) const;
    virtual std::vector<BlockInfo> convertIndexToBuffer(int layer_id, std::string_view tag, int block_id) const;
    virtual std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, std::string_view tag, int block_id, int partition_count, int partition_id) const;
    virtual std::shared_ptr<KVCacheResource>
    incrKVCacheRef(const KVCacheResource& kvcache_resource, const CacheKeysType& cache_keys, bool is_connector = false);

    virtual GroupedCacheLayerLayout allLayerCacheBase() const;
    virtual bool                    updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                  const std::vector<int>&        block_src_batch,

                                                  bool                            copy_last_block,
                                                  std::vector<TaggedBlockIdPair>& block_update_mapping);
    const CacheConfig&              cacheConfig() const {
        return config_;
    }
    virtual int seqSizePerBlock() const;
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
    void         blockBatchCopy(const std::vector<TaggedBlockIdPair>& copy_mapping);

    BlockPoolPtr blockPool(std::string_view tag) const;
    BlockPoolPtr soleGroupBlockPool() const;

    SharedBlockCachePtr sharedBlockCache() const {
        return shared_block_cache_;
    }

    void setSharedBlockCache(SharedBlockCachePtr shared_block_cache) {
        shared_block_cache_ = std::move(shared_block_cache);
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

    // Reserve some blocks for already-running streams' future allocations.
    // Only applied to "init malloc" requests where batch_kv_cache_resource has no blocks yet.
    void setReserveBlocksNum(size_t reserve_block_num) {
        reserve_block_num_ = reserve_block_num;
    }
    size_t reserveBlocksNum() const {
        return reserve_block_num_;
    }

    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    int64_t                 getMrCostTimeMs() const;
    size_t                  freeBlocksNum() const;
    virtual size_t          availableBlocksNum() const;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free);
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource);
    size_t                  requestRefBlocksNum() const;
    size_t                  connectorRefBlocksNum() const;
    size_t                  blockCacheRefBlocksNum() const;
    size_t                  notInUseBlocksNum() const;
    size_t                  availableTokensNum() const;
    size_t                  totalTokensNum() const;
    virtual size_t          totalBlocksNum() const;
    size_t                  maxAvailableTokensNum() const;
    KVCacheTokenCapacity    tokenCapacity(size_t default_seq_size_per_block) const;
    std::vector<KVCachePoolMetricsSnapshot> poolMetricsSnapshots() const;
    std::vector<std::string>                independentEvictionGroupTags() const;
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

    // Test-only seams retained for existing mocks and failure injection.
    virtual bool initSingleTypeManager(const SingleTypeCacheManagerPtr& manager);
    virtual bool
    shouldInjectGroupAllocationFailureForTest(const BatchKVCacheResource&, int, std::string_view, bool) const {
        return false;
    }

    virtual bool doInit();
    size_t       reservableAvailableBlocksNum() const;
    MallocResult initMalloc(const MallocInfo& malloc_info);
    // Classifies an init-malloc shortfall: a total-capacity shortfall is
    // PERMANENT (the request can never fit), an available-capacity shortfall is
    // RETRYABLE (the pools are momentarily full) so the stream stays WAITING
    // instead of being errored out under cache pressure.
    virtual MallocStatus
    evaluateInitCapacity(const MallocInfo& malloc_info, size_t reserve_blocks, InitCapacityMode mode) const;
    virtual MallocResult incrMalloc(const MallocInfo& malloc_info);
    virtual MallocResult initMallocForCommonLen(const MallocInfo& malloc_info);
    virtual int          getNeedBlocks(const MallocInfo& malloc_info) const;
    // Estimate peak additional blocks for one sequence resource.
    virtual int  estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                        int                    seq_len,
                                        int                    remaining_tokens,
                                        int                    reserve_step,
                                        bool                   enable_reuse_cache) const;
    virtual int  estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                    int  common_seq_len,
                                                    int  remaining_tokens,
                                                    int  reserve_step,
                                                    bool enable_reuse_cache,
                                                    int  target_batch_size) const;
    virtual void decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false);
    bool         cpShardThisGroupForCapacity(std::string_view tag) const;
    size_t       logicalSeqSizePerBlockForCapacity(std::string_view tag) const;
    int          cpEffectiveSeqLenForAlloc(std::string_view tag, int seq_len) const;
    int          deviceCacheMetricTokensPerBlock() const;

    const CacheConfig                  config_;
    AllocationType                     allocation_type_;
    SharedBlockCachePtr                shared_block_cache_;
    std::shared_ptr<CPSlotMapper>      cp_slot_mapper_;
    const kmonitor::MetricsReporterPtr metrics_reporter_           = nullptr;
    bool                               use_cuda_malloc_block_pool_ = false;

    size_t  reserve_block_num_{0};
    int64_t reserve_block_ratio_{0};

private:
    int  reuseCache(const CacheKeysType&                 cache_keys,
                    BatchKVCacheResource&                kv_resource,
                    const std::shared_ptr<CPSlotMapper>& cp_mapper);
    void referenceBlocks(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false) const;
    void freeBlocks(std::string_view tag, const BlockIndicesType& blocks, bool is_connector = false);
    void logMallocFailure(const MallocInfo& malloc_info,
                          const char*       phase,
                          int               failed_batch,
                          std::string_view  failed_tag,
                          bool              incremental,
                          int               failed_need_blocks) const;
    bool skipReuseCacheGroup(std::string_view tag) const;
    bool cpCompactSwaGroup(std::string_view tag, const std::shared_ptr<CPSlotMapper>& mapper) const;
    void rollbackBlockIdsToSize(std::string_view           tag,
                                BlockIds&                  block_ids,
                                size_t                     original_size,
                                const std::vector<size_t>& backfilled_positions);
    void rollbackInitMalloc(BatchKVCacheResource&                             kv_resource,
                            const std::map<std::string, BlockIndicesType>&    referenced_blocks,
                            const std::map<std::string, size_t>&              original_sizes,
                            const std::map<std::string, std::vector<size_t>>& backfilled_positions);
    void rollbackIncrMalloc(BatchKVCacheResource&                                          kv_resource,
                            const std::vector<std::map<std::string, size_t>>&              batch_original_sizes,
                            const std::vector<std::map<std::string, std::vector<size_t>>>& batch_backfilled_positions,
                            size_t                                                         last_touched_batch);
    void copyBlockMappingForGroup(std::string_view tag, const std::vector<BlockIdPair>& block_update_mapping) const;
    MemoryType memoryTypeForGroup(std::string_view tag) const;

    size_t                           storageIdxForTag(std::string_view tag) const;
    const SingleTypeCacheManagerPtr& singleTypeManager(std::string_view tag) const;
    const CacheGroup&                validateGroupForLayer(int layer_id, std::string_view tag) const;
    const CacheGroup&                defaultGroupForLayer(int layer_id) const;
    size_t                           minTokenCapacity(bool use_available_blocks, bool full_groups_only) const;
    size_t                           totalReservableAvailableBlocks() const;
    size_t
    reserveBlocksForPool(std::string_view tag, size_t reserve_blocks, size_t total_reservable_available_blocks) const;

    std::vector<BlockPoolPtr>               group_block_pools_;
    std::vector<SingleTypeCacheManagerPtr>  single_type_managers_;
    std::unordered_map<std::string, size_t> tag_to_idx_;
    std::vector<std::string>                full_group_tags_;
    std::vector<std::string>                linear_group_tags_;
    std::vector<std::string>                swa_group_tags_;
    RoleType                                role_type_{RoleType::PDFUSION};
};

using CoordinatorCacheManagerPtr = std::shared_ptr<CoordinatorCacheManager>;

}  // namespace rtp_llm
