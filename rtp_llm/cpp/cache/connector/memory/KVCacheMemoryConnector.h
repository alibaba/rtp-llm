#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "autil/LockFreeThreadPool.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/PrefixTreeMemoryBlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class BlockPool;
class BroadcastManager;
class KVCacheAllocator;
class MemoryAsyncContext;

class KVCacheMemoryConnector: public KVCacheConnector {
public:
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const KVCacheConfig&                     kv_cache_config,
                           const ParallelismConfig&                 parallelism_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           const std::vector<std::string>&          tp_addrs,
                           const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr);
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const KVCacheConfig&                     kv_cache_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           const std::vector<std::string>&          tp_addrs,
                           const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr);
    ~KVCacheMemoryConnector() override;

public:
    bool init();

    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context,
                                                 int                                       start_read_block_index,
                                                 int                                       read_block_num) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override {
        RTP_LLM_FAIL("KVCacheMemoryConnector asyncWriteByLayer is not implemented");
    }

    // virtual for test
    virtual bool              copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response);
    std::vector<CacheKeyType> cacheKeys() const;

private:
    struct LayerTagSlot {
        int         layer_id{-1};
        std::string tag;
        int         group_id{-1};
        size_t      stride_bytes{0};
    };
    struct CopyInfoPerKey {
        CacheKeyType              cache_key{0};
        CacheBlockKind            kind{CacheBlockKind::COMPLETE};
        CacheBackingType          backing_type{CacheBackingType::MEMORY};
        BlockIdxType              mem_block{NULL_BLOCK_IDX};
        BlockIdxType              src_mem_block{NULL_BLOCK_IDX};
        CacheBackingType          src_backing_type{CacheBackingType::MEMORY};
        int32_t                   src_disk_slot{-1};
        int32_t                   disk_slot{-1};
        size_t                    block_size{0};
        std::vector<BlockIdxType> gpu_blocks;
        std::vector<uint8_t>      slot_valid_mask;
        bool                      is_complete{true};
        bool                      request_released{false};
        uint64_t                  generation{0};
        uint64_t                  src_generation{0};
    };
    enum class CopyDirection {
        H2D = 0,
        D2H = 1
    };
    struct CopyPlan {
        std::vector<CopyInfoPerKey> copy_infos;
        CopyDirection               direction;
    };

    std::shared_ptr<CopyPlan> buildCopyPlanForRead(const CacheKeysType&             cache_keys,
                                                   const LayerAttnBlockIds&         layer_attn_block_ids,
                                                   const std::vector<LayerTagSlot>& slots,
                                                   int                              start_index,
                                                   int                              read_num);
    std::shared_ptr<CopyPlan> buildCopyPlanForRead(const CacheKeysType& cache_keys,
                                                   const LayerBlockIds& layer_block_ids,
                                                   int                  start_index,
                                                   int                  read_num);
    std::shared_ptr<CopyPlan> buildCopyPlanForWrite(const CacheKeysType&             cache_keys,
                                                    const LayerAttnBlockIds&         layer_attn_block_ids,
                                                    const std::vector<LayerTagSlot>& slots,
                                                    int                              start_index,
                                                    int                              write_num,
                                                    bool&                            no_need_write);
    std::shared_ptr<CopyPlan> buildCopyPlanForWrite(const CacheKeysType& cache_keys,
                                                    const LayerBlockIds& layer_block_ids,
                                                    int                  start_index,
                                                    int                  write_num,
                                                    bool&                no_need_write);
    std::shared_ptr<CopyPlan> createCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos,
                                             const CopyDirection&               direction);
    bool startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context, const std::shared_ptr<CopyPlan>& copy_plan);
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
    sendCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
         sendMemoryRequest(const MemoryOperationRequestPB& mem_req, int64_t timeout_ms) const;
    void printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;

    bool prepareCopyBuffers(BlockIdxType                     mem_block,
                            const std::vector<BlockIdxType>& gpu_blocks,
                            CopyDirection                    direction,
                            bool                             is_complete,
                            std::vector<torch::Tensor>&      dst,
                            std::vector<torch::Tensor>&      src);
    bool prepareLayerCopyBuffers(BlockIdxType                     mem_block,
                                 const std::vector<BlockIdxType>& gpu_blocks,
                                 CopyDirection                    direction,
                                 std::vector<torch::Tensor>&      dst,
                                 std::vector<torch::Tensor>&      src);
    bool appendCopyBytesToBuffers(const BlockInfo&            mem_block,
                                  const BlockInfo&            gpu_block,
                                  size_t                      byte_off,
                                  CopyDirection               direction,
                                  std::vector<torch::Tensor>& dst,
                                  std::vector<torch::Tensor>& src);
    bool copyDiskItems(const MemoryOperationRequestPB&  request,
                       CopyDirection                    direction,
                       const std::vector<LayerTagSlot>& slots);
    bool copyDiskItem(const MemoryOperationRequestPB::CopyItem& item,
                      CopyDirection                             direction,
                      const std::vector<LayerTagSlot>&          slots,
                      void*                                     staging_buffer);
    bool copyMemoryItemsGeneric(const MemoryOperationRequestPB&  request,
                                CopyDirection                    direction,
                                const std::vector<LayerTagSlot>& slots);
    bool validateCopyItemBacking(const MemoryOperationRequestPB::CopyItem& item) const;

    void                      checkLayerBlockStrideBytes() const;
    std::vector<LayerTagSlot> layerTagSlots() const;
    bool                      hasTypedLayerTagSlots(const std::vector<LayerTagSlot>& slots) const;
    bool                      supportsTypedPrefixCacheLayout(const std::vector<LayerTagSlot>& slots) const;
    bool                      checkLayerBlocks(const LayerBlockIds& layer_block_ids, size_t required_len) const;
    LayerAttnBlockIds         resourceLayerRegionBlocks(const KVCacheResource&           resource,
                                                        const std::vector<LayerTagSlot>& slots) const;
    bool                      checkLayerRegionBlocks(const LayerAttnBlockIds&         layer_attn_block_ids,
                                                     const std::vector<LayerTagSlot>& slots,
                                                     size_t                           required_len) const;
    bool                      gpuBlocksAllValid(const LayerBlockIds& layer_block_ids, size_t key_index) const;
    bool                      gpuBlocksAllValid(const LayerAttnBlockIds&         layer_attn_block_ids,
                                                const std::vector<LayerTagSlot>& slots,
                                                size_t                           key_index) const;
    bool                      usePrefixTreeMemoryCache() const;
    CacheGroupPolicy          groupPolicyForSlot(const LayerTagSlot& slot) const;
    CacheBlockKind            kindForSlot(const LayerTagSlot& slot) const;
    bool                      kindRequiredAt(const LayerAttnBlockIds&         layer_attn_block_ids,
                                             const std::vector<LayerTagSlot>& slots,
                                             size_t                           key_index,
                                             CacheBlockKind                   kind) const;
    std::vector<uint8_t>      prefixSlotValidMask(const LayerAttnBlockIds&         layer_attn_block_ids,
                                                  const std::vector<LayerTagSlot>& slots,
                                                  size_t                           key_index,
                                                  CacheBlockKind                   kind) const;
    size_t                    prefixKindBlockSize(CacheBlockKind kind, const std::vector<LayerTagSlot>& slots) const;
    std::shared_ptr<CopyPlan> buildPrefixCopyPlanForRead(const CacheKeysType&             cache_keys,
                                                         const BlockDependenciesType&     dependencies,
                                                         const LayerAttnBlockIds&         layer_attn_block_ids,
                                                         const std::vector<LayerTagSlot>& slots,
                                                         int                              start_index,
                                                         int                              read_num);
    std::shared_ptr<CopyPlan> buildPrefixCopyPlanForWrite(const CacheKeysType&             cache_keys,
                                                          const BlockDependenciesType&     dependencies,
                                                          const LayerAttnBlockIds&         layer_attn_block_ids,
                                                          const std::vector<LayerTagSlot>& slots,
                                                          int                              start_index,
                                                          int                              write_num,
                                                          bool&                            no_need_write);
    bool                      allocatePrefixBackingsForWrite(std::vector<CopyInfoPerKey>& copy_infos);
    bool                      allocateOnePrefixBacking(CopyInfoPerKey& copy_info);
    bool                      preparePrefixMergeSources(std::vector<CopyInfoPerKey>& copy_infos);
    void                      releasePrefixMergeSource(const CopyInfoPerKey& copy_info);
    bool                      mergePrefixExistingSlots(PrefixTreeMemoryBlockCache::CacheItem&         item,
                                                       const PrefixTreeMemoryBlockCache::MatchResult& existing,
                                                       const std::vector<LayerTagSlot>&               slots);
    bool                      mergePrefixConflictForCommit(CopyInfoPerKey&                        copy_info,
                                                           PrefixTreeMemoryBlockCache::CacheItem& item,
                                                           const std::vector<LayerTagSlot>&       slots);
    void                      putPrefixToCache(CopyInfoPerKey&                  copy_info,
                                               const BlockDependency&           dependency,
                                               const std::vector<LayerTagSlot>& slots);
    void                      releasePrefixRequestBacking(const CopyInfoPerKey& copy_info);
    void                      releasePrefixCacheBacking(const PrefixTreeMemoryBlockCache::CacheItem& item);
    void                      referencePrefixCacheBacking(const PrefixTreeMemoryBlockCache::CacheItem& item);
    bool                      copyPrefixMemoryItems(const MemoryOperationRequestPB&  request,
                                                    CopyDirection                    direction,
                                                    const std::vector<LayerTagSlot>& slots);

    bool                       freeBlocks(const std::vector<BlockIdxType>& blocks, bool cache_free = true);
    void                       referenceBlocks(const std::vector<BlockIdxType>& blocks, bool cache_ref = true);
    bool                       allocateBackingsForWrite(std::vector<CopyInfoPerKey>& copy_infos);
    bool                       allocateOneBacking(CopyInfoPerKey& copy_info);
    bool                       tryMallocMemoryBlock(CacheBlockKind kind, BlockIdxType& block);
    bool                       tryMallocDiskSlot(CacheBlockKind kind, int32_t& slot);
    void                       releaseRequestBacking(const CopyInfoPerKey& copy_info);
    void                       releaseCacheBacking(const MemoryDiskBlockCache::CacheItem& item);
    void                       referenceCacheBacking(const MemoryDiskBlockCache::CacheItem& item);
    std::shared_ptr<BlockPool> memoryPoolFor(CacheBlockKind kind) const;
    DiskBlockPoolPtr           diskPoolFor(CacheBlockKind kind) const;
    size_t                     maxDiskSlotStrideBytes() const;

    bool isDualPool() const;
    bool isFullOnlySlot(const LayerTagSlot& slot) const;
    bool mallocBlocksFromPool(const std::shared_ptr<BlockPool>&        pool,
                              const std::shared_ptr<MemoryBlockCache>& cache,
                              size_t                                   need_blocks,
                              std::vector<BlockIdxType>&               malloced_blocks);
    bool freeBlocksFromPool(const std::shared_ptr<BlockPool>& pool,
                            const std::vector<BlockIdxType>&  blocks,
                            bool                              cache_free);
    void referenceBlocksInPool(const std::shared_ptr<BlockPool>& pool,
                               const std::vector<BlockIdxType>&  blocks,
                               bool                              cache_ref);
    bool ensureEnoughFreeBlocksInPool(const std::shared_ptr<BlockPool>&        pool,
                                      const std::shared_ptr<MemoryBlockCache>& cache,
                                      size_t                                   need_blocks);
    void putToCacheInPool(const std::shared_ptr<BlockPool>&        pool,
                          const std::shared_ptr<MemoryBlockCache>& cache,
                          const MemoryBlockCache::CacheItem&       item);

    void                       initBlockPool();
    void                       initDiskBlockPools();
    bool                       diskCacheEnabled() const;
    bool                       copyItemUsesLayerBlocks(const MemoryOperationRequestPB::CopyItem& item) const;
    int64_t                    copyPlanTimeoutMs(const std::shared_ptr<CopyPlan>& copy_plan) const;
    std::shared_ptr<BlockPool> createBlockPool(size_t block_size, size_t pool_size_mb) const;
    std::string                blockPoolDebugString() const;
    size_t                     memoryCacheBlockSizeBytes() const;
    void                       putToCache(const MemoryBlockCache::CacheItem& item);
    void                       putToCache(CopyInfoPerKey& copy_info);
    bool putToCache(const MemoryDiskBlockCache::CacheItem& item, bool already_has_cache_ref = false);

    void reportMatchMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t matched_block_num);
    void reportReadMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t read_block_num);
    void reportWriteMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t write_block_num);
    void reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    void reportDiskMatchMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t matched_block_num);
    void reportDiskReadMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t read_block_num);
    void reportDiskWriteMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t write_block_num);
    void reportDiskCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    int  cpSizeForMetrics() const;
    int  cacheKeyTokensPerBlockForMetrics() const;
    void reportEvictionLifetime(CacheBlockKind kind, CacheBackingType backing_type, int64_t created_time_us);
    void reportMetricsLoop();

private:
    const CacheConfig&                cache_config_;
    const KVCacheConfig&              kv_cache_config_;
    const ParallelismConfig           parallelism_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    const std::vector<std::string>    tp_addrs_;

    std::shared_ptr<BlockPool>                  block_pool_;
    mutable std::mutex                          malloc_mutex_;
    std::shared_ptr<MemoryDiskBlockCache>       block_cache_;
    std::shared_ptr<PrefixTreeMemoryBlockCache> prefix_block_cache_;
    std::unique_ptr<DiskMountGuard>             disk_mount_guard_;
    std::shared_ptr<DiskBlockPool>              complete_disk_pool_;
    std::shared_ptr<DiskBlockPool>              incomplete_disk_pool_;
    std::shared_ptr<BroadcastManager>           broadcast_manager_;
    std::shared_ptr<autil::LockFreeThreadPool>  wait_done_thread_pool_;

    std::shared_ptr<BlockPool> complete_pool_;
    std::shared_ptr<BlockPool> incomplete_pool_;
    size_t                     complete_block_size_{0};
    size_t                     incomplete_block_size_{0};
    std::shared_ptr<BlockPool> compressed_pool_;
    std::shared_ptr<BlockPool> state_swa_pool_;
    size_t                     compressed_block_size_{0};
    size_t                     state_swa_block_size_{0};
    bool                       use_prefix_tree_memory_cache_{false};

    // metrics reporter
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<std::thread> metrics_reporter_thread_{nullptr};
    std::atomic<bool>            stop_{false};
};

}  // namespace rtp_llm
