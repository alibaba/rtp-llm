#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "autil/LockFreeThreadPool.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillCommitCoordinator.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillProtocol.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillTypes.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class BlockPool;
class BroadcastManager;
class KVCacheAllocator;
class MemoryAsyncContext;
struct StagedMemoryCopyScratch;

class KVCacheMemoryConnector: public KVCacheConnector {
public:
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const KVCacheConfig&                     kv_cache_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           const std::vector<std::string>&          tp_addrs,
                           const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr,
                           int                                      tp_rank          = 0,
                           int                                      tp_size          = 1);

    // master = the only rank that drives LRU eviction, allocates disk slots, and owns
    // the committed disk index. Non-master ranks only execute master-broadcast disk ops.
    // README §"TP / 多 rank 一致性" — single-rank deployments behave like rank 0 self-master.
    bool isMaster() const {
        return tp_rank_ == 0;
    }
    int tpRank() const {
        return tp_rank_;
    }
    int tpSize() const {
        return tp_size_;
    }
    ~KVCacheMemoryConnector() override;

public:
    bool init();

    // Called after every rank has started its gRPC server. The cross-rank
    // capability HELLO cannot run during connector init because peer RPC servers
    // are not listening yet.
    bool postInit();

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
    struct LayerRegionSlot {
        int               layer_id{-1};
        KVCacheRegionName region_name{KVCacheRegionName::DEFAULT};
        int               group_id{-1};
        size_t            stride_bytes{0};
    };
    struct CopyInfoPerKey {
        enum class SourceType {
            MEMORY_BLOCK = 0,
            DISK_SLOT    = 1,
        };
        CacheKeyType                  cache_key{0};
        BlockIdxType                  mem_block{NULL_BLOCK_IDX};
        std::vector<BlockIdxType>     gpu_blocks;
        bool                          is_complete{true};
        SourceType                    source_type{SourceType::MEMORY_BLOCK};
        DiskSpillBlockCache::DiskSlot disk_slot;
    };
    enum class CopyDirection {
        H2D = 0,
        D2H = 1
    };
    struct CopyPlan {
        std::vector<CopyInfoPerKey> copy_infos;
        CopyDirection               direction;
    };

    std::shared_ptr<CopyPlan> buildCopyPlanForRead(const CacheKeysType&                cache_keys,
                                                   const LayerAttnBlockIds&            layer_attn_block_ids,
                                                   const std::vector<LayerRegionSlot>& slots,
                                                   int                                 start_index,
                                                   int                                 read_num);
    std::shared_ptr<CopyPlan> buildCopyPlanForWrite(const CacheKeysType&                cache_keys,
                                                    const LayerAttnBlockIds&            layer_attn_block_ids,
                                                    const std::vector<LayerRegionSlot>& slots,
                                                    int                                 start_index,
                                                    int                                 write_num,
                                                    bool&                               no_need_write);
    std::shared_ptr<CopyPlan> createCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos,
                                             const CopyDirection&               direction);
    bool startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context, const std::shared_ptr<CopyPlan>& copy_plan);
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
         sendCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;
    void printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;

    bool                     prepareCopyBuffers(BlockIdxType                     mem_block,
                                                const std::vector<BlockIdxType>& gpu_blocks,
                                                CopyDirection                    direction,
                                                std::vector<torch::Tensor>&      dst,
                                                std::vector<torch::Tensor>&      src);
    bool                     tryCopyCacheWithBatchedMemoryCopy(const MemoryOperationRequestPB&     request,
                                                               CopyDirection                       direction,
                                                               const std::vector<LayerRegionSlot>& slots);
    bool                     tryCopyCacheWithStagedMemoryCopy(const MemoryOperationRequestPB&     request,
                                                              CopyDirection                       direction,
                                                              const std::vector<LayerRegionSlot>& slots);
    bool                     copyMixedMemoryDiskToDevice(const MemoryOperationRequestPB&     request,
                                                         const std::vector<LayerRegionSlot>& slots);
    bool                     spillMemoryToDisk(const MemoryOperationRequestPB& request);
    bool                     deleteDiskSlots(const MemoryOperationRequestPB& request);
    StagedMemoryCopyScratch& stagedCopyScratchForDevice(int device_index);
    bool                     appendCopyBytesToBuffers(const BlockInfo&            mem_block,
                                                      const BlockInfo&            gpu_block,
                                                      size_t                      byte_off,
                                                      CopyDirection               direction,
                                                      std::vector<torch::Tensor>& dst,
                                                      std::vector<torch::Tensor>& src);

    void                         checkLayerBlockStrideBytes() const;
    std::vector<LayerRegionSlot> layerRegionSlots() const;
    bool                         hasTypedLayerRegionSlots(const std::vector<LayerRegionSlot>& slots) const;
    bool                         isDsv4TypedCacheLayout(const std::vector<LayerRegionSlot>& slots) const;
    bool                         checkLayerBlocks(const LayerBlockIds& layer_block_ids, size_t required_len) const;
    bool                         checkLayerRegionBlocks(const LayerAttnBlockIds&            layer_attn_block_ids,
                                                        const std::vector<LayerRegionSlot>& slots,
                                                        size_t                              required_len) const;
    bool                         gpuBlocksAllValid(const LayerBlockIds& layer_block_ids, size_t key_index) const;
    bool                         gpuBlocksAllValid(const LayerAttnBlockIds&            layer_attn_block_ids,
                                                   const std::vector<LayerRegionSlot>& slots,
                                                   size_t                              key_index) const;

    bool mallocBlocks(size_t need_blocks, std::vector<BlockIdxType>& malloced_blocks);
    bool freeBlocks(const std::vector<BlockIdxType>& blocks, bool cache_free = true);
    void referenceBlocks(const std::vector<BlockIdxType>& blocks, bool cache_ref = true);
    bool ensureEnoughFreeBlocks(size_t need_blocks);

    void                       initBlockPool();
    void                       initDiskSpillCache();
    std::shared_ptr<BlockPool> createBlockPool(size_t block_size, size_t pool_size_mb) const;
    size_t                     memoryBlockSizeBytes() const;
    std::string                blockPoolDebugString() const;
    void                       putToCache(const MemoryBlockCache::CacheItem& item);
    bool                       spillMemoryItemToDisk(const MemoryBlockCache::CacheItem& item);
    bool                       sendDeleteDiskSlot(const DiskSpillBlockCache::DiskSlot& slot) const;

    // PendingSpill: produced by ensureEnoughFreeBlocks under lock, drained by
    // flushPendingSpills outside lock. Owns the heap-allocated staging buffer.
    struct PendingSpill {
        MemoryBlockCache::CacheItem   item;
        std::vector<char>             staging;
        DiskSpillBlockCache::DiskItem reserved_slot;
        bool                          slot_reserved{false};
    };

    void flushPendingSpills(std::vector<PendingSpill>&& pendings);
    void zeroNullSlots(const MemoryBlockCache::CacheItem& item, std::vector<char>& staging) const;

    // Init handshake: master broadcasts DISK_SPILL_HELLO carrying schema_hash
    // and capability_mask; waits for all workers; init fails on mismatch or
    // timeout. README §"Init 一致性和 capability handshake".
    bool runDiskSpillHandshake();
    bool handleDiskSpillHello(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response);
    bool handleSpillWriteStatus(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response);

    std::string computeSchemaHash() const;
    OpSequenceTracker& trackerForIncomingRank(int peer_seq_id);  // very simple peer-id hashing

    // Broadcast helpers used by CommitCoordinator
    bool broadcastSpillToWorkers(SpillJobId job_id,
                                  const DiskSpillBlockCache::DiskItem& slot,
                                  BlockIdxType source_mem_block);
    bool broadcastDeleteToWorkers(const DiskSpillBlockCache::DiskItem& slot);
    SpillWriteStatus pollWorkerSpillStatus(int worker_idx, SpillJobId job_id);

    void reportMatchMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t matched_block_num);
    void reportReadMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t read_block_num);
    void reportWriteMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t write_block_num);
    void reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    void reportDiskMatchMetrics(bool success, int64_t latency_us, int64_t input, int64_t matched, bool contention);
    void reportDiskWriteMetrics(bool success, int64_t latency_us, int64_t input, int64_t written);
    void reportDiskReadMetrics(bool success, int64_t latency_us, int64_t input, int64_t read_token);
    void reportDiskCopyMetrics(bool success, int64_t latency_us, const std::string& direction, int disk_id);
    void reportDiskError(const std::string& error_type, const std::string& op = "", int disk_id = -1);
    void reportMetricsLoop();

private:
    const CacheConfig&                cache_config_;
    const KVCacheConfig&              kv_cache_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    const std::vector<std::string>    tp_addrs_;
    const int                         tp_rank_{0};
    const int                         tp_size_{1};

    std::shared_ptr<BlockPool>                              block_pool_;
    mutable std::mutex                                      malloc_mutex_;
    mutable std::mutex                                      staged_copy_scratch_mutex_;
    std::map<int, std::unique_ptr<StagedMemoryCopyScratch>> staged_copy_scratch_by_device_;
    std::shared_ptr<MemoryBlockCache>                       block_cache_;
    std::shared_ptr<DiskSpillBlockCache>                    disk_spill_cache_;
    std::shared_ptr<DiskSpillCommitCoordinator>             commit_coordinator_;
    std::shared_ptr<BroadcastManager>                       broadcast_manager_;
    std::shared_ptr<BroadcastManager>                       disk_broadcast_manager_;
    std::shared_ptr<autil::LockFreeThreadPool>              wait_done_thread_pool_;
    OpSequenceTracker                                       outgoing_op_sequence_;
    std::mutex                                              incoming_op_sequence_mutex_;
    std::unordered_map<int, OpSequenceTracker>              incoming_op_sequence_;
    std::string                                             schema_hash_;
    std::atomic<int64_t>                                    last_metrics_log_ms_{0};

    // metrics reporter
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<std::thread> metrics_reporter_thread_{nullptr};
    std::atomic<bool>            stop_{false};
};

}  // namespace rtp_llm
