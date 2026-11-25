#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class CPSlotMapper;
class CacheStore;
class BroadcastManager;
class PrefillCacheHitMetricsReporter;
class KVCacheAllocationWaitState;
class KVCacheEventReporter;
class FrozenTaskRecovery;
class TransferCompletionCleanup;

using PDLoadDestinationBlocks = std::unordered_map<std::string, BlockIndicesType>;
using PDLoadDestinationSelector =
    std::function<PDLoadDestinationBlocks(const GroupBlockIds&, size_t key_count, int64_t reuse_blocks)>;

class PDLoadDestinationHold {
public:
    std::shared_ptr<const std::atomic<bool>> publicationStatus() const {
        return content_ready_;
    }
    void markLoadSuccessful() {
        content_ready_->store(true, std::memory_order_release);
    }
    const GroupBlockIds& blockIdsSnapshot() const {
        return block_ids_snapshot_;
    }
    const PDLoadDestinationBlocks& protectedBlocks() const {
        return protected_blocks_;
    }

private:
    friend class KVCacheManager;
    std::shared_ptr<std::atomic<bool>> content_ready_ = std::make_shared<std::atomic<bool>>(false);
    struct PoolPins {
        DeviceBlockPoolPtr                pool;
        std::vector<RemotePublicationPin> pins;  // Destroy before the strong pool owner.
    };
    GroupBlockIds           block_ids_snapshot_;
    PDLoadDestinationBlocks protected_blocks_;
    std::vector<PoolPins>   pools_;
};

class KVCacheManager {
public:
    KVCacheManager(const CacheConfig&                 config,
                   bool                               warmup                       = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter             = nullptr,
                   const KVCacheConfig&               kv_cache_config              = KVCacheConfig{},
                   const ParallelismConfig&           parallelism_config           = ParallelismConfig{},
                   const RuntimeConfig&               runtime_config               = RuntimeConfig{},
                   const SpeculativeExecutionConfig&  sp_config                    = SpeculativeExecutionConfig{},
                   const PDSepConfig&                 pd_sep_config                = PDSepConfig{},
                   const CacheStoreConfig&            cache_store_config           = CacheStoreConfig{},
                   bool                               use_device_malloc_block_pool = false);
    ~KVCacheManager();

    // 初始化和配置相关
    bool init();
    bool initialized() const {
        return coordinator_manager_ != nullptr;
    }

    // TP0 owns request block allocation; other ranks only receive the block IDs.
    bool isAllocatorOwner() const {
        return parallelism_config_.tp_rank == 0;
    }

    const CacheConfig& cacheConfig() const;
    const CacheConfig& getMTPModuleCacheConfig(int mtp_module_id) const;

    // 显存管理和缓存分配
    MallocResult malloc(const MallocInfo& malloc_info);
    void         free(const FreeInfo& free_info);
    bool         abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    bool         detachSharedBlocksForRecompute(const BatchKVCacheResourcePtr& resource, int device_ready_reuse_length);
    // Outputs the resident key-prefix count; ordinary inserts report zero.
    void insertIntoCache(const InsertInfo& insert_info, size_t& resident_prefix_length);
    bool captureCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end, bool finished = true) const;

    size_t decodePublicationUnit() const;
    // No writer last-use proof: keep ownership and charge the existing frozen
    // pool guard rather than silently accumulating invisible request holds.
    // False means native custody was unavailable; keep ordinary ownership.
    bool freezeUnknownDecodeWriter(const BatchKVCacheResource& resource) const noexcept;
    // The caller owns and serializes the request block table through capture.
    std::shared_ptr<PDLoadDestinationHold> holdPDLoadDestination(const GroupBlockIds&           rows,
                                                                 const PDLoadDestinationBlocks& targets) const;
    // Speculative capture may defer this wait only while the original stream
    // owns the pages and its final draft-event host wait precedes promotion in
    // specUpdate. Standalone captures retain their synchronous default.
    std::shared_ptr<const DecodeEndpoint> retainDecodeEndpoint(const torch::Tensor& physical_ids,
        const std::vector<std::string>& tags, size_t row, size_t end,
        const std::shared_ptr<const DecodeEndpoint>& previous = {}, bool defer_writer_wait = false) const;
    // Ordinary dispatch has already joined its writer before updating the
    // stream. Read its host block map without another GPU metadata copy/wait.
    std::shared_ptr<const DecodeEndpoint> retainDecodeEndpoint(const BatchKVCacheResource& resource,
        size_t end, const std::shared_ptr<const DecodeEndpoint>& previous = {}) const;

    // Decode-side P/D admission allocates destination blocks before cache handoff.  When pools are
    // temporarily full, waiters use this generation instead of polling malloc in a tight loop.
    // Capture the generation before an allocation attempt; waitForAllocationChange() then cannot
    // miss a release racing with that attempt.
    uint64_t allocationGeneration() const;
    bool     waitForAllocationChange(uint64_t observed_generation, int64_t timeout_ms, int64_t minimum_wait_ms = 0);

    int
    singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len, int reserve_step) const;

    int estimatePeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                               int                            seq_len,
                               int                            common_seq_len,
                               int                            remaining_tokens,
                               int                            reserve_step,
                               bool                           enable_reuse_cache,
                               int                            target_batch_size) const;

    // 块操作相关
    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const torch::Tensor& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);
    void blockBatchCopyByGroup(const std::vector<TaggedBlockIdPair>& copy_mapping);
    void blockBatchCopyForForward(const torch::Tensor& copy_mapping, const std::vector<std::string>& group_tags);

    bool updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                       const std::vector<int>&         block_src_batch,
                       bool                            copy_last_block,
                       std::vector<TaggedBlockIdPair>& block_update_mapping);

    // 地址转换和缓冲区访问
    BlockAddrInfo          convertIndexToAddr(int block_index, int layer_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int block_index, int layer_id) const;
    std::vector<BlockInfo>
                  convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const;
    BlockAddrInfo convertIndexToAddr(int layer_id, const std::string& group_tag, int block_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& group_tag, int block_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& group_tag, int block_id, int partition_count, int partition_id) const;

    GroupedCacheLayerLayout allLayerCacheBase() const;

    // for main model; grouped layout preserves layers that own multiple cache groups
    GroupedCacheLayerLayout getMainModelGroupedCacheLayerLayout() const;
    // for mtp module
    GroupedCacheLayerLayout getMTPModuleGroupedCacheLayerLayout(int mtp_module_id) const;

    // 资源统计和信息查询
    size_t      freeBlocksNum() const;
    size_t      availableBlocksNum() const;
    size_t      reserveBlocksNum() const;
    size_t      availableTokensNum() const;
    size_t      totalBlocksNum() const;
    size_t      maxAvailableTokensNum() const;
    KVCacheInfo getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const;

    // 系统资源管理
    void regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void stopMetricsReporter();

    // CacheStore ownership (set by RemoteRpcServer, read during model forward)
    void                        setCacheStore(std::shared_ptr<CacheStore> cache_store);
    std::shared_ptr<CacheStore> getCacheStore() const;

    // for every single rank
    // Returns whether a trustworthy mem_response was formed, not whether the transfer
    // succeeded; the transfer outcome is reported through mem_response.code.
    bool executeFunction(
        const FunctionRequestPB& request,
        FunctionResponsePB& response,
        std::chrono::steady_clock::time_point completion_deadline = std::chrono::steady_clock::time_point::max());

    // One observation of the original execution; does not release frozen references.
    TransferCompletionQueryResult queryTransferCompletion(const FrozenTask::ExecutionInfo& execution,
                                                         int timeout_ms) const;

    TransferCompletionForgetResult forgetTransferCompletion(const TransferCompletionTarget& target,
                                                            const std::vector<TransferExecutionTrace>& traces,
                                                            int timeout_ms) const;

    // Snapshot both existing owners; taking a snapshot does not transfer custody.
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

    BlockTreeCachePtr blockTreeCache() const {
        return block_tree_cache_;
    }

    // Increment KV cache reference count for PD separation (connector refcount)
    std::shared_ptr<KVCacheResource>
    incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector = true);

    // CP page-level RR sharding context. Returns nullptr when sharding is not active
    // (single-rank or kv_cache_sharded=false).  Used by connector / cache_store to
    // remap cacheKeys -> last-rank-key namespace.
    std::shared_ptr<CPSlotMapper> cpSlotMapper() const {
        return cp_slot_mapper_;
    }

private:
    template<typename ReadGroup>
    std::shared_ptr<const DecodeEndpoint> retainDecodeEndpointPages(size_t end,
        const std::shared_ptr<const DecodeEndpoint>& previous, ReadGroup&& read_group,
        bool defer_writer_wait) const;
    std::function<void()>             allocationChangeCallback() const;
    void                              reportMetricsLoop();
    void                              missingCleanupLoop();
    void                              stopMissingCleanup();
    void                              reportPrefillCacheHitMetrics(const MallocInfo& malloc_info, bool is_first_malloc);
    std::shared_ptr<BroadcastManager> createMultiRankBlockTransferManager() const;
    TransferTaskRegistry::State localTransferCompletion(
        const TransferExecutionTrace& trace_id,
        std::chrono::steady_clock::time_point completion_deadline = std::chrono::steady_clock::time_point::max()) const;
    TransferTaskRegistry::ForgetResult localForgetTransferCompletion(
        const TransferExecutionTrace& trace_id, bool force = false) const;
    bool startFrozenTaskRecovery();
    void stopFrozenTaskRecovery() noexcept;

    // 成员变量
    CacheConfig                config_;
    CoordinatorCacheManagerPtr coordinator_manager_;
    mutable std::atomic<bool>  decode_capture_stopped_{false};

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const KVCacheConfig                kv_cache_config_;
    const ParallelismConfig            parallelism_config_;
    const RuntimeConfig                runtime_config_;
    const SpeculativeExecutionConfig   sp_config_;
    const PDSepConfig                  pd_sep_config_;
    const bool                         use_device_malloc_block_pool_;
    const bool                         warmup_;
    KVCacheEventPublisherPtr           cache_event_publisher_;
    void                               initCacheEventPublisher(std::shared_ptr<KVCacheEventReporter> reporter = nullptr);
    void                               stopCacheEventPublisher();

    std::shared_ptr<CPSlotMapper>                   cp_slot_mapper_;
    std::unique_ptr<PrefillCacheHitMetricsReporter> prefill_cache_hit_metrics_reporter_;

    std::atomic<bool> stop_{false};
    std::thread       metrics_reporter_thread_;

    std::mutex              missing_cleanup_mutex_;
    std::condition_variable missing_cleanup_cv_;
    bool                    missing_cleanup_stopped_{false};
    std::thread             missing_cleanup_thread_;

    BlockTreeCachePtr                           block_tree_cache_;
    std::shared_ptr<BroadcastManager>           transfer_broadcast_manager_;
    std::unique_ptr<FrozenTaskRecovery>         frozen_task_recovery_;
    std::shared_ptr<TransferCompletionCleanup> completion_cleanup_;
    std::shared_ptr<KVCacheAllocationWaitState> allocation_wait_state_;

    mutable std::mutex          cache_store_mutex_;
    std::shared_ptr<CacheStore> cache_store_;
};

}  // namespace rtp_llm
