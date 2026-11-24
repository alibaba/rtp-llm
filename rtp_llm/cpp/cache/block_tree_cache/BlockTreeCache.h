#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class BlockTransferDispatcher;
class FrozenTask;
class TransferCompletionCleanup;
class FullPrefixInvariantScanner;
class MockMemoryCacheSource;
class MockDiskCacheSource;
class RemoteMockAdapter;
struct CacheStats {
    size_t tree_node_count{0};
    size_t device_heap_total_size{0};
    size_t host_heap_total_size{0};
    size_t disk_heap_total_size{0};
};
struct BlockTreeKeySnapshot {
    int64_t                   version{-1};
    std::vector<CacheKeyType> keys;
};
struct CachePressurePoolSnapshot: BlockTreePoolMetricsSnapshot {
    std::string pool_id;
    uint64_t    pending_release_blocks{0};
    uint64_t    published_host_blocks{0};
    bool        request_ref_blocks_available{false};
    uint64_t    disk_read_bytes{0}, disk_write_bytes{0};
};
struct CachePressureGroupSnapshot {
    size_t                     group_id{0};
    CacheGroupType             group_type{CacheGroupType::FULL};
    std::vector<std::string>   device_pool_ids;
    std::optional<std::string> host_pool_id, disk_pool_id;
    TierWatermark              device_watermark, host_watermark, disk_watermark;
    uint64_t                   watermark_trigger_count{0};  // DEVICE high-watermark transitions.
    std::vector<std::string>   group_tags;
    uint64_t                   block_tokens{0}, payload_bytes{0}, storage_bytes{0};
    bool                       crc_enabled{false};
};
struct CachePressureTransferSnapshot: TransferEvidence {
    size_t                 group_id{0};
    std::string            pool_id;
    Tier                   pool_tier{Tier::NONE};
    CacheTransferOperation operation{CacheTransferOperation::LOAD};
    Tier                   source_tier{Tier::NONE}, target_tier{Tier::NONE};
};
struct CachePressureOwnership {
    enum class Mode {
        UNSUPPORTED = 0,
        INDEPENDENT = 1,
        SHARED_TP   = 2
    };
    Mode             mode{Mode::UNSUPPORTED};
    int              owner_world_rank{0};
    std::vector<int> participant_world_ranks;
    bool             all_participant_publication{false};
};
struct LocalMockTargets {
    bool     memory_enabled, disk_enabled;
    double   memory_ratio, disk_ratio;
    uint64_t memory_ppm, disk_ppm;
};
struct CachePressureSnapshot {
    std::vector<CachePressurePoolSnapshot>        pools;
    std::vector<CachePressureGroupSnapshot>       groups;
    std::vector<CachePressureTransferSnapshot>    transfers;
    std::optional<CachePressureOwnership>         ownership;
    bool                                          complete{true};
    std::optional<MockMemoryCacheMetricsSnapshot> mock_memory;
    std::optional<MockCacheMetricsSnapshot>       mock_cache;
    MockDiskWriteMetricsSnapshot                  mock_disk_write;
    RemoteMockCompletionStats                     mock_remote_write;
    RemoteMockSafetySnapshot                      remote_mock_safety;
    std::optional<RemoteMockRuntimeSnapshot>      remote_runtime;
    uint64_t                                      remote_write_acknowledged_payload_bytes{0};
    uint64_t                                      remote_read_payload_bytes{0};  // Actual application bytes for this rank only.
    std::optional<LocalMockTargets>               local_mock_targets;
    std::optional<LocalStoreSnapshot>             local_store;
};
// Unified configuration for BlockTreeCache behavior and pool sizing.
struct BlockTreeCacheConfig {
    bool                                  mock_remote_cache{false};
    double                                mock_remote_cache_hit_ratio{0};
    int                                   mock_remote_cache_timeout_ms{10000};
    bool                                  mock_disk_cache{false};
    double                                mock_disk_cache_hit_ratio{0.0};
    uint64_t                              mock_disk_cache_seed{1};
    size_t                                mock_disk_cache_source_blocks{64};
    bool                                  mock_memory_cache{false};
    bool                                  mock_memory_cache_real_match{false};
    double                                mock_memory_cache_hit_ratio{0.0};
    bool                                  mock_memory_cache_coordinator{true};
    std::optional<CachePressureOwnership> pressure_ownership;
    // ---- Tier enable flags ----
    bool enable_device_cache{true};
    bool enable_host_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};

    // ---- Per-tier watermark ----
    using TierWatermark = rtp_llm::TierWatermark;
    TierWatermark watermark_device;
    TierWatermark watermark_host;
    TierWatermark watermark_disk;

    // ---- Per-tier eviction policy ----
    EvictionPolicy device_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy host_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy disk_eviction_policy{EvictionPolicy::LRU};

    // ---- Shared load/eviction task pool ----
    int    task_pool_size{4};
    size_t business_queue_max_size{10000};

    // ---- Per-rank WorkLoop owners or legacy workers per direction ----
    size_t transfer_worker_count{2};
    size_t transfer_queue_max_size{10000};

    // ---- Transfer deadlines, including queue wait and cross-rank work ----
    int host_cache_sync_timeout_ms{10000};
    // Preserve the existing name for disk loads.
    int disk_cache_sync_timeout_ms{3000};
    // Outbound disk transfers for eviction.
    int disk_cache_evict_timeout_ms{3000};

    // Async: blocks per load/evict pair, split by direction and shared by groups.
    // Legacy: rank-wide byte budget measured in maximum-stride blocks.
    size_t device_disk_staging_block_count{128};
    // Shared descriptor batching limit for watermark planning and all transfer directions.
    size_t max_descriptors_per_transfer_batch{32};
    size_t disk_cache_evict_max_pending_bytes{0};

    // ---- FULL prefix invariant scanner (diagnostic only) ----
    // The factory zeroes this on ranks that do not own a mutable BlockTree.
    int64_t full_prefix_scan_interval_ms{30000};  // 30 seconds; 0 = disabled, no thread

    // ---- Query helpers ----
    bool isTierEnabled(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return enable_device_cache;
            case Tier::HOST:
                return enable_host_cache;
            case Tier::DISK:
                return enable_disk_cache;
            case Tier::REMOTE:
                return enable_remote_cache;
            default:
                return false;
        }
    }

    TierWatermark watermarkForTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return watermark_device;
            case Tier::HOST:
                return watermark_host;
            case Tier::DISK:
                return watermark_disk;
            default:
                return {};
        }
    }
};

// BlockTreeCache: eviction workflow coordinator.
// Owns BlockTree, GroupSets, HostBlockPool (L2), DiskBlockPool (L3),
// StorageBackend, and the cache workflow collaborators injected by the factory.
// Local tiers require DISK -> HOST -> DEVICE; remote storage is independent.
class BlockTreeCache {
public:
    using TierWatermark = BlockTreeCacheConfig::TierWatermark;

    BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                   BlockTreeCacheConfig                           config,
                   std::shared_ptr<StorageBackend>                storage_backend,
                   std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                   std::unique_ptr<BlockTreeTaskPool>             task_pool,
                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                   std::shared_ptr<MockMemoryCacheSource>         mock_memory_cache_source = nullptr,
                   std::shared_ptr<MockDiskCacheSource>           mock_disk_cache_source   = nullptr,
                   std::shared_ptr<RemoteMockAdapter>             remote_adapter           = nullptr);

    const std::shared_ptr<RemoteMockAdapter>& remoteMockAdapter() const {
        return remote_adapter_;
    }
    // Thread-safe read while the cache is owned. Faults are permanent: stop the
    // measurement and restart the full engine; quarantine is released only at teardown.
    RemoteMockSafetySnapshot remoteMockSafety() const;
    struct RemoteCompletionRequest;
    using RemoteGeometryVisitor = std::function<bool(size_t, const BlockIndicesType&)>;
    // A request ticket covers successful allocation through actual resource release,
    // including deferred bookkeeping after consumer/HTTP completion.
    std::shared_ptr<RemoteCompletionRequest> enrollRemoteCompletion();
    void completeRemoteRequest(const std::shared_ptr<RemoteCompletionRequest>&          request,
                               const std::function<bool(const RemoteGeometryVisitor&)>& capture);
    // Permanent engine shutdown only. Already enrolled requests may still capture.
    void stopRemoteCompletionAdmission();
    // Reusable window barrier. Caller quiesces external ingress; new traffic can
    // extend the wait. Never synchronously wait under stream/cache/scheduler locks.
    // Callback follows request retirement, all logical intents and physical drain.
    // remoteMockSafety() remains authoritative for unknown-peer quarantine.
    void drainRemoteCompletions(std::function<void()> callback);

    ~BlockTreeCache();
    bool init();
    // Retry deferred watermarks after external pins release. Safe inside a pool callback.
    void onCapacityChanged();
    // Reject pending loads and wait for allocator callbacks before its members are destroyed.
    void shutdownLoads();

    BlockTreeMatchResult match(const CacheKeysType&               cache_keys,
                               const std::function<bool(size_t)>& valid_prefix           = {},
                               bool                               enable_storage_backend = true,
                               std::optional<int64_t>             request_id             = {},
                               bool                               skip_linear            = false);
    BlockTreeMatchResult matchMockCache(const CacheKeysType&                     cache_keys,
                                        const std::function<bool(size_t)>&       valid_prefix,
                                        const BlockTreeLoader::MockCachePlanner& planner);
    void                 insert(const CacheKeysType&                              cache_keys,
                                const std::vector<std::vector<GroupSetResource>>& resources,
                                Tier                                              target_tier);
    // Returns the resident key-prefix count, including nodes that were already resident.
    size_t insert(const CacheKeysType&                              cache_keys,
                  const std::vector<std::vector<GroupSetResource>>& resources,
                  Tier                                              target_tier,
                  bool                                              is_resident,
                  std::optional<int64_t>                            request_id         = {},
                  size_t                                            remote_reuse_begin = 0,
                  size_t                                            remote_reuse_end   = 0,
                  bool                                              mock_memory_cache  = false,
                  bool                                              mock_disk_cache    = false);
    // Directly reclaim up to num_blocks device blocks belonging to one group set
    // (target_tier = NONE, content dropped). Returns the number actually freed.
    // Skips plans containing device blocks that would remain referenced after dropping.
    int evictForGroup(std::string_view group_tag, size_t num_blocks);

    CachePressureSnapshot         pressureSnapshot() const;
    CacheStats                    getStats() const;
    std::vector<CachePoolMetrics> poolMetrics() const;
    void                          reportMetrics() const;
    void                          reportUnfreezeMetrics(const FrozenTask::ExecutionInfo& execution, bool forced) const;
    void                          reportUnfreezeZeroMetrics(const std::vector<CachePoolMetrics>& pools) const;
    BlockTreeKeySnapshot          getKeySnapshot(int64_t latest_version = -1) const;
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    bool            abortPendingLoad(const std::shared_ptr<AsyncContext>& context);

    BlockIndicesType matchedBlocksForGroup(std::string_view                      group_tag,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

    TransferResult                     executeTransfer(TransferTask task);
    TransferTaskRegistry::State        localTransferCompletion(uint64_t trace_id) const;
    TransferTaskRegistry::ForgetResult forgetLocalTransferCompletion(uint64_t trace_id) const;
    void cleanupExpiredMissing() const;
    // Configure before accepting coordinator work.
    void                                     setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup);
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

    // Accessors
    BlockTree* tree() const {
        return tree_.get();
    }
    const std::vector<GroupSetPtr>& groupSets() const {
        return tree_->groupSets();
    }
    std::shared_ptr<StorageBackend> storageBackend() const {
        return storage_backend_;
    }
    std::shared_ptr<const MockMemoryCacheSource> mockMemoryCacheSource() const {
        return mock_memory_cache_source_;
    }
    std::shared_ptr<const MockDiskCacheSource> mockDiskCacheSource() const {
        return mock_disk_cache_source_;
    }

    // Tier enable queries
    bool isDeviceCacheEnabled() const {
        return config_.enable_device_cache;
    }
    bool isHostCacheEnabled() const {
        return config_.enable_host_cache;
    }
    bool isDiskCacheEnabled() const {
        return config_.enable_disk_cache;
    }
    bool isRemoteCacheEnabled() const {
        return config_.enable_remote_cache;
    }
    bool isInitialized() const {
        return initialized_;
    }

    void startLocalStoreRequest() {
        metrics_reporter_->startLocalStoreRequest();
    }
    void finishLocalStoreRequest(bool released) {
        metrics_reporter_->finishLocalStoreRequest(released);
    }

    const BlockTreeCacheConfig& config() const {
        return config_;
    }

private:
    void checkWatermark();
    // Caller holds mutex_.
    void onWorkflowSettledLocked(bool check_watermark);

    struct RemoteCompletionState;
    std::shared_ptr<RemoteCompletionState>         remote_completions_;
    BlockTreeCacheConfig                           config_;
    std::unique_ptr<BlockTree>                     tree_;
    std::shared_ptr<StorageBackend>                storage_backend_;
    // Declared before workers so reservations outlive loader/transfer destruction.
    std::shared_ptr<MockMemoryCacheSource>         mock_memory_cache_source_;
    std::shared_ptr<MockDiskCacheSource>           mock_disk_cache_source_;
    std::shared_ptr<RemoteMockAdapter>             remote_adapter_;
    std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher_;
    std::unique_ptr<BlockTreeTaskPool>             task_pool_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
    mutable std::mutex                             mutex_;
    BlockTreeEvictor                               evictor_;
    bool                                           initialized_{false};
    std::atomic<bool>                              stopping_{false};
    std::atomic<bool>                              watermark_check_pending_{false};
    BlockTreeLoader                                loader_;
    BlockTreeStorer                                storer_;
    std::unique_ptr<FullPrefixInvariantScanner>    full_prefix_scanner_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
