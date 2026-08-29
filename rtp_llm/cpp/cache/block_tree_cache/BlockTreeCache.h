#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class BlockTransferDispatcher;
class FullPrefixInvariantScanner;
struct CacheStats {
    size_t tree_node_count{0};
    size_t device_heap_total_size{0};
    size_t host_heap_total_size{0};
    size_t disk_heap_total_size{0};
};
struct BlockTreeKeySnapshot {
    int64_t                   version{0};
    std::vector<CacheKeyType> keys;
};
// Unified configuration for BlockTreeCache behavior and pool sizing.
struct BlockTreeCacheConfig {
    // ---- Tier enable flags ----
    bool enable_device_cache{true};
    bool enable_host_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};
    // Compatibility-only settlement barrier. Remote writes wait for their
    // exact backend task; HOST/DISK inserts additionally wait for the entire
    // shared BlockTree task pool. Keep disabled for normal concurrent traffic.
    bool write_cache_sync{false};

    // ---- Per-tier watermark ----
    struct TierWatermark {
        double ratio{0.0};  // watermark ratio (0.0 = disabled)
    };
    TierWatermark watermark_device;
    TierWatermark watermark_host;
    TierWatermark watermark_disk;

    // ---- Per-tier eviction policy ----
    EvictionPolicy device_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy host_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy disk_eviction_policy{EvictionPolicy::FIFO};

    // ---- Shared Store/Load/Evict task pool ----
    int task_pool_size{4};

    // ---- Shared per-rank TransferEngine task pool ----
    size_t transfer_worker_count{4};

    // ---- Cross-rank transfer timeout ----
    int host_cache_sync_timeout_ms{10000};
    int disk_cache_sync_timeout_ms{30000};

    // Total Device<->Disk staging blocks per rank, split evenly across two pools.
    size_t device_disk_staging_block_count{4};
    // Device<->Host uses descriptor batching; all other directions default to singleton batches.
    size_t max_descriptors_per_transfer_batch{8};
    size_t max_descriptors_per_non_device_host_transfer_batch{1};

    // ---- FULL prefix invariant scanner (diagnostic only) ----
    // The factory zeroes this on ranks that do not own a mutable BlockTree.
    int64_t full_prefix_scan_interval_ms{0};  // 0 = disabled, no thread

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
// Owns BlockTree, GroupSets, HostBlockPool (L2), BlockTreeDiskBlockPool (L3),
// StorageBackend, and the cache workflow collaborators injected by the factory.
// Each storage tier (Device/Host/Disk/Remote) can be independently enabled/disabled.
class BlockTreeCache {
public:
    using TierWatermark = BlockTreeCacheConfig::TierWatermark;

    BlockTreeCache(std::unique_ptr<BlockTree>               tree,
                   BlockTreeCacheConfig                     config,
                   std::shared_ptr<StorageBackend>          storage_backend,
                   std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                   std::unique_ptr<BlockTreeTaskPool>       task_pool);

    ~BlockTreeCache();
    bool init();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void                 insert(const CacheKeysType&                              cache_keys,
                                const std::vector<std::vector<GroupSetResource>>& resources,
                                Tier                                              target_tier,
                                bool                                              write_remote = true);
    // Directly reclaim up to num_blocks device blocks belonging to one group set
    // (target_tier = NONE, content dropped). Returns the number actually freed.
    int evictForGroup(size_t group_id, size_t num_blocks);

    CacheStats                                getStats() const;
    std::vector<BlockTreePoolMetricsSnapshot> poolMetricsSnapshots() const;
    void                                      reportMetrics() const;
    BlockTreeKeySnapshot                      getKeySnapshot(size_t limit) const;
    bool                                      abortPendingLoad(const std::shared_ptr<AsyncContext>& context);

    BlockIndicesType matchedBlocksForGroup(size_t                                group_id,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

    bool executeTransfer(const std::vector<TransferDescriptor>& descriptors);

    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter) {
        metrics_reporter_.setMetricsReporter(metrics_reporter);
    }

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

    const BlockTreeCacheConfig& config() const {
        return config_;
    }

private:
    void checkWatermark();
    // Caller holds mutex_.
    void onWorkflowSettledLocked(bool tree_data_mutated, bool check_watermark);

    BlockTreeCacheConfig                        config_;
    std::unique_ptr<BlockTree>                  tree_;
    std::shared_ptr<StorageBackend>             storage_backend_;
    std::unique_ptr<BlockTransferDispatcher>    transfer_dispatcher_;
    std::unique_ptr<BlockTreeTaskPool>          task_pool_;
    BlockTreeCacheMetricsReporter               metrics_reporter_;
    mutable std::mutex                          mutex_;
    BlockTreeEvictor                            evictor_;
    bool                                        initialized_{false};
    int64_t                                     mutation_version_{0};
    BlockTreeLoader                             loader_;
    BlockTreeStorer                             storer_;
    std::unique_ptr<FullPrefixInvariantScanner> full_prefix_scanner_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
