#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class BlockTransferDispatcher;
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
struct DeviceBlockDebugInfo {
    size_t                    group_id{0};
    size_t                    group_set_id{0};
    size_t                    member_group_id{0};
    BlockIdxType              block_id{NULL_BLOCK_IDX};
    uintptr_t                 node_address{0};
    CacheKeyType              cache_key{0};
    GroupSetTransferState     transfer_state{GroupSetTransferState::IDLE};
    std::vector<BlockIdxType> device_blocks;
};

// Unified configuration for BlockTreeCache behavior and pool sizing.
struct BlockTreeCacheConfig {
    // ---- Tier enable flags ----
    bool enable_device_cache{true};
    bool enable_memory_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};

    // ---- Per-tier watermark ----
    struct TierWatermark {
        double ratio{0.0};   // watermark ratio (0.0 = disabled)
        size_t capacity{0};  // total block count (used for legacy DEVICE mode only)
    };
    TierWatermark watermark_device;
    TierWatermark watermark_host;
    TierWatermark watermark_disk;

    // Absolute device headroom. Applied after request references are released;
    // unlike ratio watermarks this maps directly to device_cache_min_free_blocks.
    size_t device_min_free_blocks{0};

    // ---- Reverse (leaf) cascade eviction control ----
    // When true, evicting any group set on a leaf node cascades to all other
    // group sets, regardless of group priority.
    bool enable_reverse_eviction{false};

    // ---- Per-tier eviction policy ----
    EvictionPolicy device_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy host_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy disk_eviction_policy{EvictionPolicy::FIFO};

    // ---- Shared Store/Load/Evict task pool ----
    int task_pool_size{4};

    // ---- Cross-rank transfer timeout ----
    int memory_cache_sync_timeout_ms{10000};
    int memory_cache_disk_sync_timeout_ms{30000};

    // ---- L2 Host pool sizing ----
    int64_t memory_cache_size_mb{0};  // 0 = disabled

    // ---- L3 Disk pool sizing ----
    int64_t     memory_cache_disk_size_mb{0};  // 0 = disabled
    std::string memory_cache_disk_path;
    bool        memory_cache_disk_buffered_io{true};

    // Usable transient Device<->Disk staging blocks per rank.
    size_t device_disk_staging_block_count{4};

    // Block size (from CacheConfig), used to compute pool block count
    size_t block_size_bytes{0};

    // ---- Query helpers ----
    bool isTierEnabled(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return enable_device_cache;
            case Tier::HOST:
                return enable_memory_cache;
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
                                Tier                                              target_tier);
    // Directly reclaim up to num_blocks device blocks belonging to one group set
    // (target_tier = NONE, content dropped). Returns the number actually freed.
    int evictForGroup(size_t group_id, size_t num_blocks);

    CacheStats                                getStats() const;
    std::vector<BlockTreePoolMetricsSnapshot> poolMetricsSnapshots() const;
    void                                      reportMetrics() const;
    BlockTreeKeySnapshot                      getKeySnapshot(size_t limit) const;
    bool getDeviceBlockDebugInfo(size_t group_id, BlockIdxType block_id, DeviceBlockDebugInfo& debug_info) const;
    void                                      waitForPendingTasks();
    void                                      onBlocksReleased(const std::vector<BlockReleaseReceipt>& receipts);
    bool                                      cancelLoad(const std::shared_ptr<AsyncContext>& context);

    // Release path-lock references acquired during match().
    void releaseMatchedResources(const std::vector<MultiNodeResource>& resources);

    BlockIndicesType matchedBlocksForGroup(size_t                                group_id,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

    bool executeTransfer(const TransferDescriptor& descriptor);

    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter) {
        metrics_reporter_.setMetricsReporter(metrics_reporter);
    }

    // ---- Configuration mutators (for runtime adjustment) ----
    void setTierWatermark(Tier tier, double ratio, size_t capacity) {
        switch (tier) {
            case Tier::DEVICE:
                config_.watermark_device = {ratio, capacity};
                break;
            case Tier::HOST:
                config_.watermark_host = {ratio, capacity};
                break;
            case Tier::DISK:
                config_.watermark_disk = {ratio, capacity};
                break;
            default:
                break;
        }
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
    bool isMemoryCacheEnabled() const {
        return config_.enable_memory_cache;
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
    void reserveInFlightDeviceReleaseCreditsLocked(const std::vector<EvictionReleaseCredit>& release_credits);
    void settleInFlightDeviceReleaseCreditsLocked(const std::vector<EvictionReleaseCredit>& release_credits) noexcept;

    BlockTreeCacheConfig                     config_;
    std::unique_ptr<BlockTree>               tree_;
    std::shared_ptr<StorageBackend>          storage_backend_;
    std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher_;
    std::unique_ptr<BlockTreeTaskPool>       task_pool_;
    BlockTreeCacheMetricsReporter            metrics_reporter_;
    mutable std::mutex                       mutex_;
    BlockTreeEvictor                         evictor_;
    bool                                     initialized_{false};
    // Protected by mutex_. Credits remain reserved from async queue acceptance
    // until the matching plan completes or rolls back.
    std::unordered_map<DeviceBlockPoolPtr, size_t> in_flight_device_release_credits_;
    int64_t                                        mutation_version_{0};
    BlockTreeLoader                                loader_;
    BlockTreeStorer                                storer_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
