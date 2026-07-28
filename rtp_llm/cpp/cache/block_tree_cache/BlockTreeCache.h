#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackWorker.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockCacheTaskPool;
class BlockTransferDispatcher;
class HybridKVCacheAllocator;
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

    // ---- Load-back control ----
    bool enable_load_back{false};

    // ---- Reverse (leaf) cascade eviction control ----
    // When true, evicting any group on a leaf node cascades to all other groups,
    // regardless of group priority.
    bool enable_reverse_eviction{false};

    // ---- Per-tier eviction policy ----
    EvictionPolicy device_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy host_eviction_policy{EvictionPolicy::LRU};
    EvictionPolicy disk_eviction_policy{EvictionPolicy::FIFO};

    // ---- Eviction thread pool ----
    int eviction_thread_pool_size{2};

    // ---- Cross-rank transfer timeout ----
    int memory_cache_sync_timeout_ms{10000};
    int memory_cache_disk_sync_timeout_ms{30000};

    // ---- L2 Host pool sizing ----
    int64_t memory_cache_size_mb{0};  // 0 = disabled

    // ---- L3 Disk pool sizing ----
    int64_t     memory_cache_disk_size_mb{0};  // 0 = disabled
    std::string memory_cache_disk_path;
    bool        memory_cache_disk_buffered_io{true};

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
                   std::vector<GroupSetPtr>                 group_sets,
                   BlockTreeCacheConfig                     config,
                   std::shared_ptr<StorageBackend>          storage_backend,
                   std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher,
                   std::unique_ptr<BlockCacheTaskPool>      task_pool);

    ~BlockTreeCache();
    bool init();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void
    insert(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSetResource>>& slots);
    // Directly reclaim up to num_blocks device blocks belonging to one group set
    // (target_tier = NONE, content dropped). Returns the number actually freed.
    int evictForTag(const std::string& tag, size_t num_blocks);

    CacheStats                                getStats() const;
    std::vector<BlockTreePoolMetricsSnapshot> poolMetricsSnapshots() const;
    void                                      reportMetrics() const;
    BlockTreeKeySnapshot                      getKeySnapshot(size_t limit) const;
    void                                      waitForPendingTasks();
    void                                      onBlocksReleased();
    bool                                      cancelLoadBack(const std::shared_ptr<AsyncContext>& context);

    // Release path-lock references acquired during match().
    void releaseMatchedResources(const std::vector<MultiNodeResource>& resources);

    BlockIndicesType matchedBlocksForGroup(size_t                                group_id,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

    TransferStatus executeTransfer(const TransferDescriptor& descriptor);

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
    void setEnableLoadBack(bool enable) {
        config_.enable_load_back = enable;
    }

    // Accessors
    BlockTree* tree() const {
        return tree_.get();
    }
    const std::vector<GroupSetPtr>& groupSets() const {
        return group_sets_;
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
    friend class HybridKVCacheAllocator;

    bool initializeConfiguration();
    void insertSparse(TreeNode*                                         parent,
                      const CacheKeysType&                              cache_keys,
                      const std::vector<std::vector<GroupSetResource>>& slots);
    void insertImpl(TreeNode*                                         parent,
                    const CacheKeysType&                              cache_keys,
                    const std::vector<std::vector<GroupSetResource>>& slots,
                    bool                                              allow_sparse_slots);
    void drainTreeHolds();
    void checkWatermark();
    bool reclaimOneForGroup(size_t group_set_id, Tier tier);
    struct DeviceReleaseCredit {
        DeviceBlockPoolPtr pool;
        BlockIdxType       block{NULL_BLOCK_IDX};
    };
    bool submitEvictionLocked(EvictionMove& eviction_move, std::vector<DeviceReleaseCredit>* release_credits = nullptr);
    void reserveInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits);
    void settleInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits) noexcept;
    void performEvictionCopy(const BlockTreeEvictor::EvictionPlan&   plan,
                             const std::vector<DeviceReleaseCredit>& release_credits);
    bool buildEvictionTransferBatch(const BlockTreeEvictor::EvictionPlan& plan,
                                    std::vector<TransferDescriptor>&      descriptors) const;
    int  evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const;

    void                            validateMatchedResource(const MultiNodeResource& resource) const;
    void                            prepareMatchedBlocks(const std::vector<TreeNode*>&         matched_path,
                                                         const std::vector<bool>&              candidate_logically_valid,
                                                         BlockTreeMatchResult&                 result,
                                                         LoadBackTicket::PendingLoadBackItems& pending_load_back_items);
    size_t                          computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                                                  const std::vector<bool>&      candidate_logically_valid) const;
    void                            prepareMatchedLoadBackItem(TreeNode*                             path_node,
                                                               const GroupSetPtr&                    group_set,
                                                               const GroupSetResource&               group_set_resource,
                                                               size_t                                path_index,
                                                               BlockTreeMatchResult&                 result,
                                                               LoadBackTicket::PendingLoadBackItems& pending_load_back_items);
    std::shared_ptr<LoadBackTicket> prepareLoadBackTicket(LoadBackTicket::PendingLoadBackItems& items,
                                                          size_t                                logical_matched_blocks);
    bool                            prepareJoinedLoadBackItem(LoadBackTicket::PendingLoadBackItem&         item,
                                                              const std::shared_ptr<LoadBackAsyncContext>& context);
    bool                            reserveLoadBackItems(const LoadBackTicket::PendingLoadBackItems& items);
    std::shared_ptr<AsyncContext>   commitLoadBack(const LoadBackTicket& ticket);
    void                            abortLoadBack(const LoadBackTicket& ticket);
    void                            abortLoadBackUnsafe(const LoadBackTicket::PendingLoadBackItems&  items,
                                                        size_t                                       prepared_item_count,
                                                        const std::shared_ptr<LoadBackAsyncContext>& context);
    void                            runLoadBackTask(const LoadBackWorker::TaskPtr& task);
    bool                            settleLoadBackNolock(LoadBackWorker::Task& task, bool copy_success);

    struct GroupLocation {
        size_t group_set_id{0};
        size_t local_group_index{0};
    };

    BlockTreeCacheConfig       config_;
    std::unique_ptr<BlockTree> tree_;
    std::vector<GroupSetPtr>   group_sets_;
    // Reusable topology group_id -> GroupSet/local position. Non-reusable groups
    // never enter this index or the BlockTree resource space.
    std::unordered_map<size_t, GroupLocation> reusable_group_locations_;
    LoadBackWorker                            load_back_worker_;
    std::shared_ptr<LoadBackTicketRegistry>   load_back_ticket_registry_;
    std::shared_ptr<StorageBackend>           storage_backend_;
    std::unique_ptr<BlockTransferDispatcher>  transfer_dispatcher_;
    std::unique_ptr<BlockCacheTaskPool>       task_pool_;
    BlockTreeCacheMetricsReporter             metrics_reporter_;
    BlockTreeEvictor                          evictor_;
    bool                                      initialized_{false};

    mutable std::mutex mutex_;
    // Protected by mutex_. Credits remain reserved from async queue acceptance
    // until the matching plan completes or rolls back.
    std::unordered_map<DeviceBlockPoolPtr, size_t> in_flight_device_release_credits_;
    int64_t                                        mutation_version_{0};
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
