#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackWorker.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceKVCacheGroup.h"
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
// Owns BlockTree, ComponentGroups, HostBlockPool (L2), BlockTreeDiskBlockPool (L3),
// StorageBackend, and the cache workflow collaborators injected by the factory.
// Each storage tier (Device/Host/Disk/Remote) can be independently enabled/disabled.
class BlockTreeCache {
public:
    using TierWatermark = BlockTreeCacheConfig::TierWatermark;

    // Resolves a per-tag gid (the allocator-facing id space) to its aggregated
    // component_group_id and the local device pool index within that group. A
    // component_group_id of -1 marks a NON_REUSABLE tag that is excluded from the tree.
    struct PerTagMapping {
        int component_group_id{-1};
        int local_pool_index{-1};
    };

    BlockTreeCache(std::unique_ptr<BlockTree>                    tree,
                   std::vector<ComponentGroupPtr>                component_groups,
                   std::shared_ptr<const std::vector<Component>> components,
                   BlockTreeCacheConfig                          config,
                   std::shared_ptr<StorageBackend>               storage_backend,
                   std::unique_ptr<BlockTransferDispatcher>      transfer_dispatcher,
                   std::unique_ptr<BlockCacheTaskPool>           task_pool,
                   std::vector<std::string>                      per_tag_tags,
                   std::vector<DeviceKVCacheGroupPtr>            per_tag_device_groups,
                   std::vector<PerTagMapping>                    per_tag_mapping);

    ~BlockTreeCache();
    bool init();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void insert(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);
    // Directly reclaim up to num_blocks device blocks belonging to a single component
    // group (target_tier = NONE, content dropped). Returns the number actually freed.
    int evictForTag(const std::string& tag, size_t num_blocks);

    CacheStats                                getStats() const;
    std::vector<BlockTreePoolMetricsSnapshot> poolMetricsSnapshots() const;
    void                                      reportMetrics() const;
    BlockTreeKeySnapshot                      getKeySnapshot(size_t limit) const;
    void                                      waitForPendingTasks();
    void                                      onBlocksReleased();
    bool                                      cancelLoadBack(const std::shared_ptr<AsyncContext>& context);

    // Release path-lock references acquired during match().
    void releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets);

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
    const std::vector<ComponentGroupPtr>& componentGroups() const {
        return component_groups_;
    }
    bool validateDeviceGroupTagsForComponentGroup(int component_group_id, const std::vector<std::string>& tags) const;
    // Resolve a stable declarative tag to its allocator-local group.
    DeviceKVCacheGroupPtr deviceKVGroup(const std::string& tag) const {
        for (size_t i = 0; i < per_tag_tags_.size(); ++i) {
            if (per_tag_tags_[i] == tag) {
                return i < per_tag_device_groups_.size() ? per_tag_device_groups_[i] : nullptr;
            }
        }
        return nullptr;
    }
    const std::vector<Component>& components() const {
        return *components_;
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

    bool initDeviceGroupTags();
    bool validateConfiguration() const;
    void
    insertSparse(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);
    void insertImpl(TreeNode*                                  parent,
                    const CacheKeysType&                       cache_keys,
                    const std::vector<std::vector<GroupSlot>>& slots,
                    bool                                       allow_sparse_slots);
    void drainTreeHolds();
    void checkWatermark();
    bool reclaimOneForGroup(int component_group_id, Tier tier);
    struct DeviceReleaseCredit {
        DeviceBlockPoolPtr pool;
        BlockIdxType       block{NULL_BLOCK_IDX};
    };
    bool submitEvictionLocked(EvictionMove& eviction_move, std::vector<DeviceReleaseCredit>* release_credits = nullptr);
    void reserveInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits);
    void settleInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits);
    void performEvictionCopy(const BlockTreeEvictor::EvictionPlan&   plan,
                             const std::vector<DeviceReleaseCredit>& release_credits);
    bool buildEvictionTransferBatch(const BlockTreeEvictor::EvictionPlan& plan,
                                    std::vector<TransferDescriptor>&      descriptors) const;
    int  evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const;

    bool   isNodeStructurallyMatchable(const TreeNode* node) const;
    void   prepareMatchedBlocks(const std::vector<TreeNode*>&         matched_path,
                                const std::vector<bool>&              candidate_logically_valid,
                                BlockTreeMatchResult&                 result,
                                LoadBackTicket::PendingLoadBackItems& pending_load_back_items);
    size_t computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                         const std::vector<bool>&      candidate_logically_valid) const;
    void   prepareMatchedLoadBackItem(TreeNode*                             path_node,
                                      const ComponentGroupPtr&              component_group,
                                      const GroupSlot&                      group_slot,
                                      size_t                                path_index,
                                      const std::vector<std::string>&       device_group_tags,
                                      BlockTreeMatchResult&                 result,
                                      LoadBackTicket::PendingLoadBackItems& pending_load_back_items);
    bool   changeLoadBackStateNolock(TreeNode*         node,
                                     int               group_id,
                                     SlotTransferState from,
                                     SlotTransferState to);
    bool   reserveLoadBackItems(const LoadBackTicket::PendingLoadBackItems& items);
    std::shared_ptr<AsyncContext> commitLoadBack(const LoadBackTicket& ticket);
    void                          abortLoadBack(const LoadBackTicket& ticket);
    void abortLoadBackNolock(const LoadBackTicket::PendingLoadBackItems& items, size_t prepared_item_count);
    void runLoadBackTask(const LoadBackWorker::TaskPtr& task);
    bool settleLoadBackNolock(LoadBackWorker::Task& task, bool copy_success);

    BlockTreeCacheConfig                          config_;
    std::unique_ptr<BlockTree>                    tree_;
    std::vector<ComponentGroupPtr>                component_groups_;
    std::shared_ptr<const std::vector<Component>> components_;
    // Stable CacheTopology tags, aligned with allocator-local group storage.
    const std::vector<std::string>     per_tag_tags_;
    std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups_;
    // Per-tag gid -> (component_group_id, local_pool_index).
    std::vector<PerTagMapping> per_tag_mapping_;
    // component_group_id -> local_pool_index -> stable declarative tag.
    std::vector<std::vector<std::string>>    device_group_tags_;
    LoadBackWorker                          load_back_worker_;
    std::shared_ptr<LoadBackTicketRegistry>  load_back_ticket_registry_;
    std::shared_ptr<StorageBackend>          storage_backend_;
    std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher_;
    std::unique_ptr<BlockCacheTaskPool>      task_pool_;
    BlockTreeCacheMetricsReporter            metrics_reporter_;
    BlockTreeEvictor                         evictor_;
    bool                                     initialized_{false};

    mutable std::mutex mutex_;
    // Protected by mutex_. Credits remain reserved from async queue acceptance
    // until the matching plan completes or rolls back.
    std::unordered_map<DeviceBlockPoolPtr, size_t> in_flight_device_release_credits_;
    int64_t                                        mutation_version_{0};
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
