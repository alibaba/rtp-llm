#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "autil/LambdaWorkItem.h"
#include "autil/LockFreeThreadPool.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceKVCacheGroup.h"

class MemoryOperationRequestPB;

namespace rtp_llm {

class BroadcastManager;
struct CacheStats {
    size_t tree_node_count{0};
    size_t device_heap_total_size{0};
    size_t host_heap_total_size{0};
    size_t disk_heap_total_size{0};
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

    // ---- Load-back control ----
    bool enable_load_back{false};

    // ---- Reverse (leaf) cascade eviction control ----
    // When true, evicting any group on a leaf node cascades to all other groups,
    // regardless of group priority.
    bool enable_reverse_eviction{false};

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
// Owns BlockTree, ComponentGroups, HostBlockPool (L2), DiskBlockPool (L3),
// CopyEngine (schema-aware data-movement utility), StorageBackend, thread pool.
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

    BlockTreeCache(std::unique_ptr<BlockTree>         tree,
                   std::vector<ComponentGroupPtr>     component_groups,
                   std::vector<Component>             components,
                   BlockTreeCacheConfig               config,
                   std::shared_ptr<StorageBackend>    storage_backend,
                   std::shared_ptr<BroadcastManager>  broadcast_manager,
                   std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups,
                   std::vector<PerTagMapping>         per_tag_mapping);

    ~BlockTreeCache();
    bool init();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void insert(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);
    // reclaimBlocks: directly reclaim (drop) blocks at the given tier, no demotion, no copy.
    // Block content is discarded rather than moved down to a lower tier.
    int reclaimBlocks(size_t num_blocks, Tier tier = Tier::DEVICE);

    // Number of device blocks currently held only by the cache (refcount == 1) for one
    // component group, i.e. blocks that can be reclaimed without evicting live requests.
    size_t evictableBlocksNum(int component_group_id) const;

    // Directly reclaim up to num_blocks device blocks belonging to a single component
    // group (target_tier = NONE, content dropped). Returns the number actually freed.
    int evictForGroup(int component_group_id, size_t num_blocks);

    CacheStats getStats() const;
    void       waitForPendingTasks();

    // Release path-lock references acquired during match().
    void releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets);

    CopyStatus executeTransfer(const TransferDescriptor& descriptor);

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
    // Allocator-facing per-tag DeviceKVCacheGroup, including NON_REUSABLE tags.
    DeviceKVCacheGroupPtr deviceKVGroup(int gid) const {
        if (gid < 0 || static_cast<size_t>(gid) >= per_tag_device_groups_.size()) {
            return nullptr;
        }
        return per_tag_device_groups_[static_cast<size_t>(gid)];
    }
    const std::vector<Component>& components() const {
        return components_;
    }
    CopyEnginePtr copyEngine() const {
        return copy_engine_;
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

    const BlockTreeCacheConfig& config() const {
        return config_;
    }

private:
    void taskStarted();
    void taskFinished();
    void checkWatermark();
    bool submitEvictionLocked(EvictionMove& eviction_move);
    void performEvictionCopy(const BlockTreeEvictor::EvictionPlan& plan);
    bool buildEvictionTransferRequest(const BlockTreeEvictor::EvictionPlan& plan,
                                      ::MemoryOperationRequestPB&           request) const;
    int  evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const;
    bool broadcastTransfer(const ::MemoryOperationRequestPB& request, int timeout_ms) const;

    // Per-group pool access helpers
    std::shared_ptr<HostBlockPool> hostPoolForGroup(int component_group_id) const;
    std::shared_ptr<DiskBlockPool> diskPoolForGroup(int component_group_id) const;

    struct LoadBackItem {
        TreeNode*                 node{nullptr};
        int                       group_id{-1};
        Tier                      source_tier{Tier::NONE};
        std::vector<BlockIdxType> source_blocks;
        std::vector<BlockIdxType> target_device_blocks;
    };
    void prepareMatchedBlocks(const std::vector<TreeNode*>& matched_path, BlockTreeMatchResult& result);
    void prepareMatchedLoadBackItem(TreeNode*                path_node,
                                    const ComponentGroupPtr& component_group,
                                    const GroupSlot&         group_slot,
                                    BlockTreeMatchResult&    result);

    std::shared_ptr<AsyncContext> commitLoadBack(const std::vector<PendingLoadBackItem>& items);
    void                          abortLoadBack(const std::vector<PendingLoadBackItem>& items);
    bool executeLoadBackTransferBatch(const std::vector<TransferDescriptor>& descriptors, int timeout_ms);
    void performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx);

    // Resolve an allocator-facing per-tag gid to its component_group_id.
    int resolveComponentGroupId(int gid) const {
        if (gid < 0 || static_cast<size_t>(gid) >= per_tag_mapping_.size()) {
            return -1;
        }
        return per_tag_mapping_[static_cast<size_t>(gid)].component_group_id;
    }

    BlockTreeCacheConfig           config_;
    std::unique_ptr<BlockTree>     tree_;
    std::vector<ComponentGroupPtr> component_groups_;
    std::vector<Component>         components_;
    // Per-tag gid -> DeviceKVCacheGroup (covers NON_REUSABLE tags too).
    std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups_;
    // Per-tag gid -> (component_group_id, local_pool_index).
    std::vector<PerTagMapping>                 per_tag_mapping_;
    CopyEnginePtr                              copy_engine_;
    std::shared_ptr<StorageBackend>            storage_backend_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::shared_ptr<BroadcastManager>          broadcast_manager_;
    BlockTreeEvictor                           evictor_;

    std::atomic<int>        pending_tasks_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    mutable std::mutex      mutex_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
