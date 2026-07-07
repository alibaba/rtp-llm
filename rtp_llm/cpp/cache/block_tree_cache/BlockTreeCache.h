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
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TransferDescriptor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

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

    // ---- Eviction thread pool ----
    int eviction_thread_pool_size{2};

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
            case Tier::DEVICE: return enable_device_cache;
            case Tier::HOST:   return enable_memory_cache;
            case Tier::DISK:   return enable_disk_cache;
            case Tier::REMOTE: return enable_remote_cache;
            default:           return false;
        }
    }

    TierWatermark watermarkForTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE: return watermark_device;
            case Tier::HOST:   return watermark_host;
            case Tier::DISK:   return watermark_disk;
            default:           return {};
        }
    }

    size_t hostBlockCount() const {
        if (memory_cache_size_mb <= 0 || block_size_bytes == 0)
            return 0;
        return static_cast<size_t>(memory_cache_size_mb) * 1024 * 1024 / block_size_bytes;
    }

    size_t diskPoolSizeBytes() const {
        if (memory_cache_disk_size_mb <= 0)
            return 0;
        return static_cast<size_t>(memory_cache_disk_size_mb) * 1024 * 1024;
    }
};

// BlockTreeCache: eviction workflow coordinator.
// Owns BlockTree, ComponentGroups, BlockPool-HOST (L2), DiskBlockPool (L3),
// CopyEngine (stateless data-movement utility), StorageBackend, thread pool.
// Each storage tier (Device/Host/Disk/Remote) can be independently enabled/disabled.
class BlockTreeCache {
public:
    using TierWatermark = BlockTreeCacheConfig::TierWatermark;

    BlockTreeCache(std::unique_ptr<BlockTree>        tree,
                   std::vector<ComponentGroupPtr>    component_groups,
                   std::vector<Component>            components,
                   BlockTreeCacheConfig              config            = {},
                   std::shared_ptr<StorageBackend>   storage_backend   = nullptr,
                   std::shared_ptr<BroadcastManager> broadcast_manager = nullptr);

    ~BlockTreeCache();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void insert(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);
    int  evict(size_t num_blocks, Tier tier = Tier::DEVICE);

    bool       isEvictable(TreeNode* node, int group_id) const;
    CacheStats getStats() const;
    void       waitForPendingTasks();

    void setIsBlockEvictable(IsBlockEvictableFn fn);

    // Release path-lock references acquired during match().
    void releaseMatchedBlocks(const std::vector<BlockIdxType>& block_indices);

    // ---- Configuration mutators (for runtime adjustment) ----
    void setWatermark(double ratio, size_t device_capacity) {
        config_.watermark_device = {ratio, device_capacity};
    }
    void setTierWatermark(Tier tier, double ratio, size_t capacity) {
        switch (tier) {
            case Tier::DEVICE: config_.watermark_device = {ratio, capacity}; break;
            case Tier::HOST:   config_.watermark_host   = {ratio, capacity}; break;
            case Tier::DISK:   config_.watermark_disk   = {ratio, capacity}; break;
            default: break;
        }
    }
    void setEnableLoadBack(bool enable) { config_.enable_load_back = enable; }

    // Phase 4: DeviceBufferResolver injection for real D2H copy.
    // When set, performEvictionCopy uses this resolver instead of placeholder.
    // Typical implementation wraps DeviceBlockPool::blockBuffers:
    //   auto resolver = [&pool](int layer_id, BlockIdxType block_idx) -> BlockInfo {
    //       auto bufs = pool->convertIndexToBuffer(layer_id, block_idx);
    //       return bufs.empty() ? BlockInfo{} : bufs[0];
    //   };
    void setDeviceBufferResolver(DeviceBufferResolver resolver) {
        device_buffer_resolver_ = std::move(resolver);
    }

    // Device block allocator for load_back (allocates GPU blocks to receive H2D data).
    using DeviceBlockAllocator = std::function<std::vector<BlockIdxType>(int component_group_id, size_t count)>;
    void setDeviceBlockAllocator(DeviceBlockAllocator fn) {
        device_block_allocator_ = std::move(fn);
    }

    // Accessors
    BlockTree* tree() const {
        return tree_.get();
    }
    const std::vector<ComponentGroupPtr>& componentGroups() const {
        return component_groups_;
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
    bool isDeviceCacheEnabled() const { return config_.enable_device_cache; }
    bool isMemoryCacheEnabled() const { return config_.enable_memory_cache; }
    bool isDiskCacheEnabled() const { return config_.enable_disk_cache; }
    bool isRemoteCacheEnabled() const { return config_.enable_remote_cache; }

    const BlockTreeCacheConfig& config() const { return config_; }

private:
    void             performEvictionCopy(EvictionResult er);
    // Eviction completion. cascade_with_copy controls cascade behavior:
    //   true  — lower-priority groups' data is copied to the next tier synchronously.
    //   false — lower-priority groups' data is released directly without copy.
    void             onEvictionComplete(const EvictionResult& er, bool cascade_with_copy);
    void             cascadeEviction(TreeNode* node, int source_group_id, Tier tier,
                                     bool cascade_with_copy);
    // Executes a tier-to-tier copy for the given group. Returns true on success.
    bool             executeTierCopy(int component_group_id, Tier source_tier, Tier target_tier,
                                     const std::vector<BlockIdxType>& source_blocks,
                                     BlockIdxType target_block);
    // Releases blocks back to the appropriate pool for the given group and tier.
    void             releaseBlocksFromPool(int component_group_id, Tier tier,
                                           const std::vector<BlockIdxType>& blocks);
    // Frees a single pre-allocated target block back to its pool.
    void             freeTargetBlock(int component_group_id, Tier target_tier, BlockIdxType block);
    // Sets the target slot data after a successful copy and adds to the corresponding heap.
    void             setTargetSlot(ComponentGroupPtr& group, GroupSlot& slot,
                                   TreeNode* node, Tier target_tier, BlockIdxType target_block);
    void             finalizeEviction(TreeNode* node);
    bool             shouldDeleteNode(const TreeNode* node) const;
    std::vector<int> allGroupIds() const;
    std::vector<int> reusableGroupIds() const;
    std::vector<int> groupsBelowPriority(int source_group_id) const;
    void             taskStarted();
    void             taskFinished();
    void             checkWatermark();
    void             checkTierWatermark(Tier tier);
    size_t           computeGroupExcess(const ComponentGroup& group, Tier tier, double ratio) const;
    void             allocateTargetBlock(EvictionResult& er);
    void             submitEviction(EvictionResult& er);
    Tier             nextLowerTier(Tier tier) const;

    // Per-group pool access helpers
    std::shared_ptr<HostBlockPool> hostPoolForGroup(int component_group_id) const;
    std::shared_ptr<DiskBlockPool> diskPoolForGroup(int component_group_id) const;

    struct LoadBackItem {
        TreeNode*                 node{nullptr};
        int                       group_id{-1};
        Tier                      source_tier{Tier::NONE};
        std::vector<BlockIdxType> allocated_device_blocks;
    };
    void performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx);

    BlockTreeCacheConfig                       config_;
    std::unique_ptr<BlockTree>                 tree_;
    std::vector<ComponentGroupPtr>             component_groups_;
    std::vector<Component>                     components_;
    CopyEnginePtr                              copy_engine_;
    std::shared_ptr<StorageBackend>            storage_backend_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::shared_ptr<BroadcastManager>          broadcast_manager_;

    // Runtime-injected collaborators
    DeviceBlockAllocator device_block_allocator_;
    DeviceBufferResolver device_buffer_resolver_;

    std::atomic<int>        pending_tasks_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    mutable std::mutex      mutex_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
