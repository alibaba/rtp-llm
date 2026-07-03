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

// Configuration for BlockTreeCache L2 (Host) and L3 (Disk) tier pools.
// Capacity is derived from KVCacheConfig as described in the design doc:
//   L2 Host blocks = memory_cache_size_mb * 1MB / block_size_bytes
//   L3 Disk blocks = memory_cache_disk_size_mb * 1MB / block_size_bytes
struct BlockTreeCacheConfig {
    // L2 Host pool configuration
    int64_t memory_cache_size_mb{0};  // 0 = disabled

    // L3 Disk pool configuration
    int64_t     memory_cache_disk_size_mb{0};  // 0 = disabled
    std::string memory_cache_disk_path;
    bool        memory_cache_disk_buffered_io{true};

    // Block size (from CacheConfig), used to compute pool block count
    size_t block_size_bytes{0};

    // Tier enable flags
    bool enable_device_cache{true};
    bool enable_memory_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};

    // Eviction thread pool
    int eviction_thread_pool_size{2};

    // Compute the number of host blocks from config.
    size_t hostBlockCount() const {
        if (memory_cache_size_mb <= 0 || block_size_bytes == 0)
            return 0;
        return static_cast<size_t>(memory_cache_size_mb) * 1024 * 1024 / block_size_bytes;
    }

    // Compute the disk pool total size in bytes.
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
    BlockTreeCache(std::unique_ptr<BlockTree>        tree,
                   std::vector<ComponentGroupPtr>    component_groups,
                   std::vector<Component>            components,
                   BlockPoolPtr                      host_pool                 = nullptr,
                   std::shared_ptr<DiskBlockPool>    disk_pool                 = nullptr,
                   int                               eviction_thread_pool_size = 2,
                   std::shared_ptr<StorageBackend>   storage_backend           = nullptr,
                   bool                              enable_device_cache       = true,
                   bool                              enable_memory_cache       = false,
                   bool                              enable_disk_cache         = false,
                   bool                              enable_remote_cache       = false,
                   std::shared_ptr<BroadcastManager> broadcast_manager         = nullptr);

    ~BlockTreeCache();

    BlockTreeMatchResult match(const CacheKeysType& cache_keys);
    void insert(TreeNode* parent, const CacheKeysType& cache_keys, const std::vector<std::vector<GroupSlot>>& slots);
    int  evict(size_t num_blocks, Tier tier = Tier::DEVICE);

    bool       isEvictable(TreeNode* node, int group_id) const;
    CacheStats getStats() const;
    void       waitForPendingTasks();

    // Phase 2: Reference counting callbacks
    using ReferenceBlocksFn = std::function<void(const std::vector<BlockIdxType>&)>;
    void setIsBlockEvictable(IsBlockEvictableFn fn);
    void setReferenceBlocksCallbacks(ReferenceBlocksFn ref_fn, ReferenceBlocksFn release_fn) {
        reference_blocks_ = std::move(ref_fn);
        release_blocks_   = std::move(release_fn);
    }

    // Release path-lock references acquired during match().
    // Must be called by the request owner when matched blocks are no longer needed.
    void releaseMatchedBlocks(const std::vector<BlockIdxType>& block_indices);

    // Per-tier watermark configuration.
    struct TierWatermark {
        double ratio{0.0};   // watermark ratio (0.0 = disabled)
        size_t capacity{0};  // total block count for this tier
    };

    // Backward-compatible: sets Device tier watermark only.
    void setWatermark(double ratio, size_t device_capacity) {
        watermark_device_ = {ratio, device_capacity};
    }

    // Per-tier watermark setter (DEVICE / HOST / DISK).
    void setTierWatermark(Tier tier, double ratio, size_t capacity) {
        switch (tier) {
            case Tier::DEVICE:
                watermark_device_ = {ratio, capacity};
                break;
            case Tier::HOST:
                watermark_host_ = {ratio, capacity};
                break;
            case Tier::DISK:
                watermark_disk_ = {ratio, capacity};
                break;
            default:
                break;
        }
    }

    // Phase 4: DeviceBufferResolver injection for real D2H copy.
    // When set, performEvictionCopy uses this resolver instead of placeholder.
    // Typical implementation wraps BlockPool::convertIndexToBuffer:
    //   auto resolver = [&pool](int layer_id, BlockIdxType block_idx) -> BlockInfo {
    //       auto bufs = pool->convertIndexToBuffer(layer_id, block_idx);
    //       return bufs.empty() ? BlockInfo{} : bufs[0];
    //   };
    void setDeviceBufferResolver(DeviceBufferResolver resolver) {
        device_buffer_resolver_ = std::move(resolver);
    }

    // Phase 2: load_back enable flag
    void setEnableLoadBack(bool enable) {
        enable_load_back_ = enable;
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

    // Pool accessors (L2 / L3 tier resources owned by BlockTreeCache)
    BlockPoolPtr hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<DiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    // Tier enable queries
    bool isDeviceCacheEnabled() const {
        return enable_device_cache_;
    }
    bool isMemoryCacheEnabled() const {
        return enable_memory_cache_;
    }
    bool isDiskCacheEnabled() const {
        return enable_disk_cache_;
    }
    bool isRemoteCacheEnabled() const {
        return enable_remote_cache_;
    }

private:
    void             performEvictionCopy(EvictionResult er);
    void             onEvictionComplete(const EvictionResult& er);
    void             cascadeEviction(TreeNode* node, int source_group_id, Tier tier);
    bool             shouldDeleteNode(const TreeNode* node) const;
    std::vector<int> allGroupIds() const;
    std::vector<int> reusableGroupIds() const;
    std::vector<int> groupsBelowPriority(int source_group_id) const;
    bool             isTierEnabled(Tier tier) const;
    void             taskStarted();
    void             taskFinished();
    void             checkWatermark();
    void             checkTierWatermark(Tier tier);

    struct LoadBackItem {
        TreeNode*                 node{nullptr};
        int                       group_id{-1};
        Tier                      source_tier{Tier::NONE};
        std::vector<BlockIdxType> allocated_device_blocks;
    };
    void performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx);

    std::unique_ptr<BlockTree>                 tree_;
    std::vector<ComponentGroupPtr>             component_groups_;
    std::vector<Component>                     components_;
    CopyEnginePtr                              copy_engine_;
    BlockPoolPtr                               host_pool_;
    std::shared_ptr<DiskBlockPool>             disk_pool_;
    std::shared_ptr<StorageBackend>            storage_backend_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::shared_ptr<BroadcastManager>          broadcast_manager_;

    // Phase 2: Reference counting & path lock callbacks
    ReferenceBlocksFn reference_blocks_;
    ReferenceBlocksFn release_blocks_;

    // Phase 2: load_back enable flag
    bool enable_load_back_{false};

    // Device block allocator for load_back
    DeviceBlockAllocator device_block_allocator_;

    // Phase 4: DeviceBufferResolver for real D2H copy
    DeviceBufferResolver device_buffer_resolver_;

    // Per-tier watermark mechanism
    TierWatermark watermark_device_;
    TierWatermark watermark_host_;
    TierWatermark watermark_disk_;

    // Tier enable flags (design doc section 2.7)
    bool enable_device_cache_{true};
    bool enable_memory_cache_{false};
    bool enable_disk_cache_{false};
    bool enable_remote_cache_{false};

    std::atomic<int>        pending_tasks_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    mutable std::mutex      mutex_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
