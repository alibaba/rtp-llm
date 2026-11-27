#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

class FunctionRequestPB;
class FunctionResponsePB;

namespace rtp_llm {
class BlockTreeReclaimer;

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
    int64_t                   version{-1};
    std::vector<CacheKeyType> keys;
};
// BlockTreeCache: eviction workflow coordinator.
// Owns BlockTree, GroupSets, HostBlockPool (L2), DiskBlockPool (L3),
// StorageBackend, and the cache workflow collaborators injected by the factory.
// Local tiers require DISK -> HOST -> DEVICE; remote storage is independent.
class BlockTreeCache {
public:
    BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                   BlockTreeCacheConfig                           config,
                   std::shared_ptr<StorageBackend>                storage_backend,
                   std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                   std::unique_ptr<BlockTreeTaskPool>             task_pool,
                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                   std::shared_ptr<BlockTreeReclaimer>            reclaimer);

    ~BlockTreeCache();
    bool                  init();
    std::function<void()> capacityChangeCallback() const;
    BlockTreeReclaimer& reclaimer() const { return *reclaimer_; }
    void stopBackgroundTasks();
    // Reject pending loads and wait for allocator callbacks before its members are destroyed.
    void shutdownLoads();

    BlockTreeMatchResult match(const CacheKeysType&               cache_keys,
                               const std::function<bool(size_t)>& valid_prefix           = {},
                               bool                               enable_storage_backend = true,
                               std::optional<int64_t>             request_id             = {},
                               bool                               skip_linear            = false);
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
                  size_t                                            remote_reuse_end   = 0);
    // Evict on demand to free num_blocks in the DEVICE pool for this group
    // (target_tier = NONE, content dropped). Returns the number actually freed.
    // Skips plans containing device blocks that would remain referenced after dropping.
    int evictForGroup(std::string_view group_tag, size_t num_blocks);

    CacheStats                    getStats() const;
    std::vector<CachePoolMetrics> poolMetrics() const;
    void                          reportMetrics() const;
    BlockTreeKeySnapshot          getKeySnapshot(int64_t latest_version = -1) const;
    void setEventPublisher(KVCacheEventPublisherPtr publisher, const std::vector<std::string>& required_group_tags);
    KVCacheSnapshot logicalCacheSnapshot() const;
    bool            abortPendingLoad(const std::shared_ptr<AsyncContext>& context);

    BlockIndicesType matchedBlocksForGroup(std::string_view                      group_tag,
                                           const std::vector<MultiNodeResource>& matched_resources) const;

    TransferResult                     executeTransfer(TransferTask task);
    static bool
    executeFunction(BlockTreeCache* cache, int rank, const FunctionRequestPB& request, FunctionResponsePB& response,
                    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max(),
                    bool is_pp = false);
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
    struct CapacityCallbackState;

    void scheduleCapacityCheck();

    BlockTreeCacheConfig                           config_;
    std::unique_ptr<BlockTree>                     tree_;
    std::shared_ptr<StorageBackend>                storage_backend_;
    std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher_;
    std::unique_ptr<BlockTreeTaskPool>             task_pool_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
    mutable std::mutex                             mutex_;
    BlockTreeEvictor                               evictor_;
    bool                                           initialized_{false};
    std::atomic<bool>                              stopping_{false};
    BlockTreeLoader                                loader_;
    BlockTreeStorer                                storer_;
    std::unique_ptr<FullPrefixInvariantScanner>    full_prefix_scanner_;
    std::shared_ptr<BlockTreeReclaimer> reclaimer_;
    std::shared_ptr<CapacityCallbackState>         capacity_callback_state_;
};

using BlockTreeCachePtr = std::shared_ptr<BlockTreeCache>;

}  // namespace rtp_llm
