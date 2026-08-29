#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;

class BlockTreeStorer {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool tree_data_mutated, bool check_watermark)>;

    BlockTreeStorer(BlockTree*                      tree,
                    BlockTreeEvictor&               evictor,
                    BlockTransferDispatcher*        transfer_dispatcher,
                    BlockTreeTaskPool*              task_pool,
                    BlockTreeCacheMetricsReporter&  metrics_reporter,
                    std::mutex&                     mutex,
                    int                             host_timeout_ms,
                    int                             disk_timeout_ms,
                    std::shared_ptr<StorageBackend> storage_backend,
                    SettledFn                       settled);

    StorageWriteTask storeLocked(const CacheKeysType&                              cache_keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources,
                                 Tier                                              target_tier,
                                 bool                                              write_remote);
    void             stopAdmissionLocked();

private:
    using StoreTask    = StoreTaskRunner::Task;
    using StoreTaskPtr = std::shared_ptr<StoreTask>;

    StorageWriteTask publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                         const std::vector<std::vector<GroupSetResource>>& resources);
    StorageRequest   makeStorageRequest(const CacheKeysType&                              cache_keys,
                                        const std::vector<std::vector<GroupSetResource>>& resources) const;
    void             submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                           const std::vector<std::vector<GroupSetResource>>& resources,
                                           Tier                                              target_tier);
    void             runStoreTask(const StoreTaskPtr& task);
    void             scheduleStoreSettlement(const StoreTaskPtr& task, ErrorInfo error);
    void             settleTask(const StoreTask& task, bool copy_success);
    size_t           settleLocked(const StoreTask& task, bool publish);

    BlockTree*                      tree_;
    BlockTreeEvictor&               evictor_;
    BlockTransferDispatcher*        transfer_dispatcher_;
    BlockTreeTaskPool*              task_pool_;
    BlockTreeCacheMetricsReporter&  metrics_reporter_;
    StoreTaskRunner                 store_task_runner_;
    std::mutex&                     mutex_;
    int                             host_timeout_ms_{0};
    int                             disk_timeout_ms_{0};
    std::shared_ptr<StorageBackend> storage_backend_;
    SettledFn                       settled_;
    std::atomic<bool>               stopping_{false};
};

}  // namespace rtp_llm
