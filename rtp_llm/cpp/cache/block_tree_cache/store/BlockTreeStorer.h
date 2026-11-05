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
    using SettledFn = std::function<void(bool check_watermark)>;

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
                                 bool                                              is_resident,
                                 size_t&                                           resident_prefix_length,
                                 const std::vector<std::vector<int>>*              token_segments = nullptr,
                                 PrefixCachePlanState*                             prefix_plan_state = nullptr,
                                 const std::vector<size_t>*                         token_ends = nullptr,
                  std::optional<int64_t> observation_request_id = std::nullopt,
                  bool internal_fork = false,
                  bool* ready_after_insert = nullptr);
    void             stopAdmissionLocked();
    void observePublicationSkipped(std::optional<int64_t> request_id,
                                   Tier tier,
                                   const char* reason,
                                   std::optional<size_t> internal_fork_pos = std::nullopt);

private:
    using StoreTask    = StoreTaskRunner::Task;
    using StoreTaskPtr = std::shared_ptr<StoreTask>;

    void             publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                         const std::vector<std::vector<GroupSetResource>>& resources,
                                         bool                                              is_resident,
                                         size_t&                                           resident_prefix_length,
                                         const std::vector<std::vector<int>>*              token_segments,
                                         PrefixCachePlanState*                             prefix_plan_state,
                                         const std::vector<size_t>*                         token_ends,
                                         std::optional<int64_t> observation_request_id,
                                         bool internal_fork,
                                         bool* ready_after_insert);
    StorageRequest   makeStorageRequest(const CacheKeysType&                              cache_keys,
                                        const std::vector<std::vector<GroupSetResource>>& resources,
                                        const std::vector<size_t>*                         token_ends = nullptr) const;
    void             submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                           const std::vector<std::vector<GroupSetResource>>& resources,
                                           Tier                                              target_tier,
                                           const std::vector<std::vector<int>>*              token_segments,
                                           PrefixCachePlanState*                             prefix_plan_state,
                                           std::optional<int64_t> observation_request_id);
    void             runStoreTask(const StoreTaskPtr& task);
    void             scheduleStoreSettlement(const StoreTaskPtr& task,
                                             ErrorInfo            error,
                                             const char*          failure_reason);
    void             settleTask(StoreTask& task, bool copy_success, const char* failure_reason);
    size_t           settleLocked(StoreTask& task, bool publish, const char* failure_reason);

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
