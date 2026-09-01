#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

struct BlockTreeMatchPolicy {
    bool enable_device{true};
    bool enable_host{true};
    bool enable_disk{true};
    bool enable_remote{true};

    bool allows(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return enable_device;
            case Tier::HOST:
                return enable_host;
            case Tier::DISK:
                return enable_disk;
            case Tier::REMOTE:
                return enable_remote;
            default:
                return false;
        }
    }
};

class BlockTransferDispatcher;
class BlockTreeTaskPool;

struct BlockTreeMatchResult {
    size_t                                              matched_device_blocks{0};
    std::vector<MultiNodeResource>                      matched_device_resources;
    std::shared_ptr<LoadAsyncContext>                   async_context;
    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> reuse_time_metrics_snapshots;
};

// Owns matching and the complete lower-tier-to-device load workflow.
// BlockTreeCache owns synchronization.
class BlockTreeLoader {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool tree_data_mutated, bool check_watermark)>;

    BlockTreeLoader(BlockTree*                      tree,
                    BlockTreeEvictor&               evictor,
                    BlockTransferDispatcher*        transfer_dispatcher,
                    BlockTreeTaskPool*              task_pool,
                    BlockTreeCacheMetricsReporter&  metrics_reporter,
                    std::mutex&                     mutex,
                    int                             disk_timeout_ms,
                    int                             host_timeout_ms,
                    bool                            enable_device_cache,
                    std::shared_ptr<StorageBackend> storage_backend,
                    SettledFn                       settled);

    // The caller must hold the shared BlockTreeCache mutex.
    BlockTreeMatchResult matchLocked(const CacheKeysType& cache_keys, const BlockTreeMatchPolicy& policy);
    BlockIndicesType     matchedBlocksForGroup(size_t                                group_id,
                                               const std::vector<MultiNodeResource>& matched_resources) const;
    bool                 abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    void                 shutdown();

private:
    bool validMatch(std::vector<TreeNode*>&     path,
                    std::vector<bool>&          candidate_valid,
                    const BlockTreeMatchPolicy& policy) const;
    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot>
                         collectReuseTimeSnapshots(const std::vector<TreeNode*>& path,
                                                   size_t                        matched_device_blocks,
                                                   int64_t                       access_time_us,
                                                   const BlockTreeMatchPolicy&   policy) const;
    BlockTreeMatchResult createMatchResult(std::vector<TreeNode*>&     path,
                                           const CacheKeysType&        cache_keys,
                                           const BlockTreeMatchPolicy& policy);
    Tier                 sourceTier(const GroupSetResource& resource, const BlockTreeMatchPolicy& policy) const;
    StorageRequest       makeStorageRequest(const CacheKeysType& cache_keys, size_t local_matched_blocks_num) const;
    bool                 commitLoad(const std::shared_ptr<LoadAsyncContext>& context);
    void                 abortLoadLocked(const std::vector<TransferDescriptor>& load_descs,
                                         const std::vector<bool>&               joined_loads,
                                         size_t                                 prepared_desc_count,
                                         uint64_t                               context_id,
                                         bool                                   release_transferred_refs);
    void                 runLoadTask(const LoadTaskRunner::TaskPtr& task);
    void                 scheduleLoadSettlement(const LoadTaskRunner::TaskPtr& task, ErrorInfo error);
    bool                 settleLoadLocked(LoadTaskRunner::Task& task, bool copy_success);

    bool changeTransferState(TreeNode*             node,
                             size_t                group_set_id,
                             GroupSetTransferState expected_state,
                             GroupSetTransferState target_state);

    BlockTree*                              tree_;
    BlockTreeEvictor&                       evictor_;
    BlockTransferDispatcher*                transfer_dispatcher_;
    BlockTreeTaskPool*                      task_pool_;
    BlockTreeCacheMetricsReporter&          metrics_reporter_;
    std::mutex&                             mutex_;
    int                                     disk_timeout_ms_{0};
    int                                     host_timeout_ms_{0};
    bool                                    enable_device_cache_{true};
    std::shared_ptr<StorageBackend>         storage_backend_;
    SettledFn                               settled_;
    LoadTaskRunner                          load_task_runner_;
    LoadJoinRegistry                        load_join_registry_;
    std::shared_ptr<LoadContextCoordinator> load_context_coordinator_;
};

}  // namespace rtp_llm
