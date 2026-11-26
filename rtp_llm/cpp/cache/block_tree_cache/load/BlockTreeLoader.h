#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;
struct CacheStageContext;

struct BlockTreeMatchResult {
    std::optional<PrefixCachePlanPoint>                 farthest_fork;
    size_t                                              matched_device_blocks{0};
    size_t                                              matched_device_host_blocks{0};
    std::vector<MultiNodeResource>                      matched_device_resources;
    std::shared_ptr<LoadAsyncContext>                   async_context;
    std::vector<BlockTreeCacheReuseTimeMetrics>         reuse_time_metrics;
};

// Content-match authentication capability for the partial-prefix matching path.
// Scenario-agnostic: the same struct backs tail reuse and common-prefix recovery;
// the scenario only changes how the caller assembles these callbacks.
struct ContentMatchContext {
    const std::vector<int>*             tokens{nullptr};
    std::function<CacheKeyType(size_t)> endpoint_key;
    // Verify the complete source segment with the request's original hash/image
    // identity even when only a prefix of that segment matches request tokens.
    std::function<bool(size_t, const std::vector<int>&, CacheKeyType)> source_identity;
};

// Owns matching and the complete lower-tier-to-device load workflow.
// BlockTreeCache owns synchronization.
class BlockTreeLoader {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool check_watermark)>;

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
    BlockTreeMatchResult matchLocked(const CacheKeysType&               cache_keys,
                                     const std::function<bool(size_t)>& valid_prefix           = {},
                                     bool                               enable_storage_backend = true,
                                     std::optional<int64_t>             request_id             = {},
                                     bool                               skip_linear            = false,
                                     const ContentMatchContext*         partial_tail           = nullptr);
    BlockIndicesType     matchedBlocksForGroup(std::string_view                      group_tag,
                                               const std::vector<MultiNodeResource>& matched_resources) const;
    bool                 abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    void                 shutdown();

private:
    bool                 validMatch(std::vector<TreeNode*>&            path,
                                    std::vector<bool>&                 candidate_valid,
                                    const std::function<bool(size_t)>& valid_prefix,
                                    bool                               skip_linear) const;
    void                 collectReuseTimeMetrics(TreeNode*                                    node,
                                                 size_t                                       group_set_id,
                                                 uint64_t                                     access_seq,
                                                 int64_t                                      access_time_us,
                                                 std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list);
    BlockTreeMatchResult createMatchResult(std::vector<TreeNode*>&            path,
                                           const std::vector<bool>&           candidate_valid,
                                           const CacheKeysType&               cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix,
                                           bool                               enable_storage_backend,
                                           std::optional<int64_t>             request_id,
                                           bool                               skip_linear);
    StorageRequest
         makeStorageRequest(const CacheKeysType& cache_keys, size_t local_matched_blocks_num, bool skip_linear) const;
    bool commitLoad(const std::shared_ptr<LoadAsyncContext>& context);
    void abortLoadLocked(const std::vector<TransferDescriptor>&             load_descs,
                         const std::vector<bool>&                           joined_loads,
                         const std::vector<std::shared_ptr<LoadOperation>>& operations,
                         bool                                               release_transferred_refs);
    void scheduleContextSettlement(const std::shared_ptr<LoadTask>&         task,
                                   const std::shared_ptr<LoadAsyncContext>& context);
    bool validateLoadTaskLocked(const LoadTask& task) const;
    bool settleLoadLocked(LoadTask&                task,
                          bool                     aggregate_success,
                          bool&                    host_success,
                          const CacheStageContext& stage_context);

    bool changeTransferState(TreeNode*             node,
                             size_t                group_set_id,
                             GroupSetTransferState expected_state,
                             GroupSetTransferState target_state);

    BlockTree*                              tree_;
    BlockTreeEvictor&                       evictor_;
    BlockTreeTaskPool*                      task_pool_;
    BlockTreeCacheMetricsReporter&          metrics_reporter_;
    std::mutex&                             mutex_;
    const int                               host_timeout_ms_;
    const int                               disk_timeout_ms_;
    bool                                    enable_device_cache_{true};
    std::shared_ptr<StorageBackend>         storage_backend_;
    SettledFn                               settled_;
    LoadTaskRunner                          load_task_runner_;
    std::shared_ptr<LoadContextCoordinator> load_context_coordinator_;
};

}  // namespace rtp_llm
