#pragma once

#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;

struct BlockTreeMatchResult {
    std::optional<PrefixCachePlanPoint>                farthest_fork;
    std::optional<PrefixCacheMatchObservation>          match_observation;
    size_t                                              matched_device_blocks{0};
    size_t                                              matched_device_endpoint{0};
    bool                                                precise_local{false};
    bool                                                short_leaf{false};
    std::vector<MultiNodeResource>                      matched_device_resources;
    // Parallel to matched_device_resources; copied under the tree lock so no
    // borrowed TreeNode pointer escapes into allocator materialization.
    std::vector<std::vector<size_t>>                   precise_resource_endpoints;
    std::shared_ptr<struct PinnedLocalFallback>         local_fallback;
    std::shared_ptr<LoadAsyncContext>                   async_context;
    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> reuse_time_metrics_snapshots;
};

struct PinnedLocalFallback {
    size_t                              endpoint{0};
    size_t                              path_nodes{0};
    std::vector<GroupSetPtr>            group_sets;
    std::vector<MultiNodeResource>      resources;
    std::vector<std::vector<size_t>>   resource_endpoints;
    bool                                transferred_to_request{false};

    ~PinnedLocalFallback();
};

struct PartialTailMatchRequest {
    const std::vector<int>*                     tokens{nullptr};
    std::function<CacheKeyType(size_t)>        endpoint_key;
    std::function<bool(size_t)>                valid_endpoint;
    // Verify the complete source segment with the request's original hash/image
    // identity even when only a prefix of that segment matches request tokens.
    std::function<bool(size_t, const std::vector<int>&, CacheKeyType)> source_identity;
    // Farthest legal endpoint at or before upper (0 means no reusable prefix).
    std::function<size_t(size_t)>              preceding_valid_endpoint;
    size_t                                     reuse_unit_tokens{1};
    // Farthest endpoint the unchanged complete-key backend request can reach.
    size_t                                     remote_ceiling{0};
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
    BlockTreeMatchResult matchLocked(const CacheKeysType& cache_keys,
                                 const std::function<bool(size_t)>& valid_prefix,
                                 const std::vector<int>* request_tokens,
                                 const PartialTailMatchRequest* partial_tail = nullptr);
    BlockIndicesType     matchedBlocksForGroup(std::string_view                      group_tag,
                                               const std::vector<MultiNodeResource>& matched_resources) const;
    bool                 abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    void                 shutdown();

private:
    std::optional<BlockTreeMatchResult> preciseLocalMatchLocked(
        const PartialTailMatchRequest& request, const std::vector<BlockTreeContentRange>& ranges);
    bool validMatch(std::vector<TreeNode*>& path,
                    std::vector<bool>&       candidate_valid,
                    std::vector<std::vector<bool>>& group_valid,
                    const std::function<bool(size_t)>& valid_prefix) const;
    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> collectReuseTimeSnapshots(const std::vector<TreeNode*>& path,
                                                                                  size_t  matched_device_blocks,
                                                                                  int64_t access_time_us) const;
    BlockTreeMatchResult createMatchResult(std::vector<TreeNode*>& path,
                                           const CacheKeysType&      cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix,
                                           const std::vector<int>*   request_tokens);
    StorageRequest       makeStorageRequest(const CacheKeysType& cache_keys, size_t local_matched_blocks_num) const;
    bool                 commitLoad(const std::shared_ptr<LoadAsyncContext>& context);
    void                 abortLoadLocked(const std::vector<TransferDescriptor>& load_descs,
                                         const std::vector<bool>&               joined_loads,
                                         size_t                                 prepared_desc_count,
                                         uint64_t                               context_id,
                                         bool                                   release_transferred_refs);
    void                 runLoadTask(const LoadTaskRunner::TaskPtr& task);
    void                 scheduleContextSettlement(const LoadTaskRunner::TaskPtr&           task,
                                                   const std::shared_ptr<LoadAsyncContext>& context);
    bool                 validateLoadTaskLocked(const LoadTaskRunner::Task& task) const;
    bool                 settleLoadLocked(LoadTaskRunner::Task&                           task,
                                          bool                                            aggregate_success,
                                          std::vector<std::shared_ptr<LoadAsyncContext>>& joined_contexts);

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
    bool                                    enable_device_cache_{true};
    std::shared_ptr<StorageBackend>         storage_backend_;
    SettledFn                               settled_;
    LoadTaskRunner                          load_task_runner_;
    LoadJoinRegistry                        load_join_registry_;
    std::shared_ptr<LoadContextCoordinator> load_context_coordinator_;
};

}  // namespace rtp_llm
