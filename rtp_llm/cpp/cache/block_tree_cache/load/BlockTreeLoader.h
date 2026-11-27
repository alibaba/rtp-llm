#pragma once

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

enum class DiskStage { None, Retry, Pending };

class BlockTransferDispatcher;
class BlockTreeTaskPool;
class HostPrefetchMatch;
struct CacheStageContext;

struct BlockTreeMatchResult {
    size_t                                              matched_device_blocks{0};
    size_t                                              matched_device_host_blocks{0};
    std::vector<MultiNodeResource>                      matched_device_resources;
    std::shared_ptr<LoadAsyncContext>                   async_context;
    std::vector<BlockTreeCacheReuseTimeMetrics>         reuse_time_metrics;
    size_t                                              matched_recoverable_blocks{0};
};

class LocalMatchAdmission {
public:
    ~LocalMatchAdmission();
    size_t readyRows() const;
    size_t matchedRows() const;

private:
    struct State;
    LocalMatchAdmission();
    std::unique_ptr<State> state_;
    friend class BlockTreeLoader;
};

// Owns matching and the complete lower-tier-to-device load workflow.
// BlockTreeCache owns synchronization.
class BlockTreeLoader {
public:
    DiskStage prefetchDisk(const BlockTreeKeys& keys, const std::vector<std::pair<size_t, size_t>>& positions,
                           std::optional<int64_t> request, std::chrono::milliseconds timeout, std::function<void(bool)> done);
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool check_watermark)>;

    BlockTreeLoader(BlockTree*                      tree,
                    BlockTreeEvictor&               evictor,
                    BlockTransferDispatcher*        transfer_dispatcher,
                    BlockTreeTaskPool*              task_pool,
                    CacheObserver&  metrics_reporter,
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
                                     bool                               skip_linear            = false);
    BlockTreeMatchResult matchLocked(const BlockTreeKeys& cache_keys,
                                     const std::function<bool(size_t)>& valid_prefix,
                                     bool enable_storage_backend,
                                     std::optional<int64_t> request_id,
                                     bool skip_linear,
                                     const std::vector<bool>& selected_groups,
                                     size_t max_load_pages,
                                     bool stage_disk_through_host = false);
    std::shared_ptr<LocalMatchAdmission> prepareHostAdmissionLocked(
        const BlockTreeKeys& keys, const std::function<bool(size_t)>& valid_prefix,
        std::optional<int64_t> request_id, const std::vector<bool>& selected_groups,
        size_t max_load_pages, bool stage_disk_through_host, size_t max_load_rows);
    std::optional<BlockTreeMatchResult> completeHostAdmissionLocked(
        LocalMatchAdmission& admission, const std::shared_ptr<HostPrefetchMatch>& warmed);
    BlockIdList          matchedBlocksForGroup(std::string_view                      group_tag,
                                               const std::vector<MultiNodeResource>& matched_resources) const;
    bool                 abortPendingLoad(const std::shared_ptr<AsyncContext>& context);
    void                 shutdown();
    using PublishHost = std::function<void(const BlockTreeKeys&, const std::vector<std::vector<GroupSetResource>>&)>;
    using RemotePrefetchDone = std::function<void(bool, std::shared_ptr<HostPrefetchMatch>)>;
    bool prefetchRemote(const BlockTreeKeys& cache_keys,
                        size_t locally_matched_rows,
                        std::function<bool(size_t)> valid_prefix,
                        std::vector<bool> selected_groups,
                        size_t max_load_pages,
                        std::optional<int64_t> request_id,
                        std::chrono::milliseconds timeout,
                        PublishHost publish,
                        std::function<bool()> still_needed,
                        std::function<void(bool)> notice,
                        RemotePrefetchDone on_ready);
    void resumeRemotePrefetchesLocked(bool capacity_changed);
    void failRemotePrefetchesLocked(const HostBlockPool& pool);

private:
    // A native Disk slot is pinned while its Host restore is in flight.
    std::map<std::pair<size_t, BlockIdxType>, std::vector<std::function<void(bool)>>> disk_waiters_;
    struct RemotePrefetch;
    std::vector<std::shared_ptr<RemotePrefetch>> remote_prefetches_;
    void advanceRemotePrefetchLocked(const std::shared_ptr<RemotePrefetch>& attempt);
    template<class Keys>
    BlockTreeMatchResult matchLockedImpl(const Keys& cache_keys,
                                         const std::function<bool(size_t)>& valid_prefix,
                                         bool enable_storage_backend,
                                         std::optional<int64_t> request_id,
                                         bool skip_linear,
                                         const std::vector<bool>& selected_groups,
                                         size_t max_load_pages,
                                         bool stage_disk_through_host,
                                         LocalMatchAdmission* admission = nullptr,
                                         size_t max_load_rows = std::numeric_limits<size_t>::max());
    bool                 validMatch(std::vector<TreeNode*>&            path,
                                    std::vector<bool>&                 candidate_valid,
                                    const std::function<bool(size_t)>& valid_prefix,
                                    bool                               skip_linear,
                                    const std::vector<bool>&           selected_groups = {}) const;
    void                 collectReuseTimeMetrics(TreeNode*                                    node,
                                                 size_t                                       group_set_id,
                                                 uint64_t                                     access_seq,
                                                 int64_t                                      access_time_us,
                                                 std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list);
    template<class Keys>
    BlockTreeMatchResult createMatchResult(std::vector<TreeNode*>&            path,
                                           const Keys&                       cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix,
                                           bool                               enable_storage_backend,
                                           std::optional<int64_t>             request_id,
                                           bool                               skip_linear,
                                           const std::vector<bool>&           selected_groups,
                                           size_t                             max_load_pages,
                                           bool                               stage_disk_through_host,
                                           LocalMatchAdmission*               admission,
                                           size_t                             max_load_rows);
    void                 appendLoadDescriptor(TreeNode* node, size_t group, size_t row,
                                             std::vector<TransferDescriptor>& descriptors, std::vector<bool>& joins);
    std::shared_ptr<LoadAsyncContext> createLoadContextLocked(const std::vector<TransferDescriptor>& descriptors,
                                                           const std::vector<bool>& joins,
                                                           size_t matched_rows, StorageRequest request,
                                                           bool use_storage);
    template<class Keys>
    StorageRequest       makeStorageRequest(const Keys& cache_keys,
                                           size_t               local_matched_blocks_num,
                                           bool                 skip_linear,
                                           const std::vector<bool>& selected_groups) const;
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
    struct JoinedLoadCompletion {
        std::shared_ptr<LoadAsyncContext> context;
        Tier                              source_tier;
        bool                              success;
    };
    bool settleLoadLocked(LoadTaskRunner::Task&              task,
                          bool                               aggregate_success,
                          bool&                              host_success,
                          std::vector<JoinedLoadCompletion>& joined_contexts,
                          const CacheStageContext&           stage_context);

    bool changeTransferState(TreeNode*             node,
                             size_t                group_set_id,
                             GroupSetTransferState expected_state,
                             GroupSetTransferState target_state);

    BlockTree*                              tree_;
    BlockTreeEvictor&                       evictor_;
    BlockTransferDispatcher*                transfer_dispatcher_;
    BlockTreeTaskPool*                      task_pool_;
    CacheObserver&          metrics_reporter_;
    std::mutex&                             mutex_;
    bool                                    enable_device_cache_{true};
    std::shared_ptr<StorageBackend>         storage_backend_;
    SettledFn                               settled_;
    LoadTaskRunner                          load_task_runner_;
    LoadJoinRegistry                        load_join_registry_;
    std::shared_ptr<LoadContextCoordinator> load_context_coordinator_;
};

}  // namespace rtp_llm
