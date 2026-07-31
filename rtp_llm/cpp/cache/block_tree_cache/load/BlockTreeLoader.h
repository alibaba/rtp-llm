#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;

// Owns the complete lower-tier-to-device load workflow. BlockTreeCache owns
// synchronization and tree matching; BlockTreeLoader owns load planning,
// context coordination, transfer execution, state transitions, and settlement.
class BlockTreeLoader {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool tree_data_mutated, bool check_watermark)>;

    BlockTreeLoader(const std::vector<GroupSetPtr>& group_sets,
                    BlockTreeEvictor&               evictor,
                    BlockTransferDispatcher*        transfer_dispatcher,
                    BlockTreeTaskPool*              task_pool,
                    BlockTreeCacheMetricsReporter&  metrics_reporter,
                    std::mutex&                     mutex,
                    int                             disk_timeout_ms,
                    int                             host_timeout_ms,
                    bool                            enable_device_cache,
                    SettledFn                       settled);

    // The caller must hold the shared BlockTreeCache mutex.
    void prepareLoadLocked(const std::vector<TreeNode*>& matched_path, BlockTreeMatchResult& result);
    // The caller must hold the shared BlockTreeCache mutex.
    bool cancelLoadLocked(const std::shared_ptr<AsyncContext>& context);
    void shutdown();

private:
    void                              prepareMatchedLoadDescriptor(TreeNode*                           path_node,
                                                             const GroupSetPtr&                  group_set,
                                                             const GroupSetResource&             group_set_resource,
                                                             size_t                              path_index,
                                                             BlockTreeMatchResult&               result,
                                                             std::vector<TransferDescriptor>&    pending_load_descs,
                                                             std::vector<bool>&                  joined_load);
    std::shared_ptr<LoadAsyncContext> prepareLoadContext(std::vector<TransferDescriptor>& load_descs,
                                                         const std::vector<bool>&            joined_load,
                                                         size_t                              logical_matched_blocks);
    bool                              prepareJoinedLoadDescriptor(TransferDescriptor& desc);
    bool reserveLoadDescriptors(const std::vector<TransferDescriptor>& load_descs, const std::vector<bool>& joined_load);
    bool commitLoad(const std::shared_ptr<LoadAsyncContext>& context);
    void abortLoad(LoadAsyncContext& context);
    void abortLoadNolock(const std::vector<TransferDescriptor>& load_descs,
                         const std::vector<bool>&                  joined_load,
                         size_t                                    prepared_desc_count,
                         uint64_t                                  context_id);
    void runLoadTask(const LoadTaskRunner::TaskPtr& task);
    bool settleLoadLocked(LoadTaskRunner::Task& task, bool copy_success);

    bool changeTransferState(TreeNode*             node,
                             size_t                group_set_id,
                             GroupSetTransferState expected_state,
                             GroupSetTransferState target_state);

    const std::vector<GroupSetPtr>&         group_sets_;
    BlockTreeEvictor&                       evictor_;
    BlockTransferDispatcher*                transfer_dispatcher_;
    BlockTreeTaskPool*                      task_pool_;
    BlockTreeCacheMetricsReporter&          metrics_reporter_;
    std::mutex&                             mutex_;
    int                                     disk_timeout_ms_{0};
    int                                     host_timeout_ms_{0};
    bool                                    enable_device_cache_{true};
    SettledFn                               settled_;
    LoadTaskRunner                          load_task_runner_;
    LoadJoinRegistry                        load_join_registry_;
    std::shared_ptr<LoadContextCoordinator> load_context_coordinator_;
};

}  // namespace rtp_llm
