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
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;

// Owns the complete lower-tier-to-device load workflow. BlockTreeCache owns
// synchronization and tree matching; BlockTreeLoader owns load planning,
// tickets, transfer execution, state transitions, and settlement.
class BlockTreeLoader {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool tree_data_mutated, bool check_watermark)>;

    BlockTreeLoader(const std::vector<GroupSetPtr>& group_sets,
                    BlockTreeEvictor&                   evictor,
                    BlockTransferDispatcher*       transfer_dispatcher,
                    BlockTreeTaskPool*             task_pool,
                    BlockTreeCacheMetricsReporter& metrics_reporter,
                    std::mutex&                    mutex,
                    int                            disk_timeout_ms,
                    int                            host_timeout_ms,
                    bool                           enable_device_cache,
                    SettledFn                      settled);

    // The caller must hold the shared BlockTreeCache mutex.
    void prepareLoadLocked(const std::vector<TreeNode*>& matched_path, BlockTreeMatchResult& result);
    // The caller must hold the shared BlockTreeCache mutex.
    bool cancelLoadLocked(const std::shared_ptr<AsyncContext>& context);
    void shutdown();

private:
    void                        prepareMatchedLoadItem(TreeNode*                     path_node,
                                                       const GroupSetPtr&            group_set,
                                                       const GroupSetResource&       group_set_resource,
                                                       size_t                        path_index,
                                                       BlockTreeMatchResult&         result,
                                                       LoadTicket::PendingLoadItems& pending_load_items);
    std::shared_ptr<LoadTicket> prepareLoadTicket(LoadTicket::PendingLoadItems& items, size_t logical_matched_blocks);
    bool prepareJoinedLoadItem(LoadTicket::PendingLoadItem& item, const std::shared_ptr<LoadAsyncContext>& context);
    bool reserveLoadItems(const LoadTicket::PendingLoadItems& items);
    std::shared_ptr<AsyncContext> commitLoad(const LoadTicket& ticket);
    void                          abortLoad(const LoadTicket& ticket);
    void                          abortLoadUnsafe(const LoadTicket::PendingLoadItems&      items,
                                                  size_t                                   prepared_item_count,
                                                  const std::shared_ptr<LoadAsyncContext>& context);
    void                          runLoadTask(const LoadTaskRunner::TaskPtr& task);
    bool                          settleLoadNolock(LoadTaskRunner::Task& task, bool copy_success);

    bool reserveLoad(TreeNode* node, size_t group_set_id, Tier source, const std::vector<BlockIdxType>& source_blocks);
    bool
    abortPendingLoad(TreeNode* node, size_t group_set_id, Tier source, const std::vector<BlockIdxType>& source_blocks);
    bool beginLoad(TreeNode* node, size_t group_set_id, Tier source);
    bool finishLoad(TreeNode* node, size_t group_set_id, Tier source, bool copy_ok);

    const std::vector<GroupSetPtr>&           group_sets_;
    BlockTreeEvictor&                   evictor_;
    BlockTransferDispatcher*            transfer_dispatcher_;
    BlockTreeTaskPool*                  task_pool_;
    BlockTreeCacheMetricsReporter&      metrics_reporter_;
    std::mutex&                         mutex_;
    int                                 disk_timeout_ms_{0};
    int                                 host_timeout_ms_{0};
    bool                                enable_device_cache_{true};
    SettledFn                           settled_;
    LoadTaskRunner                      load_task_runner_;
    LoadJoinRegistry                    load_join_registry_;
    std::shared_ptr<LoadTicketRegistry> load_ticket_registry_;
};

}  // namespace rtp_llm
