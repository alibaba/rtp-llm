#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;

class LoadBackWorker {
public:
    struct Task {
        LoadBackTicket::PendingLoadBackItems items;
        std::vector<ComponentGroupPtr>        item_groups;
        std::vector<BlockIdxType>             staging_host_blocks;
        std::vector<TransferDescriptor>       disk_to_host_descriptors;
        std::vector<TransferDescriptor>       host_to_device_descriptors;
        std::vector<bool>                     target_installed;
        std::shared_ptr<LoadBackAsyncContext> context;
    };
    using TaskPtr = std::shared_ptr<Task>;

    enum class PrepareStatus {
        READY,
        NEED_HOST_RECLAIM,
        FAILED,
    };

    bool          createTask(const LoadBackTicket::PendingLoadBackItems&  items,
                             const std::vector<ComponentGroupPtr>&        component_groups,
                             const std::shared_ptr<LoadBackAsyncContext>& context,
                             TaskPtr&                                     task);
    PrepareStatus prepareTransferItem(Task& task, size_t item_index);
    bool runTransfer(Task&                          task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     int                            disk_timeout_ms,
                     int                            host_timeout_ms,
                     bool                           prepared);
    void releaseTaskResources(Task& task);
    bool cancelLoadBackNolock(const std::shared_ptr<AsyncContext>& context);

    // Registry operations rely on the BlockTreeCache mutex.
    bool startLoading(TreeNode*                                    node,
                      int                                          group_id,
                      const std::vector<BlockIdxType>&             target_blocks,
                      const std::shared_ptr<LoadBackAsyncContext>& context);
    std::optional<std::vector<BlockIdxType>>
    joinLoading(TreeNode* node, int group_id, const std::shared_ptr<LoadBackAsyncContext>& context);
    bool finishLoading(TreeNode* node, int group_id, bool success);
    bool eraseLoadingForOneContext(TreeNode* node, int group_id, const std::shared_ptr<LoadBackAsyncContext>& context);

private:
    struct LoadingKey {
        TreeNode* node;
        int       group_id;

        bool operator==(const LoadingKey& other) const {
            return node == other.node && group_id == other.group_id;
        }
    };

    struct LoadingKeyHash {
        size_t operator()(const LoadingKey& key) const {
            const size_t node_hash  = std::hash<TreeNode*>{}(key.node);
            const size_t group_hash = std::hash<int>{}(key.group_id);
            return node_hash ^ (group_hash << 1);
        }
    };

    struct LoadingRecord {
        std::vector<BlockIdxType>                          target_blocks;
        std::vector<std::shared_ptr<LoadBackAsyncContext>> contexts;
    };
    using LoadingRecordMap = std::unordered_map<LoadingKey, LoadingRecord, LoadingKeyHash>;

    void releaseStagingBlocks(Task& task);
    void releaseUninstalledTargetHolders(const Task& task);

    LoadingRecordMap loading_records_;
};

}  // namespace rtp_llm
