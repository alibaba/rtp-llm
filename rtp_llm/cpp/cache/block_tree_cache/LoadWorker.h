#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;

class LoadWorker {
public:
    struct Task {
        LoadTicket::PendingLoadItems          items;
        std::vector<GroupSetPtr>              item_groups;
        std::vector<BlockIdxType>             staging_host_blocks;
        std::vector<TransferDescriptor>       disk_to_host_descriptors;
        std::vector<TransferDescriptor>       host_to_device_descriptors;
        std::vector<bool>                     target_installed;
        std::shared_ptr<LoadAsyncContext>     context;
    };
    using TaskPtr = std::shared_ptr<Task>;

    enum class PrepareStatus {
        READY,
        NEED_HOST_RECLAIM,
        FAILED,
    };

    bool          createTask(const LoadTicket::PendingLoadItems&      items,
                             const std::vector<GroupSetPtr>&          group_sets,
                             const std::shared_ptr<LoadAsyncContext>& context,
                             TaskPtr&                                 task);
    PrepareStatus prepareTransferItem(Task& task, size_t item_index);
    bool          runTransfer(Task&                          task,
                              const BlockTransferDispatcher& transfer_dispatcher,
                              BlockTreeCacheMetricsReporter& metrics_reporter,
                              int                            disk_timeout_ms,
                              int                            host_timeout_ms,
                              bool                           prepared);
    void          releaseTaskResources(Task& task);
    bool          cancelLoadNolock(const std::shared_ptr<AsyncContext>& context);

    // Registry operations rely on the BlockTreeCache mutex.
    bool startLoading(TreeNode*                                node,
                      size_t                                   group_set_id,
                      const std::vector<BlockIdxType>&         target_blocks,
                      const std::shared_ptr<LoadAsyncContext>& context);
    std::optional<std::vector<BlockIdxType>>
         joinLoading(TreeNode* node, size_t group_set_id, const std::shared_ptr<LoadAsyncContext>& context);
    bool finishLoading(TreeNode* node, size_t group_set_id, bool success);
    bool
    eraseLoadingForOneContext(TreeNode* node, size_t group_set_id, const std::shared_ptr<LoadAsyncContext>& context);

private:
    struct LoadingKey {
        TreeNode* node;
        size_t    group_set_id;

        bool operator==(const LoadingKey& other) const {
            return node == other.node && group_set_id == other.group_set_id;
        }
    };

    struct LoadingKeyHash {
        size_t operator()(const LoadingKey& key) const {
            const size_t node_hash  = std::hash<TreeNode*>{}(key.node);
            const size_t group_hash = std::hash<size_t>{}(key.group_set_id);
            return node_hash ^ (group_hash << 1);
        }
    };

    struct LoadingRecord {
        std::vector<BlockIdxType>                          target_blocks;
        std::vector<std::shared_ptr<LoadAsyncContext>>     contexts;
    };
    using LoadingRecordMap = std::unordered_map<LoadingKey, LoadingRecord, LoadingKeyHash>;

    void releaseStagingBlocks(Task& task);
    void releaseUninstalledTargetHolders(const Task& task);

    LoadingRecordMap loading_records_;
};

}  // namespace rtp_llm
