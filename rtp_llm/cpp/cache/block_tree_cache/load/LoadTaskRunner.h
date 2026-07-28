#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;

class LoadTaskRunner {
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
private:
    void releaseStagingBlocks(Task& task);
    void releaseUninstalledTargetHolders(const Task& task);
};

}  // namespace rtp_llm
