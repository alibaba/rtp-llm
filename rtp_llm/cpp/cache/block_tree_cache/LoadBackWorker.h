#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackTicket.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;

class LoadBackAsyncContext: public AsyncContext {
public:
    enum class State : int {
        PENDING          = 0,
        CANCEL_REQUESTED = 1,
        SUCCEEDED        = 2,
        FAILED           = 3,
        CANCELLED        = 4
    };

    bool requestCancel();
    bool cancelRequested() const;
    void onTaskComplete(bool ok);
    void waitDone() override;
    bool done() const override;
    bool success() const override;

private:
    std::atomic<State>      state_{State::PENDING};
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
};

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

    TaskPtr createTask(const LoadBackTicket::PendingLoadBackItems& items,
                       const std::vector<ComponentGroupPtr>&       item_groups);
    PrepareStatus prepareTransferItem(Task& task, size_t item_index);
    bool runTransfer(Task&                          task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     int                            disk_timeout_ms,
                     int                            host_timeout_ms,
                     bool                           prepared);
    void completeTask(Task& task, bool success);
    bool cancelLoadBackNolock(const std::shared_ptr<AsyncContext>& context);

private:
    void releaseStagingBlocks(Task& task);
    void releaseUninstalledTargetHolders(const Task& task);
};

}  // namespace rtp_llm
