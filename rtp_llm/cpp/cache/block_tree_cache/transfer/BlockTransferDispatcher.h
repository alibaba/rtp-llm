#pragma once

#include <cstddef>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;
struct BlockTreeQueueSizes;

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(TransferResult)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine = nullptr,
                            size_t                                        max_descriptors_per_batch = 8);
    ~BlockTransferDispatcher();

    std::shared_ptr<TransferAsyncContext> executePerRank(TransferTask task) const;

    // Callback-driven path used by Load/Store/Evict: stable grouping and bounded batches.
    void runTransfer(TransferTask task, TransferDoneCallback callback) const;

    void                cancelPendingStagingTransfers() const;
    BlockTreeQueueSizes queueSizes() const;

    // Lifecycle-only drain, not task admission or a request-level sync API.
    // Call after producers stop; callbacks can still submit a subsequent stage.
    void drainTransfers() const;
    void pollFrozenTasks() const;
    // Lifecycle owner calls this without holding cache/pool locks; waits for in-flight polling.
    void stopFrozenPolling();
    // IDs must come from FINISHED results confirmed across every worker.
    void releaseCompletedFrozenTasks(const std::unordered_set<std::string>& completed_ids,
                                     FrozenTask::QueryKind query_kind = FrozenTask::QueryKind::TRANSFER) const;
    void retainFrozenTask(FrozenTask task) const {
        std::lock_guard<std::mutex> lock(frozen_mutex_);
        frozen_tasks_.push_back(std::move(task));
    }

private:
    using QueryFrozenTasks = std::function<std::vector<TaskCompletion>(const std::vector<std::string>&)>;
    void pollFrozenTasksWithQuery(const QueryFrozenTasks& query) const;
    void pollFrozenTasksForKind(FrozenTask::QueryKind query_kind, const QueryFrozenTasks& query) const;
    mutable std::mutex frozen_poll_mutex_;
    std::mutex frozen_stop_mutex_;
    std::condition_variable frozen_stop_cv_;
    bool frozen_stopping_{false};
    std::thread frozen_thread_;
    std::shared_ptr<TransferAsyncContext> executeMultiRank(TransferTask task) const;

    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    size_t                                        max_descriptors_per_batch_{8};
    mutable std::mutex                            completion_mutex_;
    mutable std::vector<std::shared_future<void>>  transfer_completions_;
    std::function<void()>                         drain_observer_for_test_;
    mutable std::mutex                            frozen_mutex_;
    mutable std::vector<FrozenTask>                frozen_tasks_;
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
