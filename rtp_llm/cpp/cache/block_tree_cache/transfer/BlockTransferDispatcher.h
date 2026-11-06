#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;
class StorageBackend;
struct BlockTreeQueueSizes;

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(TransferResult)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine = nullptr,
                            size_t                                        max_descriptors_per_batch = 32,
                            std::shared_ptr<StorageBackend>                local_kvcm_backend = nullptr);
    ~BlockTransferDispatcher();

    // Requests are grouped and split locally before entering either engine.
    std::shared_ptr<TransferAsyncContext> executePerRank(TransferTask task) const;

    // Broadcast once per direction; each rank owns physical batching.
    // completion_operation is an optional borrowed literal for new completion
    // observations only. Existing physical transfer stage identities are unchanged.
    void
    runTransfer(TransferTask task, TransferDoneCallback callback, const char* completion_operation = nullptr) const;

    void                cancelPendingStagingTransfers() const;
    BlockTreeQueueSizes queueSizes() const;

    void pollFrozenTasks() const;
    // Lifecycle owner calls this without holding cache/pool locks; waits for in-flight polling.
    void stopFrozenPolling();
    // IDs must come from FINISHED results confirmed by the local backend or every worker.
    void releaseCompletedFrozenTasks(const std::unordered_set<std::string>& completed_ids,
                                     FrozenTask::QueryKind query_kind = FrozenTask::QueryKind::TRANSFER) const;
    void retainFrozenTask(FrozenTask task) const {
        std::lock_guard<std::mutex> lock(frozen_mutex_);
        frozen_tasks_.push_back(std::move(task));
    }
    void enqueueFinishedKVCMOperation(const std::string& operation_id) const;


    // Close per-rank admission, drain local owners, then wait for transfer
    // callback chains. Call after business producers/task bodies stop; keep
    // business completion consumers alive and drain their settlement afterwards.
    // The lifecycle caller serializes shutdown(), waitForTransfers() and destruction.
    void shutdown() const;

    // Nonterminal drain of submission stacks and final callback chains, including
    // synchronously submitted stages. Stop independent producers before destruction;
    // queued business settlement must be drained separately afterwards.
    void waitForTransfers() const;

private:
    struct CompletionTicket;
    void registerTransfer() const;
    void retireTransfer() const noexcept;

    // Local dispatch uses bounded, homogeneous batches. Broadcast dispatch
    // groups by direction only, as required by MemoryOperationRequestPB.
    void dispatchGroups(TransferTask         task,
                        TransferDoneCallback callback,
                        bool                 local,
                        const char*          completion_operation = nullptr) const;

    // Frozen work is queried after transfers have reported an unknown result.
    using QueryFrozenTasks = std::function<std::vector<TaskCompletion>(const std::vector<std::string>&)>;
    void pollFrozenTasksWithQuery(const QueryFrozenTasks& query) const;
    void pollFrozenTasksForKind(FrozenTask::QueryKind query_kind, const QueryFrozenTasks& query) const;
    void forgetFinishedLocalKVCMOperations() const;
    mutable std::mutex frozen_poll_mutex_;
    std::mutex frozen_stop_mutex_;
    std::condition_variable frozen_stop_cv_;
    bool frozen_stopping_{false};
    std::thread frozen_thread_;

    std::shared_ptr<TransferAsyncContext> executeMultiRank(TransferTask task,
                                                           const char*  completion_operation = nullptr) const;

    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    std::shared_ptr<StorageBackend>                local_kvcm_backend_;
    mutable std::mutex                            local_kvcm_cleanup_mutex_;
    mutable std::mutex                            local_kvcm_cleanup_poll_mutex_;
    mutable std::unordered_set<std::string>        local_kvcm_finished_ids_;
    size_t                                        max_descriptors_per_batch_{32};
    mutable std::mutex                            completion_mutex_;
    mutable std::condition_variable               completion_cv_;
    mutable size_t                                inflight_transfers_{0};
    std::function<void()>                         drain_observer_for_test_;
    mutable std::mutex                            frozen_mutex_;
    mutable std::vector<FrozenTask>                frozen_tasks_;
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
