#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;
class TransferBatchAsyncContext;
class TransferCompletionCleanup;
struct BlockTreeQueueSizes;

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(TransferResult)>;
    using LocalCopyObserver    = std::function<void(const TransferTask&, const TransferResult&)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine         = nullptr,
                            size_t                                        max_descriptors_per_batch = 0);
    ~BlockTransferDispatcher();

    // The per-rank engine groups and batches the logical request.
    std::shared_ptr<TransferAsyncContext> executePerRank(TransferTask task) const;

    // Installed at cache construction, before transfer producers start.
    void setLocalCopyObserver(LocalCopyObserver observer) {
        local_copy_observation_ = std::make_shared<LocalCopyObservation>(std::move(observer));
    }
    bool localCopyEvidenceComplete() const {
        return !local_copy_observation_ || local_copy_observation_->complete.load();
    }
    void runTransfer(CacheTransferOperation operation,
                     TransferTask           task,
                     TransferDoneCallback   callback,
                     const char*            completion_operation = nullptr) const {
        task.operation = operation;
        runTransfer(std::move(task), std::move(callback), completion_operation);
    }

    // Broadcast one single-direction request; each rank owns physical batching.
    // completion_operation is an optional borrowed literal for new completion
    // observations only. Existing physical transfer stage identities are unchanged.
    void
    runTransfer(TransferTask task, TransferDoneCallback callback, const char* completion_operation = nullptr) const;

    void                     cancelPendingStagingTransfers() const;
    BlockTreeQueueSizes      queueSizes() const;
    RemoteMockSafetySnapshot remoteMockSafety() const;

    // Close per-rank admission, drain local owners, then wait for transfer
    // callback chains. Call after business producers/task bodies stop; keep
    // business completion consumers alive and drain their settlement afterwards.
    // The lifecycle caller serializes shutdown(), waitForTransfers() and destruction.
    void shutdown() const;

    // Nonterminal drain of submission stacks and final callback chains, including
    // synchronously submitted stages. Stop independent producers before destruction;
    // queued business settlement must be drained separately afterwards.
    void waitForTransfers() const;
    void drainTransfers() const {
        waitForTransfers();
    }

    // Snapshot retained records, retiring RELEASED owners outside the list lock.
    // FROZEN and RELEASING records remain; enumeration never unfreezes blocks.
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

    // Worker-local evidence only; no query RPC or recovery loop.
    const TransferTaskRegistry& taskRegistry() const {
        return *task_registry_;
    }

    void cleanupExpiredMissing() const {
        task_registry_->cleanupExpiredMissing();
    }

    // Recovery queries fence unseen executions before returning their evidence.
    TransferTaskRegistry::State queryLocalCompletion(uint64_t trace_id) const {
        return task_registry_->queryAndBlockMissing(std::vector<uint64_t>{trace_id}).front();
    }

    TransferTaskRegistry::ForgetResult forgetLocalCompletion(uint64_t trace_id) const {
        return task_registry_->forget(trace_id);
    }

    // Initialization only, before any coordinator execution can be submitted.
    void setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup) {
        completion_cleanup_ = std::move(cleanup);
    }

private:
    class InflightTicket;
    std::shared_ptr<InflightTicket> trackTransfer() const;
    struct CoordinatedExecution;
    void warnRegistrySize() const noexcept;

    std::shared_ptr<TransferAsyncContext> executeMultiRank(TransferTask task,
                                                           const char*  completion_operation = nullptr) const;
    void submitCoordinated(TransferTask task, TransferDoneCallback done, const char* completion_operation) const;
    FrozenTask::ExecutionInfo buildExecutionInfo(const TransferTask& task, const char* completion_operation) const;
    void                      retainFrozenTask(std::list<std::shared_ptr<FrozenTask>>& prepared_slot) const noexcept;

    struct LocalCopyObservation {
        explicit LocalCopyObservation(LocalCopyObserver callback): observer(std::move(callback)) {}
        LocalCopyObserver observer;
        std::atomic<bool> complete{true};
    };

    std::shared_ptr<PerRankBlockTransferEngine>    per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine>  multi_rank_engine_;
    size_t                                         max_descriptors_per_batch_{0};
    std::shared_ptr<LocalCopyObservation>          local_copy_observation_;
    mutable std::mutex                             inflight_mutex_;
    mutable std::condition_variable                inflight_cv_;
    mutable size_t                                 inflight_count_{0};
    std::function<void()>                          drain_observer_for_test_;
    std::shared_ptr<TransferBatchAsyncContext> (*make_observed_context_)(size_t);
    mutable std::mutex                             frozen_mutex_;
    mutable std::list<std::shared_ptr<FrozenTask>> frozen_tasks_;
    std::shared_ptr<TransferTaskRegistry>          task_registry_{std::make_shared<TransferTaskRegistry>()};
    std::shared_ptr<TransferCompletionCleanup>     completion_cleanup_;
    mutable std::atomic<size_t>                    next_registry_warning_{4096};
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
