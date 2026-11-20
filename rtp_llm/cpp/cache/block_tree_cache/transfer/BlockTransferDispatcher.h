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
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;
struct BlockTreeQueueSizes;

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(TransferResult)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine = nullptr);
    ~BlockTransferDispatcher();

    // The per-rank engine groups and batches the logical request.
    std::shared_ptr<TransferAsyncContext> executePerRank(TransferTask task) const;

    // Broadcast one single-direction request; each rank owns physical batching.
    // completion_operation is an optional borrowed literal for new completion
    // observations only. Existing physical transfer stage identities are unchanged.
    void
    runTransfer(TransferTask task, TransferDoneCallback callback, const char* completion_operation = nullptr) const;

    void                cancelPendingStagingTransfers() const;
    BlockTreeQueueSizes queueSizes() const;

    // Close per-rank admission, drain local owners, then wait for transfer
    // callback chains. Call after business producers/task bodies stop; keep
    // business completion consumers alive and drain their settlement afterwards.
    // The lifecycle caller serializes shutdown(), waitForTransfers() and destruction.
    void shutdown() const;

    // Nonterminal drain of submission stacks and final callback chains, including
    // synchronously submitted stages. Stop independent producers before destruction;
    // queued business settlement must be drained separately afterwards.
    void waitForTransfers() const;

    // Snapshot retained records, retiring RELEASED owners outside the list lock.
    // FROZEN and RELEASING records remain; enumeration never unfreezes blocks.
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

    const TransferTaskRegistry& taskRegistry() const {
        return *task_registry_;
    }

    void setReclaimers(std::shared_ptr<FrozenTaskReclaimer> coordinator,
                       std::shared_ptr<TransferTaskRegistry> worker);

private:
    class InflightTicket;
    std::shared_ptr<InflightTicket> trackTransfer() const;
    struct CoordinatedExecution;
    void warnRegistrySize() const noexcept;

    std::shared_ptr<TransferAsyncContext> executeMultiRank(TransferTask task,
                                                           const char*  completion_operation = nullptr) const;
    void submitCoordinated(TransferTask task, TransferDoneCallback done, const char* completion_operation) const;
    std::vector<FrozenTask::TargetBlock> targetBlocks(const TransferTask& task) const;

    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    mutable std::mutex                            inflight_mutex_;
    mutable std::condition_variable               inflight_cv_;
    mutable size_t                                inflight_count_{0};
    std::function<void()>                         drain_observer_for_test_;
    std::shared_ptr<FrozenTaskReclaimer> coordinator_{std::make_shared<FrozenTaskReclaimer>(
        std::make_shared<TransferCompletionCleanup>())};
    std::shared_ptr<TransferTaskRegistry>          task_registry_{std::make_shared<TransferTaskRegistry>()};
    std::shared_ptr<TransferCompletionCleanup>     completion_cleanup_;
    mutable std::atomic<size_t>                   next_registry_warning_{4096};
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
