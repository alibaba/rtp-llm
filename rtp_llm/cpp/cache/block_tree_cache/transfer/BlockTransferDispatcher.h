#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

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

private:
    class InflightTicket;
    std::shared_ptr<InflightTicket> trackTransfer() const;

    std::shared_ptr<TransferAsyncContext> executeMultiRank(TransferTask task,
                                                           const char*  completion_operation = nullptr) const;

    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    mutable std::mutex                            inflight_mutex_;
    mutable std::condition_variable               inflight_cv_;
    mutable size_t                                inflight_count_{0};
    std::function<void()>                         drain_observer_for_test_;
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
