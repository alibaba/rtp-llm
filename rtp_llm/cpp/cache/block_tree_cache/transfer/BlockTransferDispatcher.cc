#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

class BlockTransferDispatcher::InflightTicket {
public:
    explicit InflightTicket(const BlockTransferDispatcher& dispatcher): dispatcher_(dispatcher) {
        std::lock_guard<std::mutex> lock(dispatcher_.inflight_mutex_);
        RTP_LLM_CHECK(dispatcher_.inflight_count_ < std::numeric_limits<size_t>::max());
        ++dispatcher_.inflight_count_;
    }
    ~InflightTicket() noexcept {
        std::lock_guard<std::mutex> lock(dispatcher_.inflight_mutex_);
        RTP_LLM_CHECK(dispatcher_.inflight_count_ != 0);
        if (--dispatcher_.inflight_count_ == 0) {
            dispatcher_.inflight_cv_.notify_all();
        }
    }
    InflightTicket(const InflightTicket&)            = delete;
    InflightTicket& operator=(const InflightTicket&) = delete;

private:
    const BlockTransferDispatcher& dispatcher_;
};

struct BlockTransferDispatcher::TargetBlockSafetyHandler {
    TargetBlockSafetyHandler(std::shared_ptr<BlockTreeReclaimer> reclaimer,
                             TransferCompletionKey key,
                             TransferDoneCallback done):
        reclaimer(std::move(reclaimer)), key(std::move(key)), done(std::move(done)) {}

    void complete(TransferResult result) {
        if (claimed.exchange(true))
            return;
        if (result.checkNeedFreeze())
            reclaimer->freezeTargetBlocksWithTraceIDAndStillIOWorkerID(key, {}, result.still_io_worker_ids);
        else
            reclaimer->unregisterTask(key);
        auto notify = std::move(done);
        notify(std::move(result));
    }

    std::shared_ptr<BlockTreeReclaimer> reclaimer;
    TransferCompletionKey key;
    TransferDoneCallback done;
    std::atomic<bool> claimed{false};
};

std::shared_ptr<BlockTransferDispatcher::InflightTicket> BlockTransferDispatcher::trackTransfer() const {
    return std::make_shared<InflightTicket>(*this);
}

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine):
    per_rank_engine_(std::move(per_rank_engine)), multi_rank_engine_(std::move(multi_rank_engine)) {}

BlockTransferDispatcher::~BlockTransferDispatcher() {
    shutdown();
}

void BlockTransferDispatcher::shutdown() const {
    if (per_rank_engine_)
        per_rank_engine_->shutdown();
    waitForTransfers();
}

void BlockTransferDispatcher::waitForTransfers() const {
    std::unique_lock<std::mutex> lock(inflight_mutex_);
    if (inflight_count_ != 0 && drain_observer_for_test_) {
        lock.unlock();
        drain_observer_for_test_();
        lock.lock();
    }
    inflight_cv_.wait(lock, [this] { return inflight_count_ == 0; });
}

std::shared_ptr<TransferAsyncContext> BlockTransferDispatcher::executePerRank(TransferTask task) const {
    const auto numeric_trace_id = task.traceId();
    const auto trace_id = std::to_string(numeric_trace_id);
    RTP_LLM_LOG_INFO("transfer_lifecycle event=worker_execute_enter trace=%s request=%ld",
                      trace_id.c_str(), task.requestId().value_or(-1));
    // Allocate the owner and its completion path before registering execution.
    // The safety callback owns the registry, not the dispatcher's business lifetime.
    const auto ticket = trackTransfer();
    auto context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    auto safety = std::make_shared<TransferBufferSafetyTracker>([registry = reclaimer_, trace_id] {
        registry->ioFinishedOnWorkerWithTraceID(trace_id);
    });
    if (numeric_trace_id == 0 || !reclaimer_->registerOnWorkerTaskWithTraceID(trace_id)) {
        const auto* reason = numeric_trace_id == 0 ? "empty transfer trace" : "duplicate transfer trace";
        RTP_LLM_LOG_WARNING("%s: trace=%s", reason, trace_id.c_str());
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, std::string(reason) + ": " + trace_id),
                              CacheCopyError::NOT_EXECUTED);
        // This input never reaches I/O. Do not finish or overwrite the existing trace.
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(
            std::move(result), task.descriptors().size());
    }
    block_tree_cache_detail::ScopeRollback close_submission([safety] {
        safety->finishSubmitting();
    });
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;  // No batch was submitted.
        close_submission.run();
        context->complete(std::move(result));
        return context;
    }
    TransferAsyncContext::DoneCallback complete =
        [ticket, context](TransferResult result) { context->complete(std::move(result)); };
    TransferResult observation_failure(ErrorInfo(
        ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted local transfer completion"));
    close_submission.dismiss();
    auto child = per_rank_engine_->execute(std::move(task), std::move(safety));
    if (!child) {
        context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
    } else {
        try {
            child->onDone(complete);
        } catch (...) {
            // The producer still owns its proof even when business observation fails.
            try {
                if (child->done()) {
                    complete(child->result());
                    return context;
                }
            } catch (...) {
            }
            complete(std::move(observation_failure));
        }
    }
    return context;
}

std::shared_ptr<TransferAsyncContext>
BlockTransferDispatcher::executeMultiRank(TransferTask task, const char* completion_operation) const {
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(std::move(task), completion_operation);
    }
    return executePerRank(std::move(task));
}

std::vector<TransferTargetBlock> BlockTransferDispatcher::resolveTargetBlockRefs(const TransferTask& task) const {
    std::vector<TransferTargetBlock> targets;
    const auto& groups = multi_rank_engine_ ? multi_rank_engine_->groupSets() : per_rank_engine_->groupSets();
    for (const auto& descriptor : task.descriptors()) {
        const GroupSet* group = descriptor.group_set_id < groups.size() ? groups[descriptor.group_set_id].get() : nullptr;
        for (size_t member = 0; member < descriptor.target_blocks.size(); ++member) {
            std::shared_ptr<IBlockPool> pool;
            if (group) {
                if (descriptor.target_tier == Tier::DEVICE && member < group->devicePools().size())
                    pool = group->devicePools()[member];
                else if (descriptor.target_tier == Tier::HOST)
                    pool = group->hostPool();
                else if (descriptor.target_tier == Tier::DISK)
                    pool = group->diskPool();
            }
            targets.push_back({std::move(pool), descriptor.target_blocks[member]});
        }
    }
    return targets;
}

void BlockTransferDispatcher::registerTargetsAndSubmitTransfer(TransferTask task,
                                                               TransferDoneCallback done,
                                                               const char* completion_operation) const {
    auto safety_handler = std::make_shared<TargetBlockSafetyHandler>(reclaimer_, task.traceId(), std::move(done));
    // Construct the subscription before starting any I/O. The child's callback
    // vector may still reject it; that path also passes through this finalizer.
    TransferAsyncContext::DoneCallback complete =
        [safety_handler](TransferResult result) { safety_handler->complete(std::move(result)); };
    TransferResult observation_failure(
        ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted transfer completion"));
    std::shared_ptr<TransferAsyncContext> context;
    const bool remote = multi_rank_engine_ != nullptr;
    const auto ranks = remote ? multi_rank_engine_->workerCount() : 1;
    if (ranks && !reclaimer_->registerTask(safety_handler->key, {remote, ranks}, resolveTargetBlockRefs(task))) {
        TransferResult duplicate(ErrorInfo(ErrorCode::INVALID_PARAMS, "duplicate or empty transfer trace"),
                                 CacheCopyError::NOT_EXECUTED);
        duplicate.buffer_safe = true;
        safety_handler->done(std::move(duplicate));
        return;
    }
    try {
        context = executeMultiRank(std::move(task), completion_operation);
        if (!context) {
            safety_handler->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
            return;
        }
        context->onDone(std::move(complete));
    } catch (...) {
        if (safety_handler->claimed.load(std::memory_order_acquire))
            throw; // Preserve exceptions from an inline business completion callback.
        if (!context) {
            reclaimer_->unregisterTask(safety_handler->key);
            throw;
        }
        auto result = std::move(observation_failure);
        try {
            if (context && context->done()) {
                result = context->result();
            }
        } catch (...) {
            // A failed observation cannot prove a submitted execution safe.
        }
        safety_handler->complete(std::move(result));
    }
}

void BlockTransferDispatcher::runTransfer(TransferTask         task,
                                          TransferDoneCallback callback,
                                          const char*          completion_operation) const {
    // Keep a local owner even if the final callback completes synchronously.
    const auto ticket = trackTransfer();
    callback = [ticket, block_count = task.descriptors().size(), done = std::move(callback)](TransferResult result) {
        result.expandBlockErrors(block_count);
        if (!result.ok()) {
            RTP_LLM_LOG_ERROR("block transfer failed: descriptor_count=%zu, error_type=%s, error_code=%d, details=%s",
                              block_count,
                              cacheCopyErrorName(result.copy_error),
                              static_cast<int>(result.status.code()),
                              result.status.ToString().c_str());
        }
        done(std::move(result));
    };
    registerTargetsAndSubmitTransfer(std::move(task), std::move(callback), completion_operation);
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
