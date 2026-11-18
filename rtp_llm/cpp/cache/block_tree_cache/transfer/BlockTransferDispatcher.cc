#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"
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

// The parent callback and retention node exist before the child can submit I/O.
// Only this coordinator boundary owns freezing; worker-local dispatch never does.
struct BlockTransferDispatcher::CoordinatedExecution {
    CoordinatedExecution(const BlockTransferDispatcher* owner,
                         FrozenTask::ExecutionInfo      execution,
                         TransferDoneCallback           done):
        owner(owner), execution(std::move(execution)), done(std::move(done)) {
        prepared_slot.emplace_back();
        if (owner->completion_cleanup_) {
            TransferCompletionTarget target;
            target.remote = this->execution.uses_rpc;
            target.peers.reserve(this->execution.participant_ranks.size());
            for (const auto& peer : this->execution.participant_ranks)
                target.peers.push_back({peer.rank, peer.address});
            cleanup = owner->completion_cleanup_->prepare(std::move(target), TransferCompletionKey{this->execution.registry_type, this->execution.trace_id});
        }
    }

    void complete(TransferResult result) {
        if (claimed.exchange(true, std::memory_order_acq_rel))
            return;
        try {
            if (result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
                // All allocation/validation precedes the first frozen hold. If
                // preparation fails, never turn an unprotected UNKNOWN into a
                // business failure that would release the ordinary references.
                auto frozen = std::make_shared<FrozenTask>(
                    FrozenTask::prepare(std::move(execution)), result, std::move(cleanup));
                prepared_slot.front() = frozen;
                frozen->freeze();
                owner->retainFrozenTask(prepared_slot);
            } else {
                // A known failure can receive its physical proof later. Queue
                // its identity now; worker forget retains RUNNING evidence.
                if (owner->completion_cleanup_)
                    owner->completion_cleanup_->enqueue(std::move(cleanup));
            }
        } catch (...) {
            std::terminate();
        }
        // A retained or late child callback must not retain the business owner
        // after completion. Invoke outside all registry/pool/frozen-task locks.
        auto notify = std::move(done);
        notify(std::move(result));
    }

    const BlockTransferDispatcher*                owner;
    FrozenTask::ExecutionInfo                      execution;
    TransferDoneCallback                          done;
    TransferCompletionCleanup::Prepared           cleanup;
    std::list<std::shared_ptr<FrozenTask>>         prepared_slot;
    std::atomic<bool>                             claimed{false};
};

std::shared_ptr<BlockTransferDispatcher::InflightTicket> BlockTransferDispatcher::trackTransfer() const {
    return std::make_shared<InflightTicket>(*this);
}

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine):
    per_rank_engine_(std::move(per_rank_engine)), multi_rank_engine_(std::move(multi_rank_engine)) {
    RTP_LLM_LOG_INFO("transfer_lifecycle event=dispatcher_init registry=%s engine=%p",
                    task_registry_->epoch().c_str(), static_cast<const void*>(per_rank_engine_.get()));
}

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
    RTP_LLM_LOG_INFO("transfer_lifecycle event=worker_execute_enter trace=%s request=%ld registry=%s",
                      trace_id.c_str(), task.requestId().value_or(-1), task_registry_->epoch().c_str());
    // Allocate the owner and its completion path before registering execution.
    // The safety callback owns the registry, not the dispatcher's business lifetime.
    const auto ticket = trackTransfer();
    auto context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    auto safety = std::make_shared<TransferBufferSafetyTracker>([registry = task_registry_, trace_id] {
        registry->finish(trace_id, true);
    });
    if (numeric_trace_id == 0 || !task_registry_->begin(trace_id)) {
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
    warnRegistrySize();
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

void BlockTransferDispatcher::warnRegistrySize() const noexcept {
    try {
        const auto count = task_registry_->size();
        auto threshold = next_registry_warning_.load(std::memory_order_relaxed);
        while (threshold != 0 && count >= threshold) {
            auto next = threshold;
            do {
                next = next > std::numeric_limits<size_t>::max() / 2 ? 0 : next * 2;
            } while (next != 0 && count >= next);
            if (next_registry_warning_.compare_exchange_weak(threshold, next, std::memory_order_relaxed)) {
                RTP_LLM_LOG_WARNING("worker transfer registry retains %zu records (warning threshold=%zu); "
                                    "records are not deleted and admission remains open",
                                    count,
                                    threshold);
                return;
            }
        }
    } catch (...) {
        // Diagnostics must not revoke an execution's recorded admission.
    }
}

std::shared_ptr<TransferAsyncContext>
BlockTransferDispatcher::executeMultiRank(TransferTask task, const char* completion_operation) const {
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(std::move(task), completion_operation);
    }
    return executePerRank(std::move(task));
}

FrozenTask::ExecutionInfo
BlockTransferDispatcher::describeExecution(const TransferTask& task) const {
    FrozenTask::ExecutionInfo info;
    info.registry_type = RegistryType::BLOCK_TRANSFER;
    info.trace_id = std::to_string(task.traceId());
    info.uses_rpc = multi_rank_engine_ != nullptr;
    if (multi_rank_engine_) {
        const auto addresses = multi_rank_engine_->participantAddresses();
        for (size_t rank = 0; rank < addresses.size(); ++rank)
            info.participant_ranks.push_back({rank, addresses[rank]});
    } else {
        info.participant_ranks.push_back({0, "local"});
    }
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
            info.target_blocks.push_back({std::move(pool), descriptor.target_blocks[member]});
        }
    }
    return info;
}

void BlockTransferDispatcher::submitCoordinated(TransferTask task,
                                               TransferDoneCallback done,
                                               const char* completion_operation) const {
    auto execution = std::make_shared<CoordinatedExecution>(
        this, describeExecution(task), std::move(done));
    // Construct the subscription before starting any I/O. The child's callback
    // vector may still reject it; that path also passes through this finalizer.
    TransferAsyncContext::DoneCallback complete =
        [execution](TransferResult result) { execution->complete(std::move(result)); };
    TransferResult observation_failure(
        ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted transfer completion"));
    std::shared_ptr<TransferAsyncContext> context;
    try {
        context = executeMultiRank(std::move(task), completion_operation);
        if (!context) {
            execution->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
            return;
        }
        context->onDone(std::move(complete));
    } catch (...) {
        if (execution->claimed.load(std::memory_order_acquire))
            throw; // Preserve exceptions from an inline business completion callback.
        if (!context) {
            // The caller still owns error classification and settlement.
            // Preserve the execution's cleanup candidate even without a returned context.
            if (completion_cleanup_)
                completion_cleanup_->enqueue(std::move(execution->cleanup));
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
        execution->complete(std::move(result));
    }
}

void BlockTransferDispatcher::retainFrozenTask(std::list<std::shared_ptr<FrozenTask>>& prepared_slot) const noexcept {
    std::lock_guard<std::mutex> lock(frozen_mutex_);
    frozen_tasks_.splice(frozen_tasks_.end(), prepared_slot);
}

std::vector<std::shared_ptr<FrozenTask>> BlockTransferDispatcher::frozenTasks() const {
    // Keep retired owners alive until the list lock is released, including if
    // snapshot allocation throws.
    std::list<std::shared_ptr<FrozenTask>>   retired;
    std::vector<std::shared_ptr<FrozenTask>> tasks;
    {
        std::lock_guard<std::mutex> lock(frozen_mutex_);
        for (auto it = frozen_tasks_.begin(); it != frozen_tasks_.end();) {
            const auto current = it++;
            if ((*current)->snapshot().state == FrozenTask::State::RELEASED) {
                retired.splice(retired.end(), frozen_tasks_, current);
            }
        }
        tasks.assign(frozen_tasks_.begin(), frozen_tasks_.end());
    }
    return tasks;
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
    submitCoordinated(std::move(task), std::move(callback), completion_operation);
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
