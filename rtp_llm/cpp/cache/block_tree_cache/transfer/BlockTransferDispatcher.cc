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
            target.remote = this->execution.remote;
            target.peers.reserve(this->execution.expected_peers.size());
            for (const auto& peer : this->execution.expected_peers)
                target.peers.push_back({peer.rank, peer.address});
            cleanup = owner->completion_cleanup_->prepare(std::move(target), this->execution.trace_id);
        }
    }

    void complete(TransferResult result) {
        if (claimed.exchange(true, std::memory_order_acq_rel))
            return;
        try {
            if (result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
                if (!result.execution_diagnostics) {
                    auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
                    diagnostics->trace_id = execution.trace_id;
                    for (const auto& peer : execution.expected_peers) {
                        TransferRankSnapshot rank;
                        rank.rank = peer.rank;
                        rank.server_address = peer.address;
                        rank.remote = execution.remote;
                        rank.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
                        if (!execution.remote) {
                            rank.worker_error_code = static_cast<int64_t>(result.status.code());
                            rank.worker_error_message = result.status.ToString();
                            rank.worker_copy_error = result.copy_error;
                            rank.block_errors = result.block_errors;
                            rank.buffer_safe = result.buffer_safe;
                        }
                        diagnostics->ranks.push_back(std::move(rank));
                    }
                    result.execution_diagnostics = std::move(diagnostics);
                }
                // All allocation/validation precedes the first frozen hold. If
                // preparation fails, never turn an unprotected UNKNOWN into a
                // business failure that would release the ordinary references.
                auto frozen = std::make_shared<FrozenTask>(
                    FrozenTask::prepare(std::move(execution)), result, std::move(cleanup));
                prepared_slot.front() = frozen;
                frozen->freeze();
                owner->retainFrozenTask(prepared_slot);
                try {
                    RTP_LLM_LOG_WARNING("transfer frozen: %s", frozen->debugString().c_str());
                } catch (...) {
                    // Diagnostics are best-effort after protection is established.
                }
            } else {
                // A known failure can receive its physical proof later. Queue
                // its identity now; worker forget retains RUNNING evidence.
                cleanup.submit();
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
    // Allocate the owner and its completion path before registering execution.
    // The safety callback owns the registry, not the dispatcher's business lifetime.
    const auto ticket = trackTransfer();
    auto context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    auto proof = std::make_shared<TransferBufferSafety>([registry = task_registry_, trace_id] {
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
    block_tree_cache_detail::ScopeRollback close_submission([proof] {
        proof->report(true);
    });
    warnRegistrySize();
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;  // No batch was submitted.
        proof->report(true);
        context->complete(std::move(result));
        return context;
    }
    TransferAsyncContext::DoneCallback complete =
        [ticket, context](TransferResult result) { context->complete(std::move(result)); };
    TransferResult observation_failure(ErrorInfo(
        ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted local transfer completion"));
    task.setCompletionProof(std::move(proof));
    close_submission.dismiss();
    auto child = per_rank_engine_->execute(std::move(task));
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
BlockTransferDispatcher::describeExecution(const TransferTask& task, const char* completion_operation) const {
    FrozenTask::ExecutionInfo info;
    info.trace_id = task.traceId();
    info.operation = completion_operation ? completion_operation : "transfer";
    info.request_id = task.requestId();
    info.workflow_id = task.workflowId();
    info.parent_trace_id = task.parentTraceId();
    info.remote = multi_rank_engine_ != nullptr;
    if (multi_rank_engine_) {
        const auto addresses = multi_rank_engine_->participantAddresses();
        for (size_t rank = 0; rank < addresses.size(); ++rank)
            info.expected_peers.push_back({rank, addresses[rank]});
    } else {
        info.expected_peers.push_back({0, "local"});
    }
    info.expected_rank_count = info.expected_peers.size();
    const auto& groups = multi_rank_engine_ ? multi_rank_engine_->groupSets() : per_rank_engine_->groupSets();
    info.descriptors.reserve(task.descriptors().size());
    for (const auto& descriptor : task.descriptors()) {
        FrozenTask::DescriptorInfo saved;
        saved.group_set_id = descriptor.group_set_id;
        saved.path_index = descriptor.path_index;
        saved.source_tier = descriptor.source_tier;
        saved.target_tier = descriptor.target_tier;
        const GroupSet* group = descriptor.group_set_id < groups.size() ? groups[descriptor.group_set_id].get() : nullptr;
        const auto capture = [group](Tier tier, const std::vector<BlockIdxType>& blocks) {
            std::vector<FrozenTask::BlockInfo> saved_blocks;
            saved_blocks.reserve(blocks.size());
            for (size_t member = 0; member < blocks.size(); ++member) {
                std::shared_ptr<IBlockPool> pool;
                if (group) {
                    if (tier == Tier::DEVICE && member < group->devicePools().size())
                        pool = group->devicePools()[member];
                    else if (tier == Tier::HOST)
                        pool = group->hostPool();
                    else if (tier == Tier::DISK)
                        pool = group->diskPool();
                }
                saved_blocks.push_back({pool, blocks[member], member, pool ? pool->poolName() : "unresolved"});
            }
            return saved_blocks;
        };
        saved.sources = capture(descriptor.source_tier, descriptor.source_blocks);
        saved.targets = capture(descriptor.target_tier, descriptor.target_blocks);
        info.descriptors.push_back(std::move(saved));
    }
    return info;
}

void BlockTransferDispatcher::submitCoordinated(TransferTask task,
                                               TransferDoneCallback done,
                                               const char* completion_operation) const {
    auto execution = std::make_shared<CoordinatedExecution>(
        this, describeExecution(task, completion_operation), std::move(done));
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
            // The outer batch still owns error classification and settlement.
            // Preserve this B's cleanup candidate even without a returned context.
            execution->cleanup.submit();
            throw;
        }
        try {
            if (context && context->done()) {
                execution->complete(context->result());
                return;
            }
        } catch (...) {
            // A failed observation cannot prove a submitted execution safe.
        }
        try {
            observation_failure.execution_diagnostics = context->executionDiagnostics();
        } catch (...) {
            // Keep expected participants as unknown if observation also fails.
        }
        execution->complete(std::move(observation_failure));
    }
}

void BlockTransferDispatcher::retainFrozenTask(std::list<std::shared_ptr<FrozenTask>>& prepared_slot) const noexcept {
    std::lock_guard<std::mutex> lock(frozen_mutex_);
    frozen_tasks_.splice(frozen_tasks_.end(), prepared_slot);
}

std::vector<std::shared_ptr<FrozenTask>> BlockTransferDispatcher::frozenTasks() const {
    // Keep retired owners alive until the list lock is released, including if
    // snapshot allocation throws. Existing snapshots still own their diagnostics.
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
    submitCoordinated(task.rpcSubtask(std::move(task.mutableDescriptorsForPreparation())),
                      std::move(callback), completion_operation);
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
