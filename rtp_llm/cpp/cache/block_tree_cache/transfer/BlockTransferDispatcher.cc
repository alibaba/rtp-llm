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

struct BlockTransferDispatcher::CompletionTicket {
    explicit CompletionTicket(const BlockTransferDispatcher* owner): owner_(owner) {
        owner_->registerTransfer();
    }

    ~CompletionTicket() noexcept {
        owner_->retireTransfer();
    }

    CompletionTicket(const CompletionTicket&)            = delete;
    CompletionTicket& operator=(const CompletionTicket&) = delete;

private:
    const BlockTransferDispatcher* owner_;
};

// The parent callback and retention node exist before the child can submit I/O.
// Only this coordinator boundary owns freezing; worker-local dispatch never does.
struct BlockTransferDispatcher::CoordinatedExecution {
    CoordinatedExecution(const BlockTransferDispatcher* owner,
                         FrozenTask::ExecutionInfo      execution,
                         TransferDoneCallback           done):
        owner(owner), execution(std::move(execution)), done(std::move(done)) {
        prepared_slot.emplace_back();
        if (this->execution.operation != "store" && owner->completion_cleanup_) {
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
            // Ordinary Store keeps its existing failure settlement; sharing
            // this dispatcher does not opt it into FrozenTask ownership.
            if (execution.operation != "store" && result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
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

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t max_descriptors_per_batch):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
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
    std::unique_lock<std::mutex> lock(completion_mutex_);
    if (inflight_transfers_ != 0 && drain_observer_for_test_) {
        lock.unlock();
        drain_observer_for_test_();
        lock.lock();
    }
    completion_cv_.wait(lock, [this] { return inflight_transfers_ == 0; });
}

void BlockTransferDispatcher::registerTransfer() const {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    RTP_LLM_CHECK(inflight_transfers_ < std::numeric_limits<size_t>::max());
    ++inflight_transfers_;
}

void BlockTransferDispatcher::retireTransfer() const noexcept {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    RTP_LLM_CHECK(inflight_transfers_ != 0);
    if (--inflight_transfers_ == 0) {
        // Notify before unlocking: a drain waiter can destroy the dispatcher as
        // soon as it acquires this mutex and observes zero.
        completion_cv_.notify_all();
    }
}

std::shared_ptr<TransferAsyncContext> BlockTransferDispatcher::executePerRank(TransferTask task) const {
    const auto numeric_trace_id = task.traceId();
    const auto trace_id = std::to_string(numeric_trace_id);
    // Allocate the owner and its completion path before registering execution.
    // The safety callback owns the registry, not the dispatcher's business lifetime.
    const auto ticket = std::make_shared<CompletionTicket>(this);
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
        if (safety) safety->finishSubmitting();
    });
    warnRegistrySize();
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;  // No batch was submitted.
        safety->finishSubmitting();
        context->complete(std::move(result));
        return context;
    }
    // The final callback releases this ticket after notifying all local consumers;
    // the submission stack retains another owner through finishSubmitting().
    try {
        dispatchGroups(std::move(task),
                       [ticket, context](TransferResult result) { context->complete(std::move(result)); },
                       true,
                       nullptr,
                       safety);
    } catch (...) {
        // Closes admission only. Any attempted batch still needs its own proof.
        safety->finishSubmitting();
        throw;
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
    const auto& groups = multi_rank_engine_ ? multi_rank_engine_->group_sets_ : per_rank_engine_->groupSets();
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
            return;
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
    const auto ticket = std::make_shared<CompletionTicket>(this);
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
    dispatchGroups(std::move(task), std::move(callback), false, completion_operation);
}

void BlockTransferDispatcher::dispatchGroups(TransferTask         task,
                                             TransferDoneCallback callback,
                                             bool                 local,
                                             const char*          completion_operation,
                                             std::shared_ptr<TransferBufferSafetyTracker> safety) const {
    block_tree_cache_detail::ScopeRollback close_submission([safety] {
        if (safety)
            safety->finishSubmitting();
    });
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, "transfer task contains no descriptors"));
        result.buffer_safe = true;
        close_submission.run();
        callback(std::move(result));
        return;
    }
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before dispatch"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;  // No batch was submitted.
        close_submission.run();
        callback(std::move(result));
        return;
    }

    struct DescriptorGroup {
        Tier                            source{Tier::NONE};
        Tier                            target{Tier::NONE};
        size_t                          group_set_id{0};
        std::vector<TransferDescriptor> descriptors;
        std::vector<size_t>             indices;
    };

    std::vector<DescriptorGroup> groups;
    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        const auto  group =
            std::find_if(groups.begin(), groups.end(), [&descriptor, local](const DescriptorGroup& item) {
                return item.source == descriptor.source_tier && item.target == descriptor.target_tier
                       && (!local || item.group_set_id == descriptor.group_set_id);
            });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
            group->indices.push_back(index);
        } else {
            groups.push_back(DescriptorGroup{
                descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id, {descriptor}, {index}});
        }
    }

    auto stage_state          = std::make_shared<TransferStageState>(std::move(callback), descriptors.size());
    bool any_batch_registered = false;
    for (const auto& group : groups) {
        const size_t batch_limit =
            local ? std::min(max_descriptors_per_batch_, per_rank_engine_->batchLimit()) : group.descriptors.size();
        for (size_t begin = 0; begin < group.descriptors.size(); begin += batch_limit) {
            std::vector<size_t> indices;
            bool                batch_registered = false;
            TransferResult      batch_error;
            TransferDoneCallback complete_batch;
            std::shared_ptr<TransferAsyncContext> context;
            std::shared_ptr<TransferBufferSafety> proof;
            bool execution_attempted = false;
            TransferResult observation_failure;
            try {
                const size_t                    end = std::min(begin + batch_limit, group.descriptors.size());
                std::vector<TransferDescriptor> batch(group.descriptors.begin() + begin,
                                                      group.descriptors.begin() + end);
                indices.assign(group.indices.begin() + begin, group.indices.begin() + end);
                if (safety)
                    proof = safety->addBatch();
                auto completed = std::make_shared<std::atomic<bool>>(false);
                complete_batch = [stage_state, indices, completed](TransferResult result) {
                    if (completed->exchange(true, std::memory_order_acq_rel))
                        return;
                    stage_state->completeBatch(std::move(result), indices);
                };
                observation_failure = TransferResult(ErrorInfo(
                    ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted local transfer completion"));
                stage_state->addBatch();
                batch_registered     = true;
                any_batch_registered = true;
                if (!local) {
                    submitCoordinated(task.rpcSubtask(std::move(batch)), complete_batch, completion_operation);
                    continue;
                }
                auto batch_task = task.subtask(std::move(batch));
                batch_task.setCompletionProof(proof);
                execution_attempted = true;
                context = per_rank_engine_->execute(std::move(batch_task));
                if (context == nullptr) {
                    batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context");
                } else {
                    context->onDone(complete_batch);
                }
            } catch (const std::exception& error) {
                if (proof && !execution_attempted)
                    proof->report(true);
                if (!any_batch_registered) {
                    throw;
                }
                try {
                    if (context) {
                        batch_error = std::move(observation_failure);
                    } else {
                        batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what());
                    }
                } catch (...) {
                    std::terminate();
                }
            } catch (...) {
                if (proof && !execution_attempted)
                    proof->report(true);
                if (!any_batch_registered) {
                    throw;
                }
                try {
                    if (context) {
                        batch_error = std::move(observation_failure);
                    } else {
                        batch_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                batch_registered ? "unknown transfer submission exception" :
                                                                   "unknown transfer preparation exception");
                    }
                } catch (...) {
                    std::terminate();
                }
            }

            if (!batch_error.ok()) {
                if (!batch_registered) {
                    // Stop new submissions and join the batches already registered.
                    close_submission.run();
                    stage_state->finishSubmitting(std::move(batch_error));
                    return;
                }
                if (context) {
                    try {
                        if (context->done()) {
                            complete_batch(context->result());
                            continue;
                        }
                    } catch (...) {
                        // Completion remains unknown; preserve protection before
                        // the coordinator can notify the business owner.
                    }
                }
                complete_batch(std::move(batch_error));
            }
        }
    }
    // Final callback exceptions must propagate without re-entering submission handling.
    close_submission.run();
    stage_state->finishSubmitting();
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
