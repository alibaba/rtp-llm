#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <limits>
#include <memory>
#include <optional>
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
                         const TransferTask&            task,
                         const char*                    completion_operation,
                         TransferDoneCallback           done):
        owner(owner),
        task(task),
        operation(completion_operation ? completion_operation : "transfer"),
        done(std::move(done)) {
        prepared_slot.emplace_back();
        if (owner->completion_cleanup_) {
            TransferCompletionTarget target;
            target.remote = owner->multi_rank_engine_ != nullptr;
            if (target.remote) {
                const auto& addresses = owner->multi_rank_engine_->participantAddresses();
                target.peers.reserve(addresses.size());
                for (size_t rank = 0; rank < addresses.size(); ++rank)
                    target.peers.push_back({rank, addresses[rank]});
            } else {
                target.peers.push_back({0, "local"});
            }
            cleanup = owner->completion_cleanup_->prepare(std::move(target), task.traceId());
        }
    }

    void complete(TransferResult result) {
        if (claimed.exchange(true, std::memory_order_acq_rel))
            return;
        try {
            result.expandBlockErrors(task->descriptors().size());
        } catch (...) {
            // Empty block_errors is a uniform UNKNOWN without another allocation.
            // Establish custody before delivering this diagnostic fallback.
            const bool safe = result.buffer_safe;
            result          = TransferResult::executionException(safe);
            result.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
            result.copy_error = CacheCopyError::RPC_FAILED;
        }
        try {
            if (result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
                // The immutable task snapshot survives worker preparation. Resolve
                // physical pools only when freezing, before business settlement.
                auto execution = owner->buildExecutionInfo(*task, operation.c_str());
                if (!result.execution_diagnostics) {
                    auto diagnostics      = std::make_shared<TransferExecutionDiagnostics>();
                    diagnostics->trace_id = execution.trace_id;
                    for (const auto& peer : execution.expected_peers) {
                        TransferRankSnapshot rank;
                        rank.rank           = peer.rank;
                        rank.server_address = peer.address;
                        rank.remote         = execution.remote;
                        rank.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
                        if (!execution.remote) {
                            rank.worker_error_code    = static_cast<int64_t>(result.status.code());
                            rank.worker_error_message = result.status.ToString();
                            rank.worker_copy_error    = result.copy_error;
                            rank.block_errors         = result.block_errors;
                            rank.buffer_safe          = result.buffer_safe;
                        }
                        diagnostics->ranks.push_back(std::move(rank));
                    }
                    result.execution_diagnostics = std::move(diagnostics);
                }
                // All allocation/validation precedes the first frozen hold. If
                // preparation fails, never turn an unprotected UNKNOWN into a
                // business failure that would release the ordinary references.
                auto frozen =
                    std::make_shared<FrozenTask>(FrozenTask::prepare(std::move(execution)), result, std::move(cleanup));
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
        task.reset();
        // A retained or late child callback must not retain the business owner
        // after completion. Invoke outside all registry/pool/frozen-task locks.
        auto       notify  = std::move(done);
        const bool unknown = result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN;
        try {
            notify(std::move(result));
        } catch (...) {
            if (unknown) {
                std::fputs("unknown transfer business completion could not be delivered safely\n", stderr);
                std::fflush(stderr);
                std::abort();
            }
            {
                std::lock_guard<std::mutex> lock(consumer_exception_mutex);
                consumer_exception = std::current_exception();
            }
            throw;
        }
    }

    const BlockTransferDispatcher*         owner;
    std::optional<TransferTask>            task;
    std::string                            operation;
    TransferDoneCallback                   done;
    TransferCompletionCleanup::Prepared    cleanup;
    std::list<std::shared_ptr<FrozenTask>> prepared_slot;
    std::atomic<bool>                      claimed{false};
    std::mutex                             consumer_exception_mutex;
    std::exception_ptr                     consumer_exception;
};

std::shared_ptr<BlockTransferDispatcher::InflightTicket> BlockTransferDispatcher::trackTransfer() const {
    return std::make_shared<InflightTicket>(*this);
}

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t                                        max_descriptors_per_batch):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    make_observed_context_([](size_t count) { return std::make_shared<TransferBatchAsyncContext>(count); }) {
    RTP_LLM_LOG_INFO("transfer_lifecycle event=dispatcher_init registry=%s engine=%p",
                     task_registry_->epoch().c_str(),
                     static_cast<const void*>(per_rank_engine_.get()));
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
    const auto trace_id         = std::to_string(numeric_trace_id);
    RTP_LLM_LOG_INFO("transfer_lifecycle event=worker_execute_enter trace=%s request=%ld registry=%s",
                     trace_id.c_str(),
                     task.requestId().value_or(-1),
                     task_registry_->epoch().c_str());
    // Allocate the owner and its completion path before registering execution.
    // The safety callback owns the registry, not the dispatcher's business lifetime.
    const auto ticket  = trackTransfer();
    auto       context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    auto       proof   = std::make_shared<TransferBufferSafety>(
        [registry = task_registry_, trace_id] { registry->finish(trace_id, true); });
    if (numeric_trace_id == 0 || !task_registry_->begin(trace_id)) {
        const auto* reason = numeric_trace_id == 0 ? "empty transfer trace" : "duplicate transfer trace";
        RTP_LLM_LOG_WARNING("%s: trace=%s", reason, trace_id.c_str());
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, std::string(reason) + ": " + trace_id),
                              CacheCopyError::NOT_EXECUTED);
        // This input never reaches I/O. Do not finish or overwrite the existing trace.
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(std::move(result), task.descriptors().size());
    }
    block_tree_cache_detail::ScopeRollback close_submission([proof] { proof->report(true); });
    warnRegistrySize();
    if (task.expired()) {
        TransferResult result(
            ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before TE submission"),
            CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;  // No batch was submitted.
        proof->report(true);
        context->complete(std::move(result));
        return context;
    }
    TransferAsyncContext::DoneCallback complete = [ticket, context](TransferResult result) {
        context->complete(std::move(result));
    };
    const auto                                 evidence = local_copy_observation_;
    std::shared_ptr<TransferBatchAsyncContext> observed;
    TransferAsyncContext::DoneCallback         observe;
    if (task.operation && evidence) {
        try {
            observed = make_observed_context_(task.descriptors().size());
            observe  = [task, evidence, observed](TransferResult result) {
                try {
                    evidence->observer(task, result);
                } catch (...) {
                    evidence->complete.store(false);
                }
                observed->complete(std::move(result));
            };
        } catch (...) {
            evidence->complete.store(false);
        }
    }
    TransferResult observation_failure(
        ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "failed to observe submitted local transfer completion"));
    task.setCompletionProof(std::move(proof));
    close_submission.dismiss();
    auto child = per_rank_engine_->execute(std::move(task));
    if (!child) {
        context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
    } else {
        try {
            child->onDone(complete);
        } catch (...) {
            if (evidence) {
                evidence->complete.store(false);
            }
            // The producer still owns its proof even when business observation fails.
            try {
                if (child->done()) {
                    complete(child->result());
                    return context;
                }
            } catch (...) {}
            complete(std::move(observation_failure));
        }
    }
    if (!observe) {
        return context;
    }
    observed->holdUntilCompletion(context);
    try {
        context->onDone(std::move(observe));
    } catch (...) {
        evidence->complete.store(false);
        return context;
    }
    return observed;
}

void BlockTransferDispatcher::warnRegistrySize() const noexcept {
    try {
        const auto count     = task_registry_->size();
        auto       threshold = next_registry_warning_.load(std::memory_order_relaxed);
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

FrozenTask::ExecutionInfo BlockTransferDispatcher::buildExecutionInfo(const TransferTask& task,
                                                                      const char*         completion_operation) const {
    FrozenTask::ExecutionInfo info;
    info.trace_id        = task.traceId();
    info.operation       = completion_operation ? completion_operation : "transfer";
    info.request_id      = task.requestId();
    info.workflow_id     = task.workflowId();
    info.parent_trace_id = task.parentTraceId();
    info.remote          = multi_rank_engine_ != nullptr;
    if (multi_rank_engine_) {
        const auto& addresses = multi_rank_engine_->participantAddresses();
        info.expected_peers.reserve(addresses.size());
        for (size_t rank = 0; rank < addresses.size(); ++rank)
            info.expected_peers.push_back({rank, addresses[rank]});
    } else {
        info.expected_peers.push_back({0, "local"});
    }
    info.expected_rank_count = info.expected_peers.size();
    const auto& groups       = multi_rank_engine_ ? multi_rank_engine_->groupSets() : per_rank_engine_->groupSets();
    info.descriptors.reserve(task.descriptors().size());
    for (const auto& descriptor : task.descriptors()) {
        FrozenTask::DescriptorInfo saved;
        saved.group_set_id = descriptor.group_set_id;
        saved.path_index   = descriptor.path_index;
        saved.source_tier  = descriptor.source_tier;
        saved.target_tier  = descriptor.target_tier;
        const GroupSet* group =
            descriptor.group_set_id < groups.size() ? groups[descriptor.group_set_id].get() : nullptr;
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
                saved_blocks.push_back({pool, blocks[member], member});
            }
            return saved_blocks;
        };
        saved.sources = capture(descriptor.source_tier, descriptor.source_blocks);
        saved.targets = capture(descriptor.target_tier, descriptor.target_blocks);
        info.descriptors.push_back(std::move(saved));
    }
    return info;
}

void BlockTransferDispatcher::submitCoordinated(TransferTask         task,
                                                TransferDoneCallback done,
                                                const char*          completion_operation) const {
    auto execution = std::make_shared<CoordinatedExecution>(this, task, completion_operation, std::move(done));
    // Construct the subscription before starting any I/O. The child's callback
    // vector may still reject it; that path also passes through this finalizer.
    TransferAsyncContext::DoneCallback complete = [execution](TransferResult result) {
        execution->complete(std::move(result));
    };
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
        {
            std::lock_guard<std::mutex> lock(execution->consumer_exception_mutex);
            // A concurrently completed callback and an unrelated subscription
            // failure do not identify the same exception.
            if (execution->consumer_exception == std::current_exception())
                throw;
        }
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
        if (!result.ok()) {
            try {
                RTP_LLM_LOG_ERROR(
                    "block transfer failed: descriptor_count=%zu, error_type=%s, error_code=%d, details=%s",
                    block_count,
                    cacheCopyErrorName(result.copy_error),
                    static_cast<int>(result.status.code()),
                    result.status.ToString().c_str());
            } catch (...) {
                // Optional diagnostics cannot prevent mandatory business completion.
            }
        }
        done(std::move(result));
    };
    const auto descriptors = task.descriptors();
    if (descriptors.empty()) {
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, "empty transfer task"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;
        callback(std::move(result));
        return;
    }
    const bool split_store = task.operation == CacheTransferOperation::STORE && !descriptors.empty()
                             && std::any_of(descriptors.begin(), descriptors.end(), [](const TransferDescriptor& d) {
                                    return d.target_tier == Tier::DISK || d.group_set_id != 0;
                                });
    const bool needs_split = !descriptors.empty() && (max_descriptors_per_batch_ > 0 || split_store);
    if (!needs_split) {
        submitCoordinated(task.rpcSubtask(), std::move(callback), completion_operation);
        return;
    }

    struct DescriptorGroup {
        Tier                            source;
        Tier                            target;
        size_t                          group_set_id;
        std::vector<TransferDescriptor> descriptors;
        std::vector<size_t>             indices;
    };
    std::vector<DescriptorGroup> groups;
    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        const bool  separate   = task.operation == CacheTransferOperation::STORE && descriptor.target_tier == Tier::DISK;
        const auto  group      = separate ?
                                     groups.end() :
                                     std::find_if(groups.begin(), groups.end(), [&descriptor](const DescriptorGroup& item) {
                                      return item.source == descriptor.source_tier
                                             && item.target == descriptor.target_tier
                                             && item.group_set_id == descriptor.group_set_id;
                                  });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
            group->indices.push_back(index);
        } else {
            groups.push_back(DescriptorGroup{
                descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id, {descriptor}, {index}});
        }
    }

    auto stage_state = std::make_shared<TransferStageState>(std::move(callback), descriptors.size());
    struct Batch {
        TransferTask        task;
        std::vector<size_t> indices;
    };
    std::vector<Batch> batches;
    for (const auto& group : groups) {
        const size_t batch_limit = max_descriptors_per_batch_ > 0 ? max_descriptors_per_batch_ : group.descriptors.size();
        for (size_t begin = 0; begin < group.descriptors.size(); begin += batch_limit) {
            const size_t end = std::min(begin + batch_limit, group.descriptors.size());
            auto         subtask =
                task.rpcSubtask({group.descriptors.begin() + begin, group.descriptors.begin() + end});
            std::vector<size_t> indices(group.indices.begin() + begin, group.indices.begin() + end);
            batches.push_back({std::move(subtask), std::move(indices)});
        }
    }
    for (auto& batch : batches) {
        stage_state->addBatch();
        try {
            submitCoordinated(
                std::move(batch.task),
                [stage_state, indices = std::move(batch.indices)](TransferResult result) {
                    stage_state->completeBatch(std::move(result), indices);
                },
                completion_operation);
        } catch (...) {
            TransferResult failure;
            failure.status.setErrorCode(ErrorCode::EXECUTION_EXCEPTION);
            failure.copy_error = CacheCopyError::NOT_EXECUTED;
            if (multi_rank_engine_) {
                failure.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
                failure.copy_error = CacheCopyError::RPC_FAILED;
            }
            stage_state->completeBatch(std::move(failure), batch.indices);
        }
    }
    stage_state->finishSubmitting();
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

RemoteMockSafetySnapshot BlockTransferDispatcher::remoteMockSafety() const {
    return multi_rank_engine_ ? multi_rank_engine_->remoteMockSafety() : RemoteMockSafetySnapshot{};
}

BlockTreeQueueSizes BlockTransferDispatcher::queueSizes() const {
    return per_rank_engine_->queueSizes();
}

}  // namespace rtp_llm
