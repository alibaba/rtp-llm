#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using TransferBroadcastResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

class MultiRankTransferAsyncContext final:
    public TransferAsyncContext,
    public std::enable_shared_from_this<MultiRankTransferAsyncContext> {
public:
    static std::shared_ptr<MultiRankTransferAsyncContext>
    create(const std::vector<std::string>&                 worker_addresses,
           size_t                                         block_count,
           SharedTransferStageTimer                       wait_timer,
           std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
           int64_t                                        request_id,
           uint64_t                                       transfer_id,
           CacheStageContext                              identity,
           const char*                                    completion_operation) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(worker_addresses.size(), block_count));
        context->wait_timer_  = std::move(wait_timer);
        context->reporter_    = std::move(reporter);
        context->request_id_  = request_id;
        context->transfer_id_ = transfer_id;
        context->identity_    = identity;
        context->completion_identity_ = identity;
        if (completion_operation)
            context->completion_identity_.operation = completion_operation;
        auto diagnostics      = std::make_shared<TransferExecutionDiagnostics>();
        diagnostics->trace_id = transfer_id;
        diagnostics->ranks.reserve(worker_addresses.size());
        for (size_t rank = 0; rank < worker_addresses.size(); ++rank) {
            TransferRankSnapshot snapshot;
            snapshot.rank           = rank;
            snapshot.server_address = worker_addresses[rank];
            snapshot.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
            snapshot.dispatch_error = "broadcast dispatch, subscription, or result publication failed";
            diagnostics->ranks.push_back(std::move(snapshot));
        }
        context->unknown_result_ = TransferResult(
            ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "transfer broadcast completion could not be established"),
            CacheCopyError::RPC_FAILED);
        context->unknown_result_.expandBlockErrors(block_count);
        context->unknown_result_.execution_diagnostics = std::move(diagnostics);
        context->unknown_publication_                 = context->unknown_result_;
        context->prepareCallback();
        return context;
    }

    void attach(std::shared_ptr<TransferBroadcastResult> result) {
        broadcast_result_ = std::move(result);
        try {
            if (auto probe = block_tree_cache_detail::before_transfer_broadcast_subscription_for_test.load(
                    std::memory_order_acquire)) {
                probe();
            }
            broadcast_result_->onDone(std::move(completion_callback_));
        } catch (...) {
            completeUnknown();
        }
    }

    void completeUnknown() {
        // Both results and the callback were prepared before submitting RPCs.
        // Registration failure must not escape as an ordinary execution error.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_ || finishing_ || fallback_started_) {
                return;
            }
            fallback_started_ = true;
        }
        completion_callback_ = {};
        if (broadcast_result_) {
            try {
                auto diagnostics = collectDiagnostics();
                unknown_result_.execution_diagnostics      = diagnostics;
                unknown_publication_.execution_diagnostics = std::move(diagnostics);
            } catch (...) {
                // The preallocated snapshot conservatively marks every rank
                // possibly submitted if collecting diagnostics also fails.
            }
        }
        publish(std::move(unknown_result_), std::move(unknown_publication_));
    }

    void discardUnsubmittedCallback() {
        completion_callback_ = {};
    }

    void waitDone() override {
        if (done()) {
            return;
        }
        try {
            broadcast_result_->waitDone(/*timeout_ms=*/0, /*fatal_on_error=*/false);
        } catch (...) {
            completeUnknown();
        }
        if (!done()) {
            evaluateWithFallback();
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return evaluated_; });
    }

    void onDone(DoneCallback callback) override {
        if (!callback) {
            return;
        }
        bool           run_now = false;
        TransferResult result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_) {
                run_now = true;
                result  = result_;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback(std::move(result));
        }
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return evaluated_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return evaluated_ && result_.ok();
    }

    TransferResult result() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return result_;
    }

    std::shared_ptr<const TransferExecutionDiagnostics> executionDiagnostics() const override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_) {
                return result_.execution_diagnostics;
            }
        }
        return collectDiagnostics();
    }

private:
    MultiRankTransferAsyncContext(size_t                                   worker_count,
                                  size_t                                   block_count):
        worker_count_(worker_count), block_count_(block_count) {}

    void prepareCallback() {
        std::shared_ptr<MultiRankTransferAsyncContext> self = shared_from_this();
        completion_callback_ = [self = std::move(self)] {
            TransferStageTimer total(self->reporter_,
                                     "completion_total",
                                     self->request_id_,
                                     0,
                                     self->block_count_,
                                     -1,
                                     "coordinator_completion",
                                     self->transfer_id_,
                                     self->completion_identity_);
            total.finish(self->evaluateWithFallback());
        };
    }

    bool evaluateWithFallback() {
        try {
            return evaluate();
        } catch (...) {
            // A transient allocation failure before the completion claim can
            // be retried immediately. Async consumers must not depend on a
            // future waitDone() call to retry the only CQ notification.
            try {
                return evaluate();
            } catch (...) {
                completeUnknown();
                return false;
            }
        }
    }

    bool evaluate() {
        if (!broadcast_result_->done()) {
            return false;
        }
        auto result = TransferResult::emptyAggregate();
        if (!broadcast_result_->success()) {
            result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                "multi-rank transfer RPC failed: " + broadcast_result_->failureSummary()),
                      CacheCopyError::RPC_FAILED};
        }

        TransferStageTimer merge(reporter_,
                                 "rank_result_merge",
                                 request_id_,
                                 0,
                                 block_count_,
                                 -1,
                                 "coordinator_completion",
                                 transfer_id_,
                                 completion_identity_);
        result.execution_diagnostics = collectDiagnostics(&result);
        // Copy before claiming completion: allocation failure must leave the
        // context retryable. Once claimed, publication cannot allocate.
        if (auto probe =
                block_tree_cache_detail::before_transfer_publication_copy_for_test.load(std::memory_order_acquire))
            probe();
        auto published_result = result;
        merge.finish(result.ok());
        return publish(std::move(result), std::move(published_result));
    }

    std::shared_ptr<const TransferExecutionDiagnostics> collectDiagnostics(TransferResult* aggregate = nullptr) const {
        auto ignored = TransferResult::emptyAggregate();
        auto& result = aggregate ? *aggregate : ignored;
        const auto snapshots = broadcast_result_->rankSnapshots();
        auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
        diagnostics->trace_id = transfer_id_;
        diagnostics->ranks.reserve(snapshots.size());
        if (snapshots.size() != worker_count_) {
            result.merge({ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "multi-rank transfer response count mismatch"),
                          CacheCopyError::RPC_FAILED});
        }
        result.expandBlockErrors(block_count_);
        for (const auto& snapshot : snapshots) {
            TransferRankSnapshot rank_snapshot;
            rank_snapshot.rank           = snapshot.rank;
            rank_snapshot.server_address = snapshot.server_address;
            switch (snapshot.dispatch_state) {
                case BroadcastDispatchState::NOT_SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::NOT_SUBMITTED;
                    break;
                case BroadcastDispatchState::POSSIBLY_SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
                    break;
                case BroadcastDispatchState::SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::SUBMITTED;
                    break;
            }
            rank_snapshot.dispatch_error = snapshot.dispatch_error;
            rank_snapshot.cq_received    = snapshot.cq_received;
            rank_snapshot.cq_ok          = snapshot.cq_ok;
            rank_snapshot.grpc_code      = snapshot.grpc_code;
            rank_snapshot.grpc_message   = snapshot.grpc_message;
            if (!snapshot.cq_received || !snapshot.cq_ok || !snapshot.grpc_code
                || *snapshot.grpc_code != grpc::StatusCode::OK || !snapshot.response) {
                diagnostics->ranks.push_back(std::move(rank_snapshot));
                continue;
            }
            rank_snapshot.response_received = true;
            const auto& response = *snapshot.response;
            TransferResult rank_result{
                ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "invalid transfer response"),
                CacheCopyError::RPC_FAILED};
            if (response.has_mem_response()) {
                const auto& memory_response = response.mem_response();
                if (memory_response.has_error_info()) {
                    rank_snapshot.worker_error_code    = memory_response.error_info().error_code();
                    rank_snapshot.worker_error_message = memory_response.error_info().error_message();
                }
                bool valid = MemoryOperationResponsePB::Code_IsValid(memory_response.code())
                             && static_cast<size_t>(memory_response.block_errors_size()) == block_count_;
                if (!MemoryOperationResponsePB::Code_IsValid(memory_response.code())) {
                    rank_snapshot.response_error = "invalid worker copy error code: "
                                                   + std::to_string(memory_response.code());
                } else {
                    rank_snapshot.worker_copy_error = BlockTransferRequestConverter::decodeCode(memory_response.code());
                    if (static_cast<size_t>(memory_response.block_errors_size()) != block_count_) {
                        rank_snapshot.response_error = "worker block error count mismatch: expected="
                                                       + std::to_string(block_count_) + " actual="
                                                       + std::to_string(memory_response.block_errors_size());
                    }
                }
                for (int index = 0; valid && index < memory_response.block_errors_size(); ++index) {
                    valid = MemoryOperationResponsePB::Code_IsValid(memory_response.block_errors(index));
                    if (!valid) {
                        rank_snapshot.response_error = "invalid worker block error code at index="
                                                       + std::to_string(index) + " code="
                                                       + std::to_string(memory_response.block_errors(index));
                    }
                }
                if (valid) {
                    try {
                        rank_result = BlockTransferRequestConverter::decodeTransferResult(memory_response, block_count_);
                        rank_snapshot.block_errors   = rank_result.block_errors;
                        rank_snapshot.buffer_safe    = rank_result.buffer_safe;
                        rank_snapshot.response_valid = true;
                    } catch (const std::exception& error) {
                        rank_snapshot.response_error = "transfer response decode failed: " + std::string(error.what());
                        rank_result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                                 rank_snapshot.response_error),
                                       CacheCopyError::RPC_FAILED};
                    } catch (...) {
                        rank_snapshot.response_error = "unknown transfer response decode failure";
                        rank_result = {
                            ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "unknown transfer response decode failure"),
                            CacheCopyError::RPC_FAILED};
                    }
                }
            } else {
                rank_snapshot.response_error = "transfer response contains no memory response";
            }
            diagnostics->ranks.push_back(std::move(rank_snapshot));
            for (size_t index = 0; index < block_count_; ++index) {
                result.block_errors[index] = std::max(result.block_errors[index], rank_result.blockError(index));
            }
            if (!rank_result.ok()) {
                rank_result.status =
                    ErrorInfo(rank_result.status.code(),
                              "rank=" + std::to_string(snapshot.rank) + " {" + rank_result.status.ToString() + "}");
            }
            result.merge(std::move(rank_result));
        }
        return diagnostics;
    }

    bool publish(TransferResult result, TransferResult published_result) {
        static_assert(std::is_nothrow_move_assignable_v<TransferResult>);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_ || finishing_) {
                // An unpublished result owned by another evaluator is not a
                // failure of this completed response evaluation.
                return result.ok();
            }
            finishing_ = true;
        }
        if (auto probe =
                block_tree_cache_detail::after_transfer_completion_claim_for_test.load(std::memory_order_acquire))
            probe();
        if (wait_timer_) {
            wait_timer_.finish(result.ok());
            try {
                const auto times = broadcast_result_ ? broadcast_result_->completionTimesUs() : std::vector<int64_t>{};
                if (times.size() > 1 && std::all_of(times.begin(), times.end(), [](int64_t t) { return t > 0; })) {
                    const auto    bounds  = std::minmax_element(times.begin(), times.end());
                    const int64_t skew_us = *bounds.second - *bounds.first;
                    if (reporter_) {
                        reporter_->reportTransferStage("rank_completion_skew",
                                                       skew_us,
                                                       result.ok(),
                                                       -1,
                                                       "coordinator",
                                                       identity_.operation,
                                                       identity_.source,
                                                       identity_.target);
                    }
                    const auto       wall_times = broadcast_result_->completionWallTimesUs();
                    const auto       end_us     = wall_times[std::distance(times.begin(), bounds.second)];
                    AsyncProfileSpan span("rank_completion_skew",
                                          request_id_,
                                          0,
                                          block_count_,
                                          end_us - skew_us,
                                          -1,
                                          transfer_id_,
                                          identity_.path,
                                          AsyncProfileKind::Derived);
                    span.finish(end_us);
                }
            } catch (...) {
                // Diagnostics must not prevent completion notification.
            }
        }
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result_    = std::move(published_result);
            evaluated_ = true;
            callbacks.swap(callbacks_);
        }
        cv_.notify_all();
        TransferStageTimer dispatch(reporter_,
                                    "callback_dispatch",
                                    request_id_,
                                    0,
                                    block_count_,
                                    -1,
                                    "coordinator_completion",
                                    transfer_id_,
                                    completion_identity_);
        for (auto& callback : callbacks) {
            try {
                callback(result);
            } catch (...) {
                RTP_LLM_LOG_WARNING("transfer completion callback failed");
            }
        }
        dispatch.finish(result.ok());
        return result.ok();
    }

    std::shared_ptr<TransferBroadcastResult>       broadcast_result_;
    size_t                                         worker_count_{0};
    size_t                                         block_count_{0};
    mutable std::mutex                             mutex_;
    std::condition_variable                        cv_;
    TransferResult                                 result_;
    TransferResult                                 unknown_result_;
    TransferResult                                 unknown_publication_;
    TransferBroadcastResult::DoneCallback           completion_callback_;
    bool                                           evaluated_{false};
    bool                                           finishing_{false};
    bool                                           fallback_started_{false};
    std::vector<DoneCallback>                      callbacks_;
    SharedTransferStageTimer                       wait_timer_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> reporter_;
    int64_t                                        request_id_{-1};
    uint64_t                                       transfer_id_{0};
    CacheStageContext                              identity_;
    CacheStageContext                              completion_identity_;
};

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(
    std::vector<GroupSetPtr>                       group_sets,
    std::shared_ptr<BroadcastManager>              broadcast_manager,
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    group_sets_(std::move(group_sets)),
    broadcast_manager_(std::move(broadcast_manager)),
    metrics_reporter_(std::move(metrics_reporter)) {}

std::vector<std::string> MultiRankBlockTransferEngine::participantAddresses() const {
    return broadcast_manager_ ? broadcast_manager_->workerAddresses() : std::vector<std::string>{};
}

std::shared_ptr<TransferAsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task,
                                                                            const char*  completion_operation) const {
    const bool   has_descriptors = !task.descriptors().empty();
    const auto   identity        = has_descriptors ? CacheStageContext::local(task.descriptors().front().source_tier,
                                                                              task.descriptors().front().target_tier) :
                                                     CacheStageContext{};
    auto         wait_timer        = has_descriptors ? TransferStageTimer::create(metrics_reporter_,
                                                                                  "multi_rank_wait",
                                                                                  task.requestId().value_or(-1),
                                                                                  0,
                                                                                  task.descriptors().size(),
                                                                                  -1,
                                                                                  "coordinator",
                                                                                  task.traceId(),
                                                                                  identity,
                                                                                  AsyncProfileKind::Wait) :
                                                       SharedTransferStageTimer{};
    const size_t block_count       = task.descriptors().size();
    const auto   deadline_exceeded = [block_count]() {
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(
                ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before multi-rank submission"),
                CacheCopyError::NOT_EXECUTED),
            block_count);
    };
    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, task, group_sets_)) {
        if (task.expired()) {
            return deadline_exceeded();
        }
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", task.descriptors().size());
        return std::make_shared<CompletedTransferAsyncContext>(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "failed to encode transfer batch"), block_count);
    }
    const size_t      worker_count = broadcast_manager_->workerNum();
    FunctionRequestPB function_request;
    function_request.mutable_mem_request()->CopyFrom(request);
    std::vector<FunctionRequestPB> requests(worker_count, function_request);
    const auto                     remaining = task.remainingTimeout();
    if (!remaining) {
        return deadline_exceeded();
    }
    const int broadcast_timeout_ms = static_cast<int>(remaining->count());
    for (auto& rank_request : requests) {
        rank_request.mutable_mem_request()->set_timeout_ms(broadcast_timeout_ms);
    }
    auto context = MultiRankTransferAsyncContext::create(broadcast_manager_->workerAddresses(),
                                                        block_count,
                                                        std::move(wait_timer),
                                                        metrics_reporter_,
                                                        task.requestId().value_or(-1),
                                                        task.traceId(),
                                                        identity,
                                                        completion_operation);
    std::shared_ptr<TransferBroadcastResult> broadcast_result;
    try {
        broadcast_result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests,
            broadcast_timeout_ms,
            [](const std::shared_ptr<RpcService::Stub>&    stub,
               const std::shared_ptr<grpc::ClientContext>& context,
               const FunctionRequestPB&                    rpc_request,
               grpc::CompletionQueue*                      completion_queue) {
                return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
            },
            task.requestId(),
            MemoryOperationRequestPB_CopyDirection_Name(request.copy_direction()),
            /*fatal_on_error=*/false);
    } catch (...) {
        context->completeUnknown();
        return context;
    }
    if (broadcast_result == nullptr) {
        context->discardUnsubmittedCallback();
        RTP_LLM_LOG_WARNING("failed to start broadcast");
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "failed to start transfer broadcast"),
                           CacheCopyError::RPC_FAILED),
            block_count);
    }
    context->attach(std::move(broadcast_result));
    return context;
}

}  // namespace rtp_llm
