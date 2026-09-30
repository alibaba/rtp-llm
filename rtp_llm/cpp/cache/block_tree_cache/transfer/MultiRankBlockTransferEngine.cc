#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using TransferBroadcastResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

class MultiRankTransferAsyncContext final:
    public AsyncContext,
    public std::enable_shared_from_this<MultiRankTransferAsyncContext> {
public:
    static std::shared_ptr<MultiRankTransferAsyncContext> create(std::shared_ptr<TransferBroadcastResult> result,
                                                                 size_t                                   worker_count,
                                                                 bool protected_transfer,
                                                                 bool require_confirmed_completion) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(new MultiRankTransferAsyncContext(
            std::move(result), worker_count, protected_transfer, require_confirmed_completion));
        context->start();
        return context;
    }

    void waitDone() override {
        try {
            result_->waitDone();
        } catch (const std::exception& error) {
            evaluate(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
        } catch (...) {
            evaluate(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown multi-rank transfer failure"));
        }
        evaluate();
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return evaluated_; });
    }

    void onDone(DoneCallback callback) override {
        if (!callback) {
            return;
        }
        bool      run_now = false;
        ErrorInfo error   = ErrorInfo::OkStatus();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_) {
                run_now = true;
                error   = error_;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback(std::move(error));
        }
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return evaluated_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return evaluated_ && error_.ok();
    }

    ErrorInfo errorInfo() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_;
    }

private:
    MultiRankTransferAsyncContext(std::shared_ptr<TransferBroadcastResult> result,
                                  size_t                                   worker_count,
                                  bool                                     protected_transfer,
                                  bool                                     require_confirmed_completion):
        result_(std::move(result)),
        worker_count_(worker_count),
        protected_transfer_(protected_transfer),
        require_confirmed_completion_(require_confirmed_completion) {}

    void start() {
        std::shared_ptr<MultiRankTransferAsyncContext> self = shared_from_this();
        result_->onDone([self = std::move(self)] { self->evaluate(); });
    }

    void evaluate(ErrorInfo forced_error = ErrorInfo::OkStatus()) {
        // An RPC failure/deadline does not prove the remote GPU stopped writing
        // our targets. Until there is a remote drain acknowledgement protocol,
        // protected transfers must fail-stop instead of releasing those targets.
        if (require_confirmed_completion_ && (!forced_error.ok() || (result_->done() && !result_->success()))) {
            RTP_LLM_LOG_ERROR("protected cache transfer lost remote completion; refusing unsafe target reuse");
            std::abort();
        }
        ErrorInfo error = std::move(forced_error);
        if (error.ok()) {
            if (!result_->done()) {
                return;
            }
            if (!result_->success()) {
                if (StaticConfig::user_ft_core_dump_on_exception) {
                    RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK");
                }
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer RPC failed");
            }
            // Inspect every completed response even if another rank had an RPC
            // failure. A typed integrity failure must dominate a generic error.
            const auto responses = result_->responses();
            if (require_confirmed_completion_ && responses.size() != worker_count_) {
                RTP_LLM_LOG_ERROR("protected cache transfer response count mismatch; completion is ambiguous");
                std::abort();
            }
            if (responses.size() != worker_count_ && error.ok()) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer response count mismatch");
            }
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                if (responses[rank].has_mem_response()
                    && responses[rank].mem_response().code() == MemoryOperationResponsePB::CACHE_INTEGRITY_ERROR) {
                    error = ErrorInfo(ErrorCode::CACHE_INTEGRITY_ERROR,
                                      "multi-rank cache integrity failure, rank=" + std::to_string(rank));
                } else if (error.ok()
                           && (!responses[rank].has_mem_response()
                               || responses[rank].mem_response().code() != MemoryOperationResponsePB::OK)) {
                    error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "multi-rank transfer failed, rank=" + std::to_string(rank));
                }
            }
        }

        if (protected_transfer_ && !error.ok()) {
            error = ErrorInfo(ErrorCode::CACHE_INTEGRITY_ERROR, error.ToString());
        }
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_) {
                return;
            }
            error_     = error;
            evaluated_ = true;
            callbacks.swap(callbacks_);
        }
        cv_.notify_all();
        for (auto& callback : callbacks) {
            callback(error);
        }
    }

    std::shared_ptr<TransferBroadcastResult> result_;
    size_t                                   worker_count_{0};
    bool                                     protected_transfer_{false};
    bool                                     require_confirmed_completion_{false};
    mutable std::mutex                       mutex_;
    std::condition_variable                  cv_;
    ErrorInfo                                error_{ErrorInfo::OkStatus()};
    bool                                     evaluated_{false};
    std::vector<DoneCallback>                callbacks_;
};

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>          group_sets,
                                                           std::shared_ptr<BroadcastManager> broadcast_manager):
    group_sets_(std::move(group_sets)), broadcast_manager_(std::move(broadcast_manager)) {}

void MultiRankBlockTransferEngine::requireProtectedCompletion(TransferTask& task) const {
    for (const auto& descriptor : task.descriptors()) {
        if (descriptor.group_set_id < group_sets_.size() && group_sets_[descriptor.group_set_id]
            && group_sets_[descriptor.group_set_id]->crcEnabled()) {
            task.requireConfirmedCompletion();
            return;
        }
    }
}

std::shared_ptr<AsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task) const {
    requireProtectedCompletion(task);
    bool protected_transfer = false;
    bool ordinary_transfer  = false;
    for (const auto& desc : task.descriptors()) {
        if (desc.group_set_id < group_sets_.size() && group_sets_[desc.group_set_id]) {
            if (group_sets_[desc.group_set_id]->crcEnabled()) {
                protected_transfer = true;
            } else {
                ordinary_transfer = true;
            }
        }
    }
    if (protected_transfer && ordinary_transfer) {
        // One wire request has one integrity mode. Preserve the backing mode
        // when a caller batches CUDA and host-resident GroupSets together.
        std::vector<TransferDescriptor> protected_descriptors, ordinary_descriptors;
        const auto&                     first = task.descriptors().front();
        for (const auto& desc : task.descriptors()) {
            if (!desc.isExecutable() || desc.group_set_id >= group_sets_.size() || !group_sets_[desc.group_set_id]
                || desc.source_tier != first.source_tier || desc.target_tier != first.target_tier) {
                return std::make_shared<CompletedAsyncContext>(
                    ErrorInfo(ErrorCode::CACHE_INTEGRITY_ERROR, "invalid mixed-mode transfer batch"));
            }
            (group_sets_[desc.group_set_id]->crcEnabled() ? protected_descriptors : ordinary_descriptors)
                .push_back(desc);
        }
        std::vector<std::shared_ptr<AsyncContext>> contexts;
        contexts.reserve(2);
        try {
            contexts.push_back(execute(task.subtask(std::move(protected_descriptors))));
            contexts.push_back(execute(task.subtask(std::move(ordinary_descriptors))));
            // FusedAsyncContext waits for both modes and gives integrity errors
            // priority over ordinary errors regardless of completion order.
            return std::make_shared<FusedAsyncContext>(contexts);
        } catch (...) {
            // A submitted subrequest may still write caller-owned blocks. Keep
            // the same fail-stop policy as a protected RPC dispatch exception.
            RTP_LLM_LOG_ERROR("mixed-mode cache RPC dispatch threw; remote completion is ambiguous");
            std::abort();
        }
    }
    const auto deadline_exceeded = [protected_transfer]() {
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(protected_transfer ? ErrorCode::CACHE_INTEGRITY_ERROR : ErrorCode::DEADLINE_EXCEEDED,
                      "transfer deadline exceeded before multi-rank submission"));
    };
    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, task, group_sets_)) {
        if (task.expired()) {
            return deadline_exceeded();
        }
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", task.descriptors().size());
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(protected_transfer ? ErrorCode::CACHE_INTEGRITY_ERROR : ErrorCode::INVALID_PARAMS,
                      "failed to encode transfer batch"));
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
            });
    } catch (...) {
        // BroadcastManager can throw after submitting an earlier rank. With no
        // result object there is no way to drain those remote writes safely.
        if (task.requiresConfirmedCompletion()) {
            RTP_LLM_LOG_ERROR("protected cache RPC dispatch threw; remote completion is ambiguous");
            std::abort();
        }
        throw;
    }
    if (broadcast_result == nullptr) {
        RTP_LLM_LOG_WARNING("failed to start broadcast");
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(protected_transfer ? ErrorCode::CACHE_INTEGRITY_ERROR : ErrorCode::EXECUTION_EXCEPTION,
                      "failed to start transfer broadcast"));
    }
    try {
        return MultiRankTransferAsyncContext::create(
            std::move(broadcast_result), worker_count, protected_transfer, task.requiresConfirmedCompletion());
    } catch (...) {
        if (task.requiresConfirmedCompletion()) {
            RTP_LLM_LOG_ERROR("protected cache completion context creation failed after RPC submission");
            std::abort();
        }
        throw;
    }
}

}  // namespace rtp_llm
