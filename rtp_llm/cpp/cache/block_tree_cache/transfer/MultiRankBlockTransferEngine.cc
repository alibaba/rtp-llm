#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <algorithm>
#include <condition_variable>
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
                                                                 std::vector<TransferDescriptor>          descriptors) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(std::move(result), worker_count, std::move(descriptors)));
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
                                  std::vector<TransferDescriptor>          descriptors):
        result_(std::move(result)), worker_count_(worker_count), descriptors_(std::move(descriptors)) {}

    void start() {
        std::shared_ptr<MultiRankTransferAsyncContext> self = shared_from_this();
        result_->onDone([self = std::move(self)] { self->evaluate(); });
    }

    void evaluate(ErrorInfo forced_error = ErrorInfo::OkStatus()) {
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
        }

        auto       copy_error = result_->success() ? CacheCopyError::NONE : CacheCopyError::RPC_FAILED;
        const auto responses  = result_->responses();
        if (responses.size() != worker_count_ && error.ok()) {
            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer response count mismatch");
        }
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            if (!result_->rpcSucceeded(rank)) {
                continue;
            }
            const auto& response = responses[rank];
            if (!response.has_mem_response() || response.mem_response().code() != MemoryOperationResponsePB::OK) {
                if (error.ok()) {
                    error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "multi-rank transfer failed, rank=" + std::to_string(rank));
                }
                copy_error = std::max(copy_error, CacheCopyError::COPY_FAILED);
            }
            if (response.has_mem_response()) {
                copy_error = std::max(
                    copy_error, BlockTransferRequestConverter::decodeCopyError(response.mem_response().copy_error()));
            }
        }
        // A rank reports one error for the batch. Only a confirmed mismatch
        // invalidates its cached sources; transport failures retain the legacy policy.
        for (const auto& descriptor : descriptors_) {
            descriptor.markCopyError(copy_error);
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
    std::vector<TransferDescriptor>          descriptors_;
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

std::shared_ptr<AsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task) const {
    const auto deadline_exceeded = []() {
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before multi-rank submission"));
    };
    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, task, group_sets_)) {
        if (task.expired()) {
            return deadline_exceeded();
        }
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", task.descriptors().size());
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "failed to encode transfer batch"));
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
    auto broadcast_result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests,
        broadcast_timeout_ms,
        [](const std::shared_ptr<RpcService::Stub>&    stub,
           const std::shared_ptr<grpc::ClientContext>& context,
           const FunctionRequestPB&                    rpc_request,
           grpc::CompletionQueue*                      completion_queue) {
            return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
        });
    if (broadcast_result == nullptr) {
        RTP_LLM_LOG_WARNING("failed to start broadcast");
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "failed to start transfer broadcast"));
    }
    return MultiRankTransferAsyncContext::create(std::move(broadcast_result), worker_count, task.descriptors());
}

}  // namespace rtp_llm
