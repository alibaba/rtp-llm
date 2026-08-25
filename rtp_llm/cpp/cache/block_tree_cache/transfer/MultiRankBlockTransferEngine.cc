#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

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
    static std::shared_ptr<MultiRankTransferAsyncContext>
    create(std::shared_ptr<TransferBroadcastResult> result, size_t worker_count) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(std::move(result), worker_count));
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
    MultiRankTransferAsyncContext(std::shared_ptr<TransferBroadcastResult> result, size_t worker_count):
        result_(std::move(result)), worker_count_(worker_count) {}

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
            } else {
                const auto responses = result_->responses();
                if (responses.size() != worker_count_) {
                    error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer response count mismatch");
                } else {
                    for (size_t rank = 0; rank < responses.size(); ++rank) {
                        if (!responses[rank].has_mem_response()
                            || responses[rank].mem_response().code() != MemoryOperationResponsePB::OK) {
                            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                              "multi-rank transfer failed, rank=" + std::to_string(rank));
                            break;
                        }
                    }
                }
            }
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

std::shared_ptr<AsyncContext>
MultiRankBlockTransferEngine::execute(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    if (descriptors.empty() || timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("invalid batch, item_count=%zu, timeout_ms=%d", descriptors.size(), timeout_ms);
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid multi-rank transfer batch"));
    }

    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, descriptors, group_sets_)) {
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", descriptors.size());
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "failed to encode transfer batch"));
    }
    const size_t              worker_count = broadcast_manager_->workerNum();
    FunctionRequestPB         function_request;
    function_request.mutable_mem_request()->CopyFrom(request);
    std::vector<FunctionRequestPB> requests(worker_count, function_request);
    auto broadcast_result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests,
        timeout_ms,
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
    return MultiRankTransferAsyncContext::create(std::move(broadcast_result), worker_count);
}

}  // namespace rtp_llm
