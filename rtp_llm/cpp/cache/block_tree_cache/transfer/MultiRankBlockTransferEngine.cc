#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <exception>
#include <mutex>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using TransferBroadcastResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

class MultiRankTransferAsyncContext final: public AsyncContext {
public:
    MultiRankTransferAsyncContext(std::shared_ptr<TransferBroadcastResult> result, size_t worker_count):
        result_(std::move(result)), worker_count_(worker_count) {}

    void waitDone() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (evaluated_) {
            return;
        }
        for (size_t attempt = 0; attempt <= worker_count_ && !result_->done(); ++attempt) {
            try {
                result_->waitDone();
            } catch (const std::exception& error) {
                if (error_.ok()) {
                    error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what());
                }
            } catch (...) {
                if (error_.ok()) {
                    error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown multi-rank transfer failure");
                }
            }
        }
        if (!result_->done()) {
            error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer did not reach terminal state");
            evaluated_ = true;
            return;
        }
        evaluateTerminalLocked();
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!evaluated_ && result_->done()) {
            evaluateTerminalLocked();
        }
        return evaluated_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!evaluated_ && result_->done()) {
            evaluateTerminalLocked();
        }
        return evaluated_ && error_.ok();
    }

    ErrorInfo errorInfo() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!evaluated_ && result_->done()) {
            evaluateTerminalLocked();
        }
        return error_;
    }

private:
    void evaluateTerminalLocked() const {
        if (evaluated_) {
            return;
        }
        if (!result_->done()) {
            return;
        }
        if (!error_.ok()) {
            evaluated_ = true;
            return;
        }
        if (!result_->success()) {
            if (StaticConfig::user_ft_core_dump_on_exception) {
                RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK");
            }
            error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer RPC failed");
            evaluated_ = true;
            return;
        }
        const auto responses = result_->responses();
        if (responses.size() != worker_count_) {
            error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer response count mismatch");
            evaluated_ = true;
            return;
        }
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            if (!responses[rank].has_mem_response()
                || responses[rank].mem_response().code() != MemoryOperationResponsePB::OK) {
                error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                   "multi-rank transfer failed, rank=" + std::to_string(rank));
                break;
            }
        }
        evaluated_ = true;
    }
    std::shared_ptr<TransferBroadcastResult> result_;
    size_t                                   worker_count_{0};
    mutable std::mutex                       mutex_;
    mutable ErrorInfo                        error_{ErrorInfo::OkStatus()};
    mutable bool                             evaluated_{false};
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
    return std::make_shared<MultiRankTransferAsyncContext>(std::move(broadcast_result), worker_count);
}

}  // namespace rtp_llm
