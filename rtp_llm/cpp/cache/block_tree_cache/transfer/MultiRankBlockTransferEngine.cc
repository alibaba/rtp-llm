#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using TransferBroadcastResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

class MultiRankTransferAsyncContext final: public AsyncContext {
public:
    explicit MultiRankTransferAsyncContext(std::shared_ptr<TransferBroadcastResult> result): result_(std::move(result)) {}

    void waitDone() override {
        std::call_once(wait_once_, [this] {
            result_->waitDone();
            if (!result_->success()) {
                RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK; worker copy state is unknown");
            }

            const auto responses = result_->responses();
            bool transfer_success = true;
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                const auto& response = responses[rank];
                if (!response.has_mem_response()) {
                    RTP_LLM_FAIL("multi-rank transfer aborted, response has no mem_response, rank=%zu", rank);
                }
                switch (response.mem_response().code()) {
                    case MemoryOperationResponsePB::OK:
                        break;
                    case MemoryOperationResponsePB::FAILED:
                        RTP_LLM_LOG_WARNING("worker transfer failed, rank=%zu", rank);
                        transfer_success = false;
                        break;
                    default:
                        RTP_LLM_FAIL("multi-rank transfer aborted, unexpected mem_response code, rank=%zu code=%d",
                                     rank,
                                     static_cast<int>(response.mem_response().code()));
                }
            }
            complete(transfer_success ? ErrorInfo::OkStatus() :
                                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "worker transfer failed"));
        });
    }

    bool done() const override {
        return done_.load(std::memory_order_acquire);
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_.load(std::memory_order_relaxed) && error_info_.ok();
    }

    ErrorInfo errorInfo() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_info_;
    }

private:
    void complete(ErrorInfo error_info) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_info_ = std::move(error_info);
        }
        done_.store(true, std::memory_order_release);
    }

    std::shared_ptr<TransferBroadcastResult> result_;
    std::once_flag                           wait_once_;
    std::atomic<bool>                        done_{false};
    mutable std::mutex                       mutex_;
    ErrorInfo                                error_info_;
};

std::shared_ptr<AsyncContext> completedError(ErrorCode code, const std::string& message) {
    return std::make_shared<CompletedAsyncContext>(ErrorInfo(code, message));
}

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>          group_sets,
                                                           std::shared_ptr<BroadcastManager> broadcast_manager):
    group_sets_(std::move(group_sets)), broadcast_manager_(std::move(broadcast_manager)) {}

std::shared_ptr<AsyncContext> MultiRankBlockTransferEngine::execute(const std::vector<TransferDescriptor>& descriptors,
                                                                    int timeout_ms) const {
    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, descriptors, group_sets_)) {
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", descriptors.size());
        return completedError(ErrorCode::INVALID_PARAMS, "failed to encode transfer batch");
    }

    const size_t worker_count = broadcast_manager_->workerNum();

    FunctionRequestPB         function_request;
    MemoryOperationRequestPB* memory_request = function_request.mutable_mem_request();
    memory_request->CopyFrom(request);
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
        return completedError(ErrorCode::EXECUTION_EXCEPTION, "failed to start broadcast");
    }
    return std::make_shared<MultiRankTransferAsyncContext>(std::move(broadcast_result));
}

}  // namespace rtp_llm
