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
    public TransferAsyncContext,
    public std::enable_shared_from_this<MultiRankTransferAsyncContext> {
public:
    static std::shared_ptr<MultiRankTransferAsyncContext>
    create(std::shared_ptr<TransferBroadcastResult> result, size_t worker_count, size_t block_count) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(std::move(result), worker_count, block_count));
        context->start();
        return context;
    }

    void waitDone() override {
        try {
            broadcast_result_->waitDone();
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

private:
    MultiRankTransferAsyncContext(std::shared_ptr<TransferBroadcastResult> result,
                                  size_t                                   worker_count,
                                  size_t                                   block_count):
        broadcast_result_(std::move(result)), worker_count_(worker_count), block_count_(block_count) {}

    void start() {
        std::shared_ptr<MultiRankTransferAsyncContext> self = shared_from_this();
        broadcast_result_->onDone([self = std::move(self)] { self->evaluate(); });
    }

    void evaluate(ErrorInfo forced_error = ErrorInfo::OkStatus()) {
        TransferResult result =
            forced_error.ok() ? TransferResult{} : TransferResult(std::move(forced_error), CacheCopyError::RPC_FAILED);
        if (result.ok()) {
            if (!broadcast_result_->done()) {
                return;
            }
            if (!broadcast_result_->success()) {
                if (StaticConfig::user_ft_core_dump_on_exception) {
                    RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK");
                }
                result = {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer RPC failed"),
                          CacheCopyError::RPC_FAILED};
            }
        }

        const auto responses = broadcast_result_->responses();
        if (responses.size() != worker_count_) {
            result.merge({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "multi-rank transfer response count mismatch"),
                          CacheCopyError::RPC_FAILED});
        }
        result.expandBlockErrors(block_count_);
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            if (!broadcast_result_->rpcSucceeded(rank)) {
                continue;
            }
            const auto& response = responses[rank];
            auto        rank_result =
                response.has_mem_response() ?
                           BlockTransferRequestConverter::decodeTransferResult(response.mem_response(), block_count_) :
                           TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "missing transfer response"));
            for (size_t index = 0; index < block_count_; ++index) {
                result.block_errors[index] = std::max(result.block_errors[index], rank_result.blockError(index));
            }
            if (!rank_result.ok()) {
                rank_result.status =
                    ErrorInfo(rank_result.status.code(),
                              "rank=" + std::to_string(rank) + " {" + rank_result.status.ToString() + "}");
                result.merge(std::move(rank_result));
            }
        }
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_) {
                return;
            }
            result_    = result;
            evaluated_ = true;
            callbacks.swap(callbacks_);
        }
        cv_.notify_all();
        for (auto& callback : callbacks) {
            callback(result);
        }
    }

    std::shared_ptr<TransferBroadcastResult> broadcast_result_;
    size_t                                   worker_count_{0};
    size_t                                   block_count_{0};
    mutable std::mutex                       mutex_;
    std::condition_variable                  cv_;
    TransferResult                           result_;
    bool                                     evaluated_{false};
    std::vector<DoneCallback>                callbacks_;
};

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>          group_sets,
                                                           std::shared_ptr<BroadcastManager> broadcast_manager):
    group_sets_(std::move(group_sets)), broadcast_manager_(std::move(broadcast_manager)) {}

std::shared_ptr<TransferAsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task) const {
    const size_t block_count       = task.descriptors().size();
    const auto   deadline_exceeded = [block_count]() {
        return std::make_shared<CompletedTransferAsyncContext>(
            ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before multi-rank submission"),
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
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "failed to start transfer broadcast"),
                           CacheCopyError::RPC_FAILED),
            block_count);
    }
    return MultiRankTransferAsyncContext::create(std::move(broadcast_result), worker_count, block_count);
}

}  // namespace rtp_llm
