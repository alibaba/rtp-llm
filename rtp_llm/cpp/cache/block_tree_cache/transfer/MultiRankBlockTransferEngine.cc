#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskId.h"
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
    create(std::shared_ptr<TransferBroadcastResult> result, size_t worker_count, size_t block_count, std::string task_id) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(std::move(result), worker_count, block_count));
        context->task_id_ = std::move(task_id);
        context->start();
        return context;
    }

    void waitDone() override {
        (void)broadcast_result_->waitForCompletion();
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

    void evaluate() {
        if (!broadcast_result_->done()) {
            return;
        }
        TransferResult result;
        auto           error = broadcast_result_->errorInfo();
        if (!error.ok()) {
            if (StaticConfig::user_ft_core_dump_on_exception) {
                RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK");
            }
            result = {std::move(error), CacheCopyError::RPC_FAILED};
            if (broadcast_result_->onlyTimeoutFailures()) result.recovery_task_ids = {task_id_};
        }

        const auto responses = broadcast_result_->responses();
        if (responses.size() != worker_count_) {
            result.merge(
                {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "multi-rank transfer response count mismatch"),
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
                           TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "missing transfer response"),
                                   CacheCopyError::RPC_FAILED);
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
    std::string                              task_id_;
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

void MultiRankBlockTransferEngine::enqueueFinishedTasks(const std::vector<std::string>& ids) const {
    std::lock_guard<std::mutex> lock(cleanup_->mutex);
    cleanup_->ids.insert(ids.begin(), ids.end());
}

void MultiRankBlockTransferEngine::forgetFinishedTasksWith(const ForgetTasks& forget) const {
    std::unique_lock<std::mutex> polling(cleanup_poll_mutex_, std::try_to_lock);
    if (!polling.owns_lock()) return;
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(cleanup_->mutex);
        constexpr size_t kMaxCleanupBatch = 1024;
        for (const auto& id : cleanup_->ids) {
            ids.push_back(id);
            if (ids.size() == kMaxCleanupBatch) break;
        }
    }
    // No queue/cache/pool lock across RPC. Failure or exception leaves the batch for retry.
    if (ids.empty() || !forget(ids)) return;
    std::lock_guard<std::mutex> lock(cleanup_->mutex);
    for (const auto& id : ids) cleanup_->ids.erase(id);
}

void MultiRankBlockTransferEngine::forgetFinishedTasks() const {
    if (!broadcast_manager_ || broadcast_manager_->workerNum() == 0) return;
    forgetFinishedTasksWith([this](const auto& ids) {
        ForgetTransferTasksRequestPB request;
        for (const auto& id : ids) request.add_task_ids(id);
        auto result = broadcast_manager_->forgetTransferTasks(
            std::vector<ForgetTransferTasksRequestPB>(broadcast_manager_->workerNum(), request));
        return result && result->waitForCompletion().ok();
    });
}

std::vector<TaskCompletion> MultiRankBlockTransferEngine::queryTasks(
    const std::vector<std::string>& ids, int timeout_ms) const {
    std::vector<TaskCompletion> states(ids.size(), TaskCompletion::UNKNOWN);
    if (ids.empty() || !broadcast_manager_ || broadcast_manager_->workerNum() == 0) return states;
    QueryTransferTasksRequestPB request;
    for (const auto& id : ids) request.add_task_ids(id);
    const auto count = broadcast_manager_->workerNum();
    auto result = broadcast_manager_->queryTransferTasks(
        std::vector<QueryTransferTasksRequestPB>(count, request), timeout_ms);
    if (!result || !result->waitForCompletion().ok()) return states;
    const auto responses = result->responses();
    for (size_t i = 0; i < ids.size(); ++i)
        states[i] = BlockTransferRequestConverter::queryCompletion(ids[i], responses, count);
    return states;
}

std::vector<TaskCompletion> MultiRankBlockTransferEngine::queryKVCMOperations(
    const std::vector<std::string>& ids, int timeout_ms) const {
    std::vector<TaskCompletion> states(ids.size(), TaskCompletion::UNKNOWN);
    if (ids.empty() || !broadcast_manager_ || broadcast_manager_->workerNum() == 0) return states;
    QueryKVCMOperationsRequestPB request;
    for (const auto& id : ids) request.add_operation_ids(id);
    const auto count = broadcast_manager_->workerNum();
    auto result = broadcast_manager_->queryKVCMOperations(
        std::vector<QueryKVCMOperationsRequestPB>(count, request), timeout_ms);
    if (!result || !result->waitForCompletion().ok()) return states;
    const auto responses = result->responses();
    if (responses.size() != count) return states;
    using PB = QueryKVCMOperationsResponsePB::OperationState;
    for (size_t index = 0; index < ids.size(); ++index) {
        bool finished = true;
        bool found_all = true;
        for (const auto& response : responses) {
            const QueryKVCMOperationsResponsePB::OperationState* found = nullptr;
            for (const auto& entry : response.operations()) {
                if (entry.operation_id() != ids[index]) continue;
                if (found) { found_all = false; break; }
                found = &entry;
            }
            if (!found || (found->state() != PB::FINISHED && found->state() != PB::RUNNING)) {
                found_all = false;
                break;
            }
            if (found->state() == PB::RUNNING) finished = false;
        }
        states[index] = !found_all ? TaskCompletion::UNKNOWN :
                        finished  ? TaskCompletion::FINISHED : TaskCompletion::PENDING;
    }
    return states;
}

std::shared_ptr<TransferAsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task) const {
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
    request.set_task_id(nextTransferTaskId());
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
    auto context = MultiRankTransferAsyncContext::create(
        std::move(broadcast_result), worker_count, block_count, request.task_id());
    context->onDone([queue = cleanup_, id = request.task_id()](TransferResult result) {
        if (!result.ok()) return;
        std::lock_guard<std::mutex> lock(queue->mutex);
        queue->ids.insert(id);
    });
    return context;
}

}  // namespace rtp_llm
