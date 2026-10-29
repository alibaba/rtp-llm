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
    create(std::shared_ptr<TransferBroadcastResult>       result,
           size_t                                         worker_count,
           size_t                                         block_count,
           SharedTransferStageTimer                       wait_timer,
           std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
           int64_t                                        request_id,
           uint64_t                                       transfer_id,
           CacheStageContext                              identity) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(
            new MultiRankTransferAsyncContext(std::move(result), worker_count, block_count));
        context->wait_timer_  = std::move(wait_timer);
        context->reporter_    = std::move(reporter);
        context->request_id_  = request_id;
        context->transfer_id_ = transfer_id;
        context->identity_    = identity;
        context->start();
        return context;
    }

    void waitDone() override {
        try {
            broadcast_result_->waitDone();
        } catch (const std::exception& error) {
            evaluate(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, error.what()));
        } catch (...) {
            evaluate(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "unknown multi-rank transfer failure"));
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
                    RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK: %s",
                                 broadcast_result_->failureSummary().c_str());
                }
                result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                    "multi-rank transfer RPC failed: " + broadcast_result_->failureSummary()),
                          CacheCopyError::RPC_FAILED};
            }
        }

        const auto responses = broadcast_result_->responses();
        if (responses.size() != worker_count_) {
            result.merge({ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "multi-rank transfer response count mismatch"),
                          CacheCopyError::RPC_FAILED});
        }
        result.expandBlockErrors(block_count_);
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            if (!broadcast_result_->rpcSucceeded(rank)) {
                continue;
            }
            const auto& response = responses[rank];
            TransferResult rank_result{
                ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "invalid transfer response"),
                CacheCopyError::RPC_FAILED};
            if (response.has_mem_response()) {
                const auto& memory_response = response.mem_response();
                bool valid = MemoryOperationResponsePB::Code_IsValid(memory_response.code())
                             && static_cast<size_t>(memory_response.block_errors_size()) == block_count_;
                for (int index = 0; valid && index < memory_response.block_errors_size(); ++index) {
                    valid = MemoryOperationResponsePB::Code_IsValid(memory_response.block_errors(index));
                }
                if (valid) {
                    try {
                        rank_result = BlockTransferRequestConverter::decodeTransferResult(memory_response, block_count_);
                    } catch (const std::exception& error) {
                        rank_result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                                 "transfer response decode failed: " + std::string(error.what())),
                                       CacheCopyError::RPC_FAILED};
                    } catch (...) {
                        rank_result = {
                            ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "unknown transfer response decode failure"),
                            CacheCopyError::RPC_FAILED};
                    }
                }
            }
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
        // Copy before claiming completion: allocation failure must leave the
        // context retryable. Once claimed, publication cannot allocate.
        if (auto probe =
                block_tree_cache_detail::before_transfer_publication_copy_for_test.load(std::memory_order_acquire))
            probe();
        auto published_result = result;
        static_assert(std::is_nothrow_move_assignable_v<TransferResult>);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (evaluated_ || finishing_) {
                return;
            }
            finishing_ = true;
        }
        if (wait_timer_) {
            wait_timer_.finish(result.ok());
            try {
                const auto times = broadcast_result_->completionTimesUs();
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
                                          identity_.path);
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
        for (auto& callback : callbacks) {
            callback(result);
        }
    }

    std::shared_ptr<TransferBroadcastResult>       broadcast_result_;
    size_t                                         worker_count_{0};
    size_t                                         block_count_{0};
    mutable std::mutex                             mutex_;
    std::condition_variable                        cv_;
    TransferResult                                 result_;
    bool                                           evaluated_{false};
    bool                                           finishing_{false};
    std::vector<DoneCallback>                      callbacks_;
    SharedTransferStageTimer                       wait_timer_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> reporter_;
    int64_t                                        request_id_{-1};
    uint64_t                                       transfer_id_{0};
    CacheStageContext                              identity_;
};

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(
    std::vector<GroupSetPtr>                       group_sets,
    std::shared_ptr<BroadcastManager>              broadcast_manager,
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    group_sets_(std::move(group_sets)),
    broadcast_manager_(std::move(broadcast_manager)),
    metrics_reporter_(std::move(metrics_reporter)) {}

std::shared_ptr<TransferAsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task) const {
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
                                                                                  identity) :
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
    auto broadcast_result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests,
        broadcast_timeout_ms,
        [](const std::shared_ptr<RpcService::Stub>&    stub,
           const std::shared_ptr<grpc::ClientContext>& context,
           const FunctionRequestPB&                    rpc_request,
           grpc::CompletionQueue*                      completion_queue) {
            return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
        },
        task.requestId(),
        MemoryOperationRequestPB_CopyDirection_Name(request.copy_direction()));
    if (broadcast_result == nullptr) {
        RTP_LLM_LOG_WARNING("failed to start broadcast");
        return std::make_shared<CompletedTransferAsyncContext>(
            TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "failed to start transfer broadcast"),
                           CacheCopyError::RPC_FAILED),
            block_count);
    }
    return MultiRankTransferAsyncContext::create(std::move(broadcast_result),
                                                 worker_count,
                                                 block_count,
                                                 std::move(wait_timer),
                                                 metrics_reporter_,
                                                 task.requestId().value_or(-1),
                                                 task.traceId(),
                                                 identity);
}

}  // namespace rtp_llm
