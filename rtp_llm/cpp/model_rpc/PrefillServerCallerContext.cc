#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/DeferredCompletionQueueDrainer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <chrono>
#include <thread>

namespace rtp_llm {

namespace {

constexpr int64_t kPrefillWaitBudgetMs  = 100;
constexpr int64_t kPrefillDrainBudgetMs = 100;

struct PrefillServerCallerAsyncStateDrainTraits {
    static std::shared_ptr<grpc::CompletionQueue>
    completionQueue(const std::shared_ptr<PrefillServerCallerAsyncState>& async_state) {
        return async_state ? async_state->completion_queue : nullptr;
    }

    static void markDrained(const std::shared_ptr<PrefillServerCallerAsyncState>& async_state) {
        if (async_state) {
            async_state->completion_queue_shutdown_drained_ = true;
        }
    }
};

using PrefillServerCallerCqDrainer =
    DeferredCompletionQueueDrainer<PrefillServerCallerAsyncState, PrefillServerCallerAsyncStateDrainTraits>;

}  // namespace

PrefillServerCallerContext::PrefillServerCallerContext(const std::string& prefill_addr, const std::string& unique_key):
    prefill_addr_(prefill_addr), unique_key_(unique_key), request_begin_time_us_(currentTimeUs()) {
    async_state_                   = std::make_shared<PrefillServerCallerAsyncState>();
    async_state_->client_context   = std::make_shared<grpc::ClientContext>();
    async_state_->completion_queue = std::make_shared<grpc::CompletionQueue>();
}

PrefillServerCallerContext::~PrefillServerCallerContext() {
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        if (!rpc_started_ || !async_state_ || !async_state_->reader) {
            finished_ = true;
        }
    }
    cancel();
    wait();
    shutdownAndDrainCompletionQueue();
}

bool PrefillServerCallerContext::getPrefillReuseLensSnapshot(ReuseLensSnapshot& snapshot) {
    checkDone();
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    if (reuse_lens_valid_) {
        snapshot = reuse_lens_snapshot_;
        return true;
    }
    if (!response_received_ || !response_.has_flatten_output() || response_.flatten_output().aux_info_size() == 0) {
        return false;
    }

    const auto& aux_info = response_.flatten_output().aux_info(0);
    snapshot.total       = aux_info.prefill_total_reuse_len();
    snapshot.local       = aux_info.prefill_local_reuse_len();
    snapshot.remote      = aux_info.prefill_remote_reuse_len();
    snapshot.memory      = aux_info.prefill_memory_reuse_len();

    if (snapshot.total > 0 || snapshot.local > 0 || snapshot.remote > 0 || snapshot.memory > 0) {
        return true;
    }

    snapshot.total  = aux_info.total_reuse_len();
    snapshot.local  = aux_info.local_reuse_len();
    snapshot.remote = aux_info.remote_reuse_len();
    snapshot.memory = aux_info.memory_reuse_len();
    return snapshot.total > 0 || snapshot.local > 0 || snapshot.remote > 0 || snapshot.memory > 0;
}

void PrefillServerCallerContext::setPrefillReuseLensSnapshotForTest(const ReuseLensSnapshot& snapshot) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    reuse_lens_snapshot_ = snapshot;
    reuse_lens_valid_    = true;
}

bool PrefillServerCallerContext::updateReuseLensSnapshotLocked(const GenerateOutputsPB& response) {
    if (!response.has_flatten_output() || response.flatten_output().aux_info_size() == 0) {
        return false;
    }

    ReuseLensSnapshot snapshot;
    const auto&       aux_info = response.flatten_output().aux_info(0);
    snapshot.total             = aux_info.prefill_total_reuse_len();
    snapshot.local             = aux_info.prefill_local_reuse_len();
    snapshot.remote            = aux_info.prefill_remote_reuse_len();
    snapshot.memory            = aux_info.prefill_memory_reuse_len();

    if (snapshot.total <= 0 && snapshot.local <= 0 && snapshot.remote <= 0 && snapshot.memory <= 0) {
        snapshot.total  = aux_info.total_reuse_len();
        snapshot.local  = aux_info.local_reuse_len();
        snapshot.remote = aux_info.remote_reuse_len();
        snapshot.memory = aux_info.memory_reuse_len();
    }

    if (snapshot.total <= 0 && snapshot.local <= 0 && snapshot.remote <= 0 && snapshot.memory <= 0) {
        return false;
    }

    reuse_lens_snapshot_ = snapshot;
    reuse_lens_valid_    = true;
    return true;
}

void PrefillServerCallerContext::cancel() {
    bool need_cancel = false;
    auto async_state = async_state_;
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        if (!finished_ && !cancel_requested_) {
            cancel_requested_ = true;
            need_cancel       = true;
        }
    }
    if (need_cancel && async_state && async_state->client_context) {
        async_state->client_context->TryCancel();
        RTP_LLM_LOG_DEBUG("PrefillServerCallerContext::cancel: cancelled grpc request, prefill_addr: %s",
                          prefill_addr_.c_str());
    }
}

void PrefillServerCallerContext::wait() {
    if (!waitWithTimeoutMs(kPrefillWaitBudgetMs)) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] PrefillServerCallerContext::wait timed out, unique_key: %s, prefill_addr: %s, "
                            "budget_ms: %ld",
                            unique_key_.c_str(),
                            prefill_addr_.c_str(),
                            kPrefillWaitBudgetMs);
    }
}

void PrefillServerCallerContext::startPolling() {
    checkDone();
}

void PrefillServerCallerContext::handleReadChunkLocked(const GenerateOutputsPB& response) {
    auto async_state = async_state_;
    if (!first_response_received_) {
        first_response_.CopyFrom(response);
        first_response_received_ = true;
    }
    response_.CopyFrom(response);
    response_received_ = true;
    updateReuseLensSnapshotLocked(response);
    if (response.has_error_info() && response.error_info().error_code() != ErrorCodePB::NONE_ERROR) {
        error_info_ =
            ErrorInfo(transRPCErrorCode(response.error_info().error_code()), response.error_info().error_message());
        if (async_state) {
            async_state->status = grpc::Status(grpc::StatusCode::INTERNAL, error_info_.ToString());
        }
        if (!cancel_requested_) {
            cancel_requested_ = true;
        }
        if (async_state && async_state->client_context) {
            async_state->client_context->TryCancel();
        }
        if (!finish_started_ && async_state && async_state->reader) {
            async_state->reader->Finish(&async_state->status, reinterpret_cast<void*>(2));
            finish_started_ = true;
        }
        RTP_LLM_LOG_WARNING(
            "PrefillServerCallerContext::checkDone: prefill response error, unique_key: %s, error_code: %s, error_message: %s",
            unique_key_.c_str(),
            ErrorCodeToString(transRPCErrorCode(response.error_info().error_code())).c_str(),
            response.error_info().error_message().c_str());
    }
}

void PrefillServerCallerContext::checkDone() {
    std::shared_ptr<PrefillServerCallerAsyncState> async_state;
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        async_state = async_state_;
        if (finished_) {
            return;
        }

        if (!rpc_started_ || !async_state || !async_state->reader) {
            finished_ = true;
            return;
        }

        if (!async_state->completion_queue) {
            RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: completion_queue is null");
            finished_  = true;
            error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "completion_queue is null");
            return;
        }
    }

    std::unique_lock<std::mutex> poll_lock(completion_queue_poll_mutex_, std::try_to_lock);
    if (!poll_lock.owns_lock()) {
        return;
    }

    void* got_tag = nullptr;
    bool  ok      = false;

    auto once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1);
    auto next_status   = async_state->completion_queue->AsyncNext(&got_tag, &ok, once_deadline);
    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        // not finish yet
        return;
    }

    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    if (finished_) {
        return;
    }

    if (next_status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
        finished_ = true;
        if (!finish_started_ && !cancel_requested_) {
            async_state->status = grpc::Status(grpc::StatusCode::CANCELLED, "completion queue shutdown");
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(0)) {
        if (!ok) {
            finished_ = true;
            RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: failed to start stream, unique_key: %s",
                                unique_key_.c_str());
            async_state->status = grpc::Status(grpc::StatusCode::INTERNAL, "failed to start async prefill stream");
            return;
        }
        if (async_state->reader) {
            async_state->read_response.Clear();
            async_state->reader->Read(&async_state->read_response, reinterpret_cast<void*>(1));
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(1)) {
        if (ok) {
            handleReadChunkLocked(async_state->read_response);
            if (!finished_ && !finish_started_ && async_state->reader) {
                async_state->read_response.Clear();
                async_state->reader->Read(&async_state->read_response, reinterpret_cast<void*>(1));
            }
            return;
        }

        if (!finish_started_ && async_state->reader) {
            async_state->reader->Finish(&async_state->status, reinterpret_cast<void*>(2));
            finish_started_ = true;
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(2)) {
        finished_ = true;
        if (!ok) {
            async_state->status = grpc::Status(grpc::StatusCode::INTERNAL, "prefill stream finish event failed");
        }
        if (error_info_.hasError()) {
            return;
        }
        if (cancel_requested_ && async_state->status.ok()) {
            async_state->status = grpc::Status(grpc::StatusCode::CANCELLED, "prefill request cancelled");
        }
        if (!async_state->status.ok() && !cancel_requested_) {
            ErrorCode resolved_code = ErrorCode::UNKNOWN_ERROR;
            if (!async_state->status.error_details().empty()) {
                ErrorDetailsPB error_details;
                if (error_details.ParseFromString(async_state->status.error_details())) {
                    resolved_code = static_cast<ErrorCode>(error_details.error_code());
                }
            }
            if (resolved_code == ErrorCode::UNKNOWN_ERROR) {
                resolved_code = transGrpcStatusToErrorCode(async_state->status.error_code());
            }
            error_info_ = ErrorInfo(resolved_code, async_state->status.error_message());
        }
        if (!async_state->status.ok()) {
            RTP_LLM_LOG_WARNING(
                "PrefillServerCallerContext::checkDone: prefill rpc failed, unique_key: %s, prefill_addr: %s, grpc_status: %d(%s)",
                unique_key_.c_str(),
                prefill_addr_.c_str(),
                async_state->status.error_code(),
                async_state->status.error_message().c_str());
        }
        return;
    }

    if (!ok) {
        finished_ = true;
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: async next failed, unique_key: %s",
                            unique_key_.c_str());
        async_state->status =
            grpc::Status(grpc::StatusCode::INTERNAL, "async get next event from grpc completion queue failed");
        return;
    }

    finished_ = true;
    async_state->status = grpc::Status(grpc::StatusCode::INTERNAL, "unexpected grpc completion tag");
}

bool PrefillServerCallerContext::waitWithTimeoutMs(int64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return done();
}

void PrefillServerCallerContext::shutdownAndDrainCompletionQueue() {
    std::unique_lock<std::mutex> poll_lock(completion_queue_poll_mutex_);
    auto async_state = async_state_;
    if (!async_state || !async_state->completion_queue || async_state->completion_queue_shutdown_drained_) {
        return;
    }
    async_state->completion_queue->Shutdown();

    void*      drain_tag = nullptr;
    bool       drain_ok  = false;
    const auto deadline  = std::chrono::system_clock::now() + std::chrono::milliseconds(kPrefillDrainBudgetMs);
    while (true) {
        auto next_status = async_state->completion_queue->AsyncNext(&drain_tag, &drain_ok, deadline);
        if (next_status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
            async_state->completion_queue_shutdown_drained_ = true;
            return;
        }
        if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
            RTP_LLM_LOG_WARNING("[PD-DIAG] PrefillServerCallerContext deferred CQ drain, unique_key: %s, "
                                "prefill_addr: %s, budget_ms: %ld",
                                unique_key_.c_str(),
                                prefill_addr_.c_str(),
                                kPrefillDrainBudgetMs);
            async_state->completion_queue_shutdown_drained_ = true;
            PrefillServerCallerCqDrainer::instance().enqueue(async_state);
            async_state_.reset();
            return;
        }
    }
}

}  // namespace rtp_llm
