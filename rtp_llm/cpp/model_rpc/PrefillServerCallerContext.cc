#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

PrefillServerCallerContext::PrefillServerCallerContext(const std::string& prefill_addr, const std::string& unique_key):
    prefill_addr_(prefill_addr), unique_key_(unique_key), request_begin_time_us_(currentTimeUs()) {
    client_context_   = std::make_shared<grpc::ClientContext>();
    completion_queue_ = std::make_shared<grpc::CompletionQueue>();
}

PrefillServerCallerContext::~PrefillServerCallerContext() {
    if (client_context_) {
        client_context_->TryCancel();
    }
    completion_queue_->Shutdown();
}

void PrefillServerCallerContext::cancel() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    if (finished_) {
        return;
    }
    if (client_context_) {
        client_context_->TryCancel();
        RTP_LLM_LOG_DEBUG("PrefillServerCallerContext::cancel: cancelled grpc request, prefill_addr: %s",
                          prefill_addr_.c_str());
    }
    finished_ = true;
}

void PrefillServerCallerContext::checkDone() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    if (finished_) {
        return;
    }

    if (!completion_queue_) {
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: completion_queue is null");
        finished_ = true;
        status_   = grpc::Status(grpc::StatusCode::INTERNAL, "completion_queue is null");
        return;
    }

    void* got_tag = nullptr;
    bool  ok      = false;

    // Calculate timeout (non-blocking, use short timeout)
    auto once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1);

    // Wait for async operation to complete
    auto next_status = completion_queue_->AsyncNext(&got_tag, &ok, once_deadline);
    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        // not finish yet
        return;
    }

    if (!ok) {
        finished_ = true;
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: async next failed, unique_key: %s",
                            unique_key_.c_str());
        status_ = grpc::Status(grpc::StatusCode::INTERNAL, "async get next event from grpc completion queue failed");
        return;
    }

    // Handle Read event
    if (got_tag == reinterpret_cast<void*>(1)) {
        // Read complete, received first response
        response_received_ = true;
        // Check for business errors in response body (error_info)
        if (response_.has_error_info() && response_.error_info().error_code() != ErrorCodePB::NONE_ERROR) {
            RTP_LLM_LOG_WARNING(
                "PrefillServerCallerContext::checkDone: prefill response error, unique_key: %s, error_code: %s, error_message: %s",
                unique_key_.c_str(),
                ErrorCodeToString(transRPCErrorCode(response_.error_info().error_code())).c_str(),
                response_.error_info().error_message().c_str());
        }
        // Start Finish to get final status
        if (!finish_started_ && reader_) {
            reader_->Finish(&status_, reinterpret_cast<void*>(2));
            finish_started_ = true;
        }
    } else if (got_tag == reinterpret_cast<void*>(2)) {
        // Finish complete
        finished_ = true;
        if (!status_.ok()) {
            RTP_LLM_LOG_WARNING(
                "PrefillServerCallerContext::checkDone: prefill rpc failed, unique_key: %s, prefill_addr: %s, grpc_status: %d(%s)",
                unique_key_.c_str(),
                prefill_addr_.c_str(),
                status_.error_code(),
                status_.error_message().c_str());
        }
    }
}

}  // namespace rtp_llm
