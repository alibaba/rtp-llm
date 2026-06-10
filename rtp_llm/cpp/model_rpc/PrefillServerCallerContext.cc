#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <thread>

namespace rtp_llm {

PrefillServerCallerContext::PrefillServerCallerContext(const std::string& prefill_addr, const std::string& unique_key):
    prefill_addr_(prefill_addr), unique_key_(unique_key), request_begin_time_us_(currentTimeUs()) {
    client_context_   = std::make_shared<grpc::ClientContext>();
    completion_queue_ = std::make_shared<grpc::CompletionQueue>();
}

PrefillServerCallerContext::~PrefillServerCallerContext() {
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        if (!rpc_started_ || !reader_) {
            finished_ = true;
        }
    }
    cancel();
    wait();
    if (!completion_queue_) {
        return;
    }
    completion_queue_->Shutdown();
    // Drain all remaining events per gRPC contract: after Shutdown(), Next() must be
    // called until it returns SHUTDOWN to avoid leaking pending async operations.
    void* drain_tag = nullptr;
    bool  drain_ok  = false;
    while (completion_queue_->Next(&drain_tag, &drain_ok)) {}
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
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        if (!finished_ && !cancel_requested_) {
            cancel_requested_ = true;
            need_cancel       = true;
        }
    }
    if (need_cancel && client_context_) {
        client_context_->TryCancel();
        RTP_LLM_LOG_DEBUG("PrefillServerCallerContext::cancel: cancelled grpc request, prefill_addr: %s",
                          prefill_addr_.c_str());
    }
}

void PrefillServerCallerContext::wait() {
    while (!done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void PrefillServerCallerContext::startPolling() {
    checkDone();
}

void PrefillServerCallerContext::handleReadChunkLocked(const GenerateOutputsPB& response) {
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
        status_ = grpc::Status(grpc::StatusCode::INTERNAL, error_info_.ToString());
        if (!cancel_requested_) {
            cancel_requested_ = true;
        }
        if (client_context_) {
            client_context_->TryCancel();
        }
        if (!finish_started_ && reader_) {
            reader_->Finish(&status_, reinterpret_cast<void*>(2));
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
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    if (finished_) {
        return;
    }

    if (!rpc_started_ || !reader_) {
        finished_ = true;
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

    if (next_status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
        finished_ = true;
        if (!finish_started_ && !cancel_requested_) {
            status_ = grpc::Status(grpc::StatusCode::CANCELLED, "completion queue shutdown");
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(0)) {
        if (!ok) {
            finished_ = true;
            RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: failed to start stream, unique_key: %s",
                                unique_key_.c_str());
            status_ = grpc::Status(grpc::StatusCode::INTERNAL, "failed to start async prefill stream");
            return;
        }
        if (reader_) {
            read_response_.Clear();
            reader_->Read(&read_response_, reinterpret_cast<void*>(1));
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(1)) {
        if (ok) {
            handleReadChunkLocked(read_response_);
            if (!finished_ && !finish_started_ && reader_) {
                read_response_.Clear();
                reader_->Read(&read_response_, reinterpret_cast<void*>(1));
            }
            return;
        }

        if (!finish_started_ && reader_) {
            reader_->Finish(&status_, reinterpret_cast<void*>(2));
            finish_started_ = true;
        }
        return;
    }

    if (got_tag == reinterpret_cast<void*>(2)) {
        finished_ = true;
        if (!ok) {
            status_ = grpc::Status(grpc::StatusCode::INTERNAL, "prefill stream finish event failed");
        }
        if (error_info_.hasError()) {
            return;
        }
        if (cancel_requested_ && status_.ok()) {
            status_ = grpc::Status(grpc::StatusCode::CANCELLED, "prefill request cancelled");
        }
        if (!status_.ok() && !cancel_requested_) {
            ErrorCode resolved_code = ErrorCode::UNKNOWN_ERROR;
            if (!status_.error_details().empty()) {
                ErrorDetailsPB error_details;
                if (error_details.ParseFromString(status_.error_details())) {
                    resolved_code = static_cast<ErrorCode>(error_details.error_code());
                }
            }
            if (resolved_code == ErrorCode::UNKNOWN_ERROR) {
                resolved_code = transGrpcStatusToErrorCode(status_.error_code());
            }
            error_info_ = ErrorInfo(resolved_code, status_.error_message());
        }
        if (!status_.ok()) {
            RTP_LLM_LOG_WARNING(
                "PrefillServerCallerContext::checkDone: prefill rpc failed, unique_key: %s, prefill_addr: %s, grpc_status: %d(%s)",
                unique_key_.c_str(),
                prefill_addr_.c_str(),
                status_.error_code(),
                status_.error_message().c_str());
        }
        return;
    }

    if (!ok) {
        finished_ = true;
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: async next failed, unique_key: %s",
                            unique_key_.c_str());
        status_ = grpc::Status(grpc::StatusCode::INTERNAL, "async get next event from grpc completion queue failed");
        return;
    }

    finished_ = true;
    status_   = grpc::Status(grpc::StatusCode::INTERNAL, "unexpected grpc completion tag");
}

}  // namespace rtp_llm
