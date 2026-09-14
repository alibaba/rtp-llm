#include "rtp_llm/cpp/model_rpc/PrefillBatchCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <chrono>

namespace rtp_llm {

PrefillBatchCallerContext::~PrefillBatchCallerContext() {
    cancel();
    queue_.Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (queue_.Next(&tag, &ok)) {}
}

void PrefillBatchCallerContext::cancel() {
    if (started_ && !finished_) {
        client_context_.TryCancel();
    }
}

bool PrefillBatchCallerContext::done() {
    if (finished_ || !started_) {
        return finished_;
    }
    void* tag = nullptr;
    bool  ok  = false;
    if (queue_.AsyncNext(&tag, &ok, std::chrono::system_clock::now()) != grpc::CompletionQueue::GOT_EVENT) {
        return false;
    }
    finished_ = true;
    if (!ok && status_.ok()) {
        status_ = grpc::Status(grpc::StatusCode::INTERNAL, "prefill batch Finish failed");
    }
    if (status_.ok()) {
        for (int i = 0; i < response_.results_size(); ++i) {
            const auto& result = response_.results(i);
            const auto& error  = result.has_error_info() ? result.error_info() : result.final_output().error_info();
            if (error.error_code() != ErrorCodePB::NONE_ERROR || !error.error_message().empty()) {
                status_ = grpcStatusFromErrorInfo(ErrorInfo(
                    error.error_code() == ErrorCodePB::NONE_ERROR ? ErrorCode::UNKNOWN_ERROR :
                                                                    transRPCErrorCode(error.error_code()),
                    "Prefill batch item=" + std::to_string(i) + " request="
                        + (i < request_.inputs_size() ? std::to_string(request_.inputs(i).request_id()) : "unmatched")
                        + ": " + error.error_message()));
                break;
            }
        }
    }
    if (status_.ok() && response_.results_size() != request_.inputs_size()) {
        status_ = grpc::Status(grpc::StatusCode::INTERNAL, "prefill batch result count mismatch");
    }
    first_error_.record(errorInfoFromGrpcStatus(status_, "Prefill BatchGenerateCall peer=" + address_));
    return true;
}

}  // namespace rtp_llm
