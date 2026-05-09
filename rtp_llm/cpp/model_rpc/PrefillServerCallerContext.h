#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>

namespace rtp_llm {

class PrefillServerCaller;

class PrefillServerCallerContext {
public:
    ~PrefillServerCallerContext();

    // Constructor (only friends can create instances)
    PrefillServerCallerContext(const std::string& prefill_addr, const std::string& unique_key);

    // Non-blocking check if Prefill is complete
    void checkDone();

    // Start CQ polling in a managed background thread.
    void startPolling();

    // Check if complete
    bool done() const {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return finished_;
    }

    // Check if successful
    bool success() const {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return finished_ && status_.ok() && response_received_;
    }

    bool failed() const {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return finished_
               && (error_info_.hasError() || (!status_.ok() && status_.error_code() != grpc::StatusCode::CANCELLED));
    }

    // Get response (only valid after done() returns true)
    const GenerateOutputsPB& response() const {
        return response_;
    }

    ErrorInfo errorInfo() const {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return error_info_;
    }

    // Cancel the ongoing RPC call
    void cancel();

    // Wait for the managed polling thread to exit.
    void wait();

private:
    friend class PrefillServerCaller;
    void handleReadChunkLocked(const GenerateOutputsPB& response);
    void joinPollingThread();

    // Metadata
    std::string prefill_addr_;
    std::string unique_key_;

    // gRPC infrastructure
    std::shared_ptr<grpc::ClientContext>                        client_context_;
    std::shared_ptr<grpc::CompletionQueue>                      completion_queue_;
    std::shared_ptr<RpcService::Stub>                           stub_;
    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader_;

    // Request/Response
    GenerateInputPB   request_;
    GenerateOutputsPB response_;
    GenerateOutputsPB read_response_;
    grpc::Status      status_;
    ErrorInfo         error_info_;

    // State
    bool finished_          = false;
    bool response_received_ = false;
    bool finish_started_    = false;
    bool cancel_requested_  = false;

    mutable std::shared_mutex state_mutex_;
    int64_t                   request_begin_time_us_;
    std::thread               polling_thread_;
};

}  // namespace rtp_llm
