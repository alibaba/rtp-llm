#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace rtp_llm {

class PrefillServerCaller;

struct PrefillServerCallerAsyncState {
    std::shared_ptr<grpc::ClientContext>                        client_context;
    std::shared_ptr<grpc::CompletionQueue>                      completion_queue;
    std::shared_ptr<RpcService::Stub>                           stub;
    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader;
    GenerateInputPB                                             request;
    GenerateOutputsPB                                           read_response;
    grpc::Status                                                status;
    bool                                                        completion_queue_shutdown_drained_{false};
};

class PrefillServerCallerContext {
public:
    struct ReuseLensSnapshot {
        int32_t total  = 0;
        int32_t local  = 0;
        int32_t remote = 0;
        int32_t memory = 0;
    };

    ~PrefillServerCallerContext();

    // Constructor (only friends can create instances)
    PrefillServerCallerContext(const std::string& prefill_addr, const std::string& unique_key);

    // Non-blocking check if Prefill is complete
    void checkDone();

    // Kick off an initial CQ poll without creating a per-request background thread.
    void startPolling();

    // Check if complete
    bool done() {
        checkDone();
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return finished_;
    }

    // Check if successful
    bool success() {
        checkDone();
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return finished_ && async_state_ && async_state_->status.ok() && response_received_;
    }

    bool failed() {
        checkDone();
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return error_info_.hasError()
               || (finished_ && async_state_ && !async_state_->status.ok()
                   && async_state_->status.error_code() != grpc::StatusCode::CANCELLED);
    }

    // Get response (only valid after done() returns true)
    const GenerateOutputsPB& response() const {
        return response_;
    }

    ErrorInfo errorInfo() {
        checkDone();
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return error_info_;
    }

    bool getPrefillReuseLensSnapshot(ReuseLensSnapshot& snapshot);

    void setPrefillReuseLensSnapshotForTest(const ReuseLensSnapshot& snapshot);

    bool hasFirstResponse() {
        checkDone();
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return first_response_received_ && !first_response_consumed_;
    }

    bool takeFirstResponse(GenerateOutputsPB& output) {
        checkDone();
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        if (!first_response_received_ || first_response_consumed_ || error_info_.hasError()) {
            return false;
        }
        output.Swap(&first_response_);
        first_response_consumed_ = true;
        return true;
    }

    // Cancel the ongoing RPC call
    void cancel();

    // Wait until the async RPC reaches a terminal state.
    void wait();

private:
    friend class PrefillServerCaller;
    bool updateReuseLensSnapshotLocked(const GenerateOutputsPB& response);
    void handleReadChunkLocked(const GenerateOutputsPB& response);
    void setErrorInfoFromStatusLocked(const grpc::Status& status);
    bool waitWithTimeoutMs(int64_t timeout_ms);
    void shutdownAndDrainCompletionQueue();

    // Metadata
    std::string prefill_addr_;
    std::string unique_key_;

    // gRPC infrastructure
    std::shared_ptr<PrefillServerCallerAsyncState> async_state_;

    // Request/Response
    GenerateOutputsPB first_response_;
    GenerateOutputsPB response_;
    ErrorInfo         error_info_;

    // State
    bool              finished_          = false;
    bool              response_received_ = false;
    bool              finish_started_    = false;
    bool              rpc_started_       = false;
    bool              cancel_requested_      = false;
    bool              first_response_received_ = false;
    bool              first_response_consumed_ = false;
    bool              reuse_lens_valid_      = false;
    ReuseLensSnapshot reuse_lens_snapshot_;

    mutable std::shared_mutex state_mutex_;
    int64_t                   request_begin_time_us_;
};

}  // namespace rtp_llm
