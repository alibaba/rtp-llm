#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"

namespace rtp_llm {

// ----------------------------- MemoryAsyncMatchContext ---------------------------------

void MemoryAsyncMatchContext::waitDone() {
    return;
}

bool MemoryAsyncMatchContext::done() const {
    return true;
}

bool MemoryAsyncMatchContext::success() const {
    return true;
}

size_t MemoryAsyncMatchContext::matchedBlockCount() const {
    return matched_block_count_;
}

int MemoryAsyncMatchContext::startReadBlockIndex() const {
    return start_read_block_index_;
}

int MemoryAsyncMatchContext::readBlockNum() const {
    return read_block_num_;
}

std::shared_ptr<void> MemoryAsyncMatchContext::readCopyPlan() const {
    return read_copy_plan_;
}

void MemoryAsyncMatchContext::clearReadCopyPlan() {
    read_copy_plan_.reset();
}

// ----------------------------- MemoryAsyncContext ---------------------------------

bool MemoryAsyncContext::done() const {
    return already_done_.load();
}

MemoryOperationResponsePB::ErrorCode MemoryAsyncContext::resultErrorLocked() const {
    using Response = MemoryOperationResponsePB;
    if (!broadcast_result_) {
        return Response::RPC_FAILED;
    }
    const bool rpc_success = broadcast_result_->success();
    auto       error       = rpc_success ? Response::NONE : Response::RPC_FAILED;
    const auto responses   = broadcast_result_->responses();
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        if (!rpc_success && !broadcast_result_->rpcSucceeded(rank)) {
            continue;
        }
        const auto& response     = responses[rank];
        auto        worker_error = Response::COPY_FAILED;
        if (response.has_mem_response()) {
            const auto& copy = response.mem_response();
            worker_error     = copy.error_code();
            if (!Response::ErrorCode_IsValid(worker_error) || (worker_error == Response::NONE && !copy.success())) {
                worker_error = Response::COPY_FAILED;
            }
        }
        // A confirmed mismatch must remain visible even if another rank's RPC failed.
        if (worker_error == Response::CRC_MISMATCH || error == Response::NONE) {
            error = worker_error;
        }
    }
    return error;
}

bool MemoryAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (already_done_.load() ? copy_error_ : resultErrorLocked()) == MemoryOperationResponsePB::NONE;
}

ErrorInfo MemoryAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!already_done_.load() || copy_error_ == MemoryOperationResponsePB::NONE) {
        return ErrorInfo::OkStatus();
    }
    // Adapt to the existing stream fallback interface; connector callbacks retain the precise error enum.
    return ErrorInfo(ErrorCode::KV_CACHE_REUSE_ERROR, MemoryOperationResponsePB::ErrorCode_Name(copy_error_));
}

void MemoryAsyncContext::waitDone() {
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> result;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return result_ready_ || already_done_.load(); });
        if (already_done_.load()) {
            return;
        }
        if (finalizing_) {
            cv_.wait(lock, [this]() { return already_done_.load(); });
            return;
        }
        finalizing_ = true;
        result      = broadcast_result_;
    }

    if (result) {
        result->waitDone();
    }

    MemoryOperationResponsePB::ErrorCode error;
    DoneCallback                         done_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        copy_error_   = resultErrorLocked();
        error         = copy_error_;
        done_callback = std::move(done_callback_);
    }
    if (done_callback) {
        done_callback(error);
    }
    // Release callback-owned block references before the scheduler can observe completion.
    done_callback = {};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        already_done_.store(true);
        finalizing_ = false;
    }
    cv_.notify_all();
}

void MemoryAsyncContext::setBroadcastResult(
    const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        broadcast_result_ = result;
        result_ready_     = true;
    }
    cv_.notify_all();
}

}  // namespace rtp_llm
