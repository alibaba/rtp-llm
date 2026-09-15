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

bool MemoryAsyncContext::successLocked() const {
    if (!broadcast_result_ || !broadcast_result_->success()) {
        return false;
    }
    const auto& responses = broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_mem_response() || !response.mem_response().success()) {
            return false;
        }
    }
    return true;
}

bool MemoryAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return successLocked();
}

ErrorInfo MemoryAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reject_reuse_on_failure_ || !broadcast_result_ || !broadcast_result_->success() || successLocked()) {
        return ErrorInfo::OkStatus();
    }
    return ErrorInfo(ErrorCode::KV_CACHE_REUSE_ERROR, "memory cache reuse failed");
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

    bool ok = false;
    std::function<void(bool)> done_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ok            = successLocked();
        done_callback = std::move(done_callback_);
    }
    auto copy_failure = std::move(copy_failure_callback_);
    if (!ok && reject_reuse_on_failure_ && result && result->success() && copy_failure) {
        copy_failure();
    }
    if (done_callback) {
        done_callback(ok);
    }
    // Captures own the CPU backing/copy-plan and connector GPU references.
    // Release them before the scheduler can observe completion and recompute.
    copy_failure  = {};
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
