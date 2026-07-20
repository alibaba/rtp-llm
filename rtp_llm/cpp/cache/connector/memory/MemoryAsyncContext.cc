#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"

#include <exception>

#include "rtp_llm/cpp/utils/Logger.h"

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
    if (failed_) {
        return false;
    }
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

    auto record_failure = [this](const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        if (!reason.empty()) {
            failure_reason_ = reason;
        }
    };

    auto finalize = [this]() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            already_done_.store(true);
            finalizing_ = false;
        }
        cv_.notify_all();
    };

    if (result) {
        try {
            result->waitDone();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("MemoryAsyncContext::waitDone broadcast wait failed: %s", e.what());
            record_failure(e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("MemoryAsyncContext::waitDone broadcast wait failed with unknown exception");
            record_failure("unknown broadcast wait exception");
        }
    }

    bool                      ok = false;
    std::function<void(bool)> done_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ok            = successLocked();
        done_callback = std::move(done_callback_);
    }

    if (done_callback) {
        try {
            done_callback(ok);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("MemoryAsyncContext::waitDone callback failed: %s", e.what());
            record_failure(e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("MemoryAsyncContext::waitDone callback failed with unknown exception");
            record_failure("unknown done callback exception");
        }
    }

    finalize();
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

void MemoryAsyncContext::markFailed(const std::string& reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_         = true;
        failure_reason_ = reason;
        result_ready_   = true;
    }
    cv_.notify_all();
}

}  // namespace rtp_llm
