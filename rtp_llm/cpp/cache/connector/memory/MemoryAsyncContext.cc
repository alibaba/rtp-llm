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

// ----------------------------- MemoryAsyncContext ---------------------------------

bool MemoryAsyncContext::done() const {
    return already_done_.load();
}

bool MemoryAsyncContext::success() const {
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

void MemoryAsyncContext::waitDone() {
    std::lock_guard<std::mutex> lock(wait_mutex_);
    if (done()) {
        return;
    }
    bool wait_ok = true;
    if (broadcast_result_) {
        try {
            wait_ok = wait_timeout_ms_ > 0 ? broadcast_result_->waitDone(wait_timeout_ms_) :
                                             (broadcast_result_->waitDone(), true);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("memory async broadcast wait failed: %s", e.what());
            wait_ok = false;
        } catch (...) {
            RTP_LLM_LOG_WARNING("memory async broadcast wait failed with unknown exception");
            wait_ok = false;
        }
    } else {
        wait_ok = false;
    }
    if (!wait_ok) {
        RTP_LLM_LOG_WARNING("memory async broadcast wait did not complete, timeout_ms=%d", wait_timeout_ms_);
        broadcast_result_.reset();
    }
    if (done_callback_) {
        done_callback_(success());
    }
    already_done_.store(true);
}

void MemoryAsyncContext::setBroadcastResult(
    const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result) {
    broadcast_result_ = result;
}

void MemoryAsyncContext::setWaitTimeoutMs(int timeout_ms) {
    wait_timeout_ms_ = timeout_ms;
}

}  // namespace rtp_llm
