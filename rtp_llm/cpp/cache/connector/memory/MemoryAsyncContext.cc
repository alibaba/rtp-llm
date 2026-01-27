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
    if (done()) {
        return;
    }
    if (broadcast_result_) {
        broadcast_result_->waitDone();
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

}  // namespace rtp_llm