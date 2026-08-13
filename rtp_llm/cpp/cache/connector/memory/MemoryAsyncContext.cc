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
    return already_done_.load(std::memory_order_acquire);
}

bool MemoryAsyncContext::success() const {
    return done() && completion_success_.load(std::memory_order_acquire);
}

void MemoryAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this]() { return done(); });
}

void MemoryAsyncContext::complete(bool success) {
    std::function<void(bool)> callback;
    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (done() || completion_started_) {
            return;
        }
        completion_started_ = true;
        callback = std::move(done_callback_);
    }
    if (callback) {
        try {
            callback(success);
        } catch (...) {
            success = false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        completion_success_.store(success, std::memory_order_release);
        already_done_.store(true, std::memory_order_release);
    }
    completion_cv_.notify_all();
}

}  // namespace rtp_llm
