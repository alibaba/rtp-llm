#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <utility>

namespace rtp_llm {

void TransferBatchAsyncContext::complete(ErrorInfo error_info) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        // Release endpoint reservations before publishing completion.
        completion_guard_.reset();
        error_info_ = std::move(error_info);
        done_       = true;
    }
    completion_cv_.notify_all();
}

void TransferBatchAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    completion_cv_.wait(lock, [this] { return done_; });
}

bool TransferBatchAsyncContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool TransferBatchAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && error_info_.ok();
}

ErrorInfo TransferBatchAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_info_;
}

}  // namespace rtp_llm
