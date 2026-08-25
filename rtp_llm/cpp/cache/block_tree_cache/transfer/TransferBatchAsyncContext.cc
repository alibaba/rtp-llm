#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <utility>

namespace rtp_llm {

void TransferBatchAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return done_; });
}

void TransferBatchAsyncContext::onDone(DoneCallback callback) {
    if (!callback) {
        return;
    }
    bool      run_now = false;
    ErrorInfo error   = ErrorInfo::OkStatus();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            run_now = true;
            error   = error_;
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }
    if (run_now) {
        callback(std::move(error));
    }
}

bool TransferBatchAsyncContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool TransferBatchAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && error_.ok();
}

ErrorInfo TransferBatchAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void TransferBatchAsyncContext::complete(ErrorInfo error) {
    std::vector<DoneCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        error_ = std::move(error);
        done_  = true;
        completion_guard_.reset();
        callbacks.swap(callbacks_);
    }
    done_cv_.notify_all();
    for (auto& callback : callbacks) {
        callback(error_);
    }
}

}  // namespace rtp_llm
