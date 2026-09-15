#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <exception>
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
    std::exception_ptr callback_error;
    for (auto& callback : callbacks) {
        try {
            callback(error_);
        } catch (...) {
            if (!callback_error) {
                callback_error = std::current_exception();
            }
        }
    }
    if (callback_error) {
        std::rethrow_exception(callback_error);
    }
}

}  // namespace rtp_llm
