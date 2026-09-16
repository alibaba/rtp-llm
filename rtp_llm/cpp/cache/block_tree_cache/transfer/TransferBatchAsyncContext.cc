#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <exception>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
void invokeDoneCallback(const AsyncContext::DoneCallback& callback, const ErrorInfo& error) {
    try {
        callback(error);
    } catch (const std::exception& exception) {
        RTP_LLM_LOG_WARNING("transfer completion callback threw: %s", exception.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("transfer completion callback threw an unknown exception");
    }
}
}  // namespace

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
        invokeDoneCallback(callback, error);
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
    std::shared_ptr<void>     completion_guard;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        error_           = std::move(error);
        done_            = true;
        completion_guard = std::move(completion_guard_);
        callbacks.swap(callbacks_);
    }
    // Guard destruction may re-enter this context; never run it under mutex_.
    completion_guard.reset();
    done_cv_.notify_all();
    for (auto& callback : callbacks) {
        invokeDoneCallback(callback, error_);
    }
}

}  // namespace rtp_llm
