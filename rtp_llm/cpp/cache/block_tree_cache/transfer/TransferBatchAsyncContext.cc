#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void TransferBatchAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return done_; });
}

void TransferBatchAsyncContext::onDone(DoneCallback callback) {
    if (!callback) {
        return;
    }
    bool           run_now = false;
    TransferResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            run_now = true;
            result  = result_;
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }
    if (run_now) {
        callback(std::move(result));
    }
}

bool TransferBatchAsyncContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool TransferBatchAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && result_.ok();
}

TransferResult TransferBatchAsyncContext::result() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return result_;
}

void TransferBatchAsyncContext::complete(TransferResult result) {
    // Physical evidence is independent of the business result and its one-shot
    // completion gate. Publish it before any result expansion or user callback.
    if (completion_proof_) {
        completion_proof_->report(result.buffer_safe);
    }
    std::vector<DoneCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        result.expandBlockErrors(block_count_);
        result_ = std::move(result);
        done_   = true;
        completion_guard_.reset();
        callbacks.swap(callbacks_);
    }
    done_cv_.notify_all();
    for (auto& callback : callbacks) {
        try {
            callback(result_);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("transfer completion callback failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("transfer completion callback failed");
        }
    }
}

}  // namespace rtp_llm
