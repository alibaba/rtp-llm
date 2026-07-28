#include "rtp_llm/cpp/cache/block_tree_cache/LoadAsyncContext.h"

namespace rtp_llm {

LoadAsyncContext::LoadAsyncContext(size_t pending_transfer_count): remaining_transfer_count_(pending_transfer_count) {
    if (remaining_transfer_count_ == 0) {
        state_.store(State::SUCCEEDED);
    }
}

bool LoadAsyncContext::requestCancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    const State                 state = state_.load();
    if (state == State::PENDING) {
        state_.store(State::CANCEL_REQUESTED);
        return true;
    }
    return state == State::CANCEL_REQUESTED;
}

bool LoadAsyncContext::isRequestCanceled() const {
    const State state = state_.load();
    return state == State::CANCEL_REQUESTED || state == State::CANCELLED;
}

bool LoadAsyncContext::completeOne(bool success) {
    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if ((state != State::PENDING && state != State::CANCEL_REQUESTED) || remaining_transfer_count_ == 0) {
            return false;
        }
        has_failure_ = has_failure_ || !success;
        --remaining_transfer_count_;
        if (remaining_transfer_count_ == 0) {
            if (state == State::CANCEL_REQUESTED) {
                state_.store(State::CANCELLED);
            } else if (has_failure_) {
                state_.store(State::FAILED);
            } else {
                state_.store(State::SUCCEEDED);
            }
            notify = true;
        }
    }
    if (notify) {
        cv_.notify_all();
    }
    return true;
}

bool LoadAsyncContext::onTaskFail() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING && state != State::CANCEL_REQUESTED) {
            return false;
        }
        remaining_transfer_count_ = 0;
        if (state == State::CANCEL_REQUESTED) {
            state_.store(State::CANCELLED);
        } else {
            state_.store(State::FAILED);
        }
    }
    cv_.notify_all();
    return true;
}

void LoadAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done(); });
}

bool LoadAsyncContext::done() const {
    const State state = state_.load();
    return state == State::SUCCEEDED || state == State::FAILED || state == State::CANCELLED;
}

bool LoadAsyncContext::success() const {
    return state_.load() == State::SUCCEEDED;
}

}  // namespace rtp_llm
