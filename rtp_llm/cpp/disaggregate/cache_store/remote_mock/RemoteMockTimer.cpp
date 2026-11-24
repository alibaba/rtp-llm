#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockTimer.h"
#include <limits>
#include <utility>

namespace rtp_llm {
RemoteMockTimer::RemoteMockTimer(size_t capacity, Now now, bool threaded): capacity_(capacity), now_(std::move(now)) {
    if (threaded) {
        thread_    = std::thread([this] { run(); });
        worker_id_ = thread_.get_id();
    }
}
RemoteMockTimer::~RemoteMockTimer() {
    stop();
}
void RemoteMockTimer::invoke(Callback& callback, bool cancelled) noexcept {
    try {
        callback(cancelled);
    } catch (...) { /* One failed handler must not strand other admitted operations. */
    }
}
bool RemoteMockTimer::schedule(Clock::time_point deadline, Callback callback) {
    if (!callback)
        return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ || queue_.size() >= capacity_ || sequence_ == UINT64_MAX)
            return false;
        queue_.push({deadline, sequence_++, std::move(callback)});
    }
    condition_.notify_one();
    return true;
}
void RemoteMockTimer::pump() {
    for (;;) {
        Callback callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_ || queue_.empty() || queue_.top().deadline > now_())
                return;
            callback = std::move(queue_.top().callback);
            queue_.pop();
        }
        invoke(callback, false);
    }
}
void RemoteMockTimer::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopped_) {
        if (queue_.empty()) {
            condition_.wait(lock);
            continue;
        }
        const auto deadline = queue_.top().deadline;
        if (now_() < deadline) {
            condition_.wait_until(lock, deadline);
            continue;
        }
        auto callback = std::move(queue_.top().callback);
        queue_.pop();
        lock.unlock();
        invoke(callback, false);
        lock.lock();
    }
}
void RemoteMockTimer::stop() {
    // Move the heap, avoiding allocation during shutdown. Reentrant cancellation is safe.
    decltype(queue_) cancelled;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cancelled.swap(queue_);
    }
    condition_.notify_all();
    while (!cancelled.empty()) {
        auto callback = std::move(cancelled.top().callback);
        cancelled.pop();
        invoke(callback, true);
    }
    // worker_id_ is immutable after construction; no join_mutex on the worker, so an
    // external joining stop cannot deadlock a callback which itself calls stop().
    if (std::this_thread::get_id() != worker_id_) {
        std::lock_guard<std::mutex> join_lock(join_mutex_);
        if (thread_.joinable())
            thread_.join();
    }
}
}  // namespace rtp_llm
