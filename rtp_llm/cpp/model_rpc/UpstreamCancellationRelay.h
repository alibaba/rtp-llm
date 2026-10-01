#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace rtp_llm {

// A synchronous gRPC read does not wake when its *other* (upstream) RPC is
// cancelled. Relay that cancellation while the downstream RPC is in flight.
class UpstreamCancellationRelay {
public:
    using Callback = std::function<void()>;
    using Predicate = std::function<bool()>;

    UpstreamCancellationRelay(Predicate is_cancelled,
                              Callback  cancel_downstream,
                              std::chrono::milliseconds poll_interval = std::chrono::milliseconds(20)):
        is_cancelled_(std::move(is_cancelled)),
        cancel_downstream_(std::move(cancel_downstream)),
        poll_interval_(poll_interval),
        worker_([this] { run(); }) {}

    UpstreamCancellationRelay(const UpstreamCancellationRelay&) = delete;
    UpstreamCancellationRelay& operator=(const UpstreamCancellationRelay&) = delete;

    ~UpstreamCancellationRelay() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        condition_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopped_) {
            lock.unlock();
            const bool cancelled = is_cancelled_();
            if (cancelled) {
                cancel_downstream_();
                return;
            }
            lock.lock();
            condition_.wait_for(lock, poll_interval_, [this] { return stopped_; });
        }
    }

    Predicate                 is_cancelled_;
    Callback                  cancel_downstream_;
    std::chrono::milliseconds poll_interval_;
    std::mutex                mutex_;
    std::condition_variable   condition_;
    bool                      stopped_ = false;
    std::thread               worker_;
};

}  // namespace rtp_llm
