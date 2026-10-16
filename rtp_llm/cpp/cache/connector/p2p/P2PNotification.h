#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>

namespace rtp_llm {

// Separate lifetime from its consumer. Producers retain only weak_ptrs to this
// object, never a checker/worker pointer. The generation closes notify-before-wait.
class P2PNotification {
public:
    uint64_t generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return generation_;
    }
    void notify() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++generation_;
        }
        cv_.notify_all();
    }
    void waitUntil(uint64_t observed, int64_t deadline_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto                   changed = [&] { return generation_ != observed; };
        if (deadline_ms == std::numeric_limits<int64_t>::max()) {
            cv_.wait(lock, changed);
        } else {
            cv_.wait_until(
                lock, std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)), changed);
        }
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    uint64_t                generation_{0};
};

}  // namespace rtp_llm
