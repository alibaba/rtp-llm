#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rtp_llm {

// Prevents a timed-out load from writing into cache blocks after their ownership
// has been returned to the allocator. A close first rejects and drains copy-based
// writes, then notifies direct-write transports so they can close their connection
// and complete the request before the buffers are released.
class LoadCopyFence {
public:
    template<typename F>
    bool runIfOpen(F&& copy) {
        if (closed_.load(std::memory_order_acquire)) {
            return false;
        }
        std::shared_lock<std::shared_mutex> lock(copy_mutex_);
        if (closed_.load(std::memory_order_acquire)) {
            return false;
        }
        std::forward<F>(copy)();
        return true;
    }

    void closeAndDrain() {
        closed_.store(true, std::memory_order_release);
        {
            std::unique_lock<std::shared_mutex> lock(copy_mutex_);
        }

        std::vector<std::function<void()>> callbacks;
        {
            std::lock_guard<std::mutex> lock(close_callbacks_mutex_);
            callbacks = std::move(close_callbacks_);
        }
        for (auto& callback : callbacks) {
            callback();
        }
    }

    void addCloseCallback(std::function<void()> callback) {
        bool run_now = false;
        {
            std::lock_guard<std::mutex> lock(close_callbacks_mutex_);
            if (closed_.load(std::memory_order_acquire)) {
                run_now = true;
            } else {
                close_callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback();
        }
    }

    bool closed() const {
        return closed_.load(std::memory_order_acquire);
    }

private:
    std::atomic_bool                   closed_{false};
    std::shared_mutex                  copy_mutex_;
    std::mutex                         close_callbacks_mutex_;
    std::vector<std::function<void()>> close_callbacks_;
};

}  // namespace rtp_llm
