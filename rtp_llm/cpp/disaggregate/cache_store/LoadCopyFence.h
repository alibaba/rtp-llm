#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace rtp_llm {

// Prevents a timed-out load from writing into cache blocks after their ownership
// has been returned to the allocator. A close first rejects future copies, then
// drains copies that already acquired a shared permit.
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
        std::unique_lock<std::shared_mutex> lock(copy_mutex_);
    }

    bool closed() const {
        return closed_.load(std::memory_order_acquire);
    }

private:
    std::atomic_bool  closed_{false};
    std::shared_mutex copy_mutex_;
};

}  // namespace rtp_llm
