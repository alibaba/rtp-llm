#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"

namespace rtp_llm {

class CacheLockGuard final {
public:
    CacheLockGuard(std::mutex& mutex, const BlockTreeCacheMetricsReporter& reporter, const char* operation):
        CacheLockGuard(mutex, reporter, operation, std::defer_lock) {
        lock();
    }

    CacheLockGuard(std::mutex&                          mutex,
                   const BlockTreeCacheMetricsReporter& reporter,
                   const char*                          operation,
                   std::defer_lock_t):
        reporter_(&reporter), operation_(operation), lock_(mutex, std::defer_lock) {}

    template<typename StopPredicate>
    CacheLockGuard(std::mutex&                           mutex,
                   const BlockTreeCacheMetricsReporter*  reporter,
                   const char*                           operation,
                   std::chrono::steady_clock::time_point deadline,
                   std::chrono::steady_clock::duration   retry_interval,
                   std::mutex&                           wait_mutex,
                   std::condition_variable&              condition,
                   StopPredicate                         stop):
        reporter_(reporter), operation_(operation), lock_(mutex, std::defer_lock) {
        begin_ = Clock::now();
        while (!stop() && Clock::now() < deadline) {
            if (lock_.try_lock()) {
                acquired_ = Clock::now();
                return;
            }
            std::unique_lock<std::mutex> wait_lock(wait_mutex);
            condition.wait_until(wait_lock, std::min(deadline, Clock::now() + retry_interval), stop);
        }
    }

    bool owns_lock() const noexcept {
        return lock_.owns_lock();
    }

    void lock() {
        begin_ = Clock::now();
        lock_.lock();
        acquired_ = Clock::now();
    }

    ~CacheLockGuard() {
        if (!lock_.owns_lock()) {
            return;
        }
        const auto released = Clock::now();
        lock_.unlock();
        if (reporter_) {
            reporter_->reportLockMetrics(operation_,
                                         std::chrono::duration<double, std::micro>(acquired_ - begin_).count(),
                                         std::chrono::duration<double, std::micro>(released - acquired_).count());
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    const BlockTreeCacheMetricsReporter* reporter_;
    const char*                          operation_;
    std::unique_lock<std::mutex>         lock_;
    Clock::time_point                    begin_;
    Clock::time_point                    acquired_;
};

}  // namespace rtp_llm
