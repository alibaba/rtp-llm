#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

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
        CacheLockGuard(mutex, operation, CacheStageContext::forLock(&reporter, operation), std::defer_lock) {}

    CacheLockGuard(std::mutex& mutex, const char* operation, const CacheStageContext& stage_context):
        CacheLockGuard(mutex, operation, stage_context, std::defer_lock) {
        lock();
    }

    template<typename StopPredicate>
    CacheLockGuard(std::mutex&                           mutex,
                   const BlockTreeCacheMetricsReporter*  reporter,
                   const char*                           operation,
                   std::chrono::steady_clock::time_point deadline,
                   std::chrono::steady_clock::duration   retry_interval,
                   std::mutex&                           wait_mutex,
                   std::condition_variable&              condition,
                   StopPredicate                         stop):
        CacheLockGuard(mutex, operation, CacheStageContext::forLock(reporter, operation), std::defer_lock) {
        wait_ = TransferStageTimer(stage_context_, "lock_wait", "cache_lock", AsyncProfileKind::Wait);
        while (!stop() && Clock::now() < deadline) {
            if (lock_.try_lock()) {
                acquired();
                return;
            }
            std::unique_lock<std::mutex> wait_lock(wait_mutex);
            condition.wait_until(wait_lock, std::min(deadline, Clock::now() + retry_interval), stop);
        }
        wait_.finish(false);
    }

    bool owns_lock() const noexcept {
        return lock_.owns_lock();
    }

    void lock() {
        wait_ = TransferStageTimer(stage_context_, "lock_wait", "cache_lock", AsyncProfileKind::Wait);
        lock_.lock();
        acquired();
    }

    ~CacheLockGuard() {
        if (!lock_.owns_lock()) {
            return;
        }
        const auto released = hold_ ? AsyncProfileSession::steadyNowUs() : 0;
        lock_.unlock();
        hold_.finishAtSteady(released, true);
    }

private:
    using Clock = std::chrono::steady_clock;
    CacheLockGuard(std::mutex& mutex, const char* operation, const CacheStageContext& stage_context, std::defer_lock_t):
        stage_context_(stage_context), lock_(mutex, std::defer_lock) {
        stage_context_.operation = operation;
    }
    void acquired() noexcept {
        if (!wait_)
            return;
        const auto acquired_us = AsyncProfileSession::steadyNowUs();
        wait_.finishAtSteady(acquired_us, true);
        hold_ = TransferStageTimer::atSteady(stage_context_, "lock_hold", "cache_lock", acquired_us);
    }
    CacheStageContext            stage_context_;
    std::unique_lock<std::mutex> lock_;
    TransferStageTimer           wait_, hold_;
};

}  // namespace rtp_llm
