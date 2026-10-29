#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rtp_llm {

struct SchedulerSynchronizationState {
    void notify() {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            ++wake_generation;
        }
        cond.notify_all();
    }

    void notifyCacheLoadCompletion() {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            // Admission may have registered another load after the count reached zero.
            if (pending_cache_load_count.load(std::memory_order_acquire) != 0) {
                return;
            }
            cache_load_completion_trigger = true;
            ++wake_generation;
        }
        cond.notify_all();
    }

    // The predicate runs under both mutexes and must only inspect state: no allocator
    // calls or callbacks. On return the caller still owns the scheduler mutex only.
    template<typename Predicate>
    void wait(std::unique_lock<std::mutex>& scheduler_lock,
              Predicate                     predicate,
              std::chrono::milliseconds     timeout = std::chrono::milliseconds::max()) {
        assert(scheduler_lock.owns_lock() && scheduler_lock.mutex() == &mutex);
        const bool timed = timeout != std::chrono::milliseconds::max();
        const auto deadline =
            timed ? std::chrono::steady_clock::now() + timeout : std::chrono::steady_clock::time_point::max();
        for (;;) {
            std::unique_lock<std::mutex> wake_lock(wake_mutex);
            if (predicate() || (timed && std::chrono::steady_clock::now() >= deadline)) {
                return;
            }
            const auto observed = wake_generation;
            // Keep wake_mutex across the predicate-to-park window. Notifiers never
            // acquire the scheduler mutex, including callbacks under allocator locks.
            scheduler_lock.unlock();
            if (timed) {
                cond.wait_until(wake_lock, deadline, [&] { return wake_generation != observed; });
            } else {
                cond.wait(wake_lock, [&] { return wake_generation != observed; });
            }
            // Never invert the scheduler -> wake lock order when waking up.
            wake_lock.unlock();
            scheduler_lock.lock();
        }
    }

    std::mutex              mutex;
    std::mutex              wake_mutex;
    std::condition_variable cond;
    std::atomic<size_t>     pending_cache_load_count{0};
    // Protected by wake_mutex. Do not hold it across stream/allocator operations.
    uint64_t wake_generation{0};
    bool     cache_load_completion_trigger{false};
};

using SchedulerSynchronizationStatePtr = std::shared_ptr<SchedulerSynchronizationState>;

// A registered callback may be discarded when an unfinished load context is destroyed.
// Share this registration across callback copies and release its count exactly once.
class PendingCacheLoadRegistration {
public:
    explicit PendingCacheLoadRegistration(const SchedulerSynchronizationStatePtr& sync_state): sync_state_(sync_state) {
        sync_state_->pending_cache_load_count.fetch_add(1, std::memory_order_acq_rel);
    }

    ~PendingCacheLoadRegistration() {
        release();
    }

    void release() {
        if (released_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        const size_t previous = sync_state_->pending_cache_load_count.fetch_sub(1, std::memory_order_acq_rel);
        assert(previous > 0);
        if (previous == 1) {
            // May run under scheduler or allocator locks; only acquire the wake mutex.
            sync_state_->notifyCacheLoadCompletion();
        }
    }

private:
    const SchedulerSynchronizationStatePtr sync_state_;
    std::atomic<bool>                      released_{false};
};

}  // namespace rtp_llm
