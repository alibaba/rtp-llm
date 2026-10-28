#include "rtp_llm/cpp/engine_base/schedulers/SchedulerSynchronizationState.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

TEST(SchedulerSynchronizationStateTest, ReadyPredicateDoesNotWait) {
    SchedulerSynchronizationState state;
    std::unique_lock<std::mutex>  lock(state.mutex);
    state.wait(lock, [] { return true; });
    EXPECT_TRUE(lock.owns_lock());
}

TEST(SchedulerSynchronizationStateTest, CompletionBeforeWaitIsObserved) {
    SchedulerSynchronizationState state;
    state.notifyCacheLoadCompletion();
    std::unique_lock<std::mutex> lock(state.mutex);
    state.wait(lock, [&] { return state.cache_load_completion_trigger; });
    EXPECT_TRUE(lock.owns_lock());
}

TEST(SchedulerSynchronizationStateTest, CompletionBetweenPredicateAndParkIsNotLost) {
    SchedulerSynchronizationState state;
    std::promise<void>            checked;
    std::promise<void>            release_predicate;
    auto                          release = release_predicate.get_future();
    auto                          waiter  = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        bool                         first = true;
        state.wait(
            lock,
            [&] {
                const bool ready = state.cache_load_completion_trigger;
                if (first) {
                    first = false;
                    checked.set_value();
                    release.wait();
                }
                return ready;
            },
            2s);
    });
    checked.get_future().wait();

    // Probe from a different thread while the predicate is held at the gate.
    // This catches a missing wake lock even if the notifier is scheduled late.
    const bool wake_lock_available = state.wake_mutex.try_lock();
    if (wake_lock_available) {
        state.wake_mutex.unlock();
    }
    EXPECT_FALSE(wake_lock_available);

    std::promise<void> notifying;
    auto               completion = std::async(std::launch::async, [&] {
        notifying.set_value();
        state.notifyCacheLoadCompletion();
    });
    const auto         started    = notifying.get_future().wait_for(1s);
    EXPECT_EQ(started, std::future_status::ready);
    EXPECT_EQ(completion.wait_for(20ms), std::future_status::timeout);
    // Always release the gate, including when an expectation above fails.
    release_predicate.set_value();
    completion.get();
    EXPECT_EQ(waiter.wait_for(1s), std::future_status::ready);
    waiter.get();
}

TEST(SchedulerSynchronizationStateTest, NotificationRechecksRatherThanReplacesBusinessPredicate) {
    SchedulerSynchronizationState state;
    bool                          ready = false;  // Protected by state.mutex.
    std::promise<void>            checked;
    std::promise<void>            rechecked;
    auto                          waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        size_t                       checks = 0;
        state.wait(
            lock,
            [&] {
                if (++checks == 1) {
                    checked.set_value();
                } else if (checks == 2) {
                    rechecked.set_value();
                }
                return ready;
            },
            2s);
        return ready;
    });
    checked.get_future().wait();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.notify();
    }
    EXPECT_EQ(rechecked.get_future().wait_for(1s), std::future_status::ready);
    EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        ready = true;
    }
    state.notify();
    EXPECT_TRUE(waiter.get());
}

TEST(SchedulerSynchronizationStateTest, NotificationsDoNotExtendTimeout) {
    SchedulerSynchronizationState state;
    std::atomic<bool>             finished{false};
    auto                          waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        const auto                   start = std::chrono::steady_clock::now();
        state.wait(lock, [] { return false; }, 50ms);
        return std::chrono::steady_clock::now() - start;
    });
    std::thread                   notifier([&] {
        while (!finished.load()) {
            state.notify();
            std::this_thread::sleep_for(1ms);
        }
    });
    const auto                    status = waiter.wait_for(1s);
    finished.store(true);
    notifier.join();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_GE(waiter.get(), 50ms);
}

TEST(SchedulerSynchronizationStateTest, ReleasesWakeMutexBeforeReacquiringSchedulerMutex) {
    SchedulerSynchronizationState state;
    bool                          ready = false;  // Protected by state.mutex.
    std::promise<void>            checked;
    auto                          waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        bool                         first = true;
        state.wait(
            lock,
            [&] {
                if (first) {
                    first = false;
                    checked.set_value();
                }
                return ready;
            },
            2s);
    });
    checked.get_future().wait();
    std::unique_lock<std::mutex> lock(state.mutex);
    ready = true;
    state.notify();
    EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
    auto       notification = std::async(std::launch::async, [&] { state.notify(); });
    const auto status       = notification.wait_for(1s);
    lock.unlock();
    EXPECT_EQ(status, std::future_status::ready);
    notification.get();
    waiter.get();
}

}  // namespace
}  // namespace rtp_llm
