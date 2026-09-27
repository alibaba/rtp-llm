#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"

namespace rtp_llm::test {
namespace {
using namespace std::chrono_literals;

// No model/device allocation: the warmup constructor accepts an empty topology.
std::shared_ptr<KVCacheManager> makeManager() {
    return std::make_shared<KVCacheManager>(CacheConfig{}, /*warmup=*/true);
}

class CallbackBarrier {
public:
    void report() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this]() { return released_; });
        dependency_was_alive = dependency_alive.load();
        completed            = true;
    }
    bool waitEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [this]() { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
    std::atomic<bool> dependency_alive{true};
    std::atomic<bool> dependency_was_alive{false};
    std::atomic<bool> completed{false};

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    entered_{false};
    bool                    released_{false};
};

bool waitStopRequested(KVCacheManager& manager) {
    std::unique_lock<std::mutex> lock(manager.metrics_wait_mutex_);
    return manager.metrics_wait_cv_.wait_for(
        lock, 5s, [&manager]() { return manager.stop_.load(std::memory_order_acquire); });
}
}  // namespace

TEST(KVCacheMetricsLifecycleTest, PartialInitializationAndRepeatedStop) {
    auto manager = makeManager();
    EXPECT_EQ(manager->allocator_, nullptr);
    EXPECT_FALSE(manager->metrics_reporter_thread_.joinable());
    manager->stopMetricsReporting();
    manager->stopMetricsReporting();
    EXPECT_TRUE(manager->stop_.load());
    EXPECT_FALSE(manager->metrics_reporter_thread_.joinable());
}

TEST(KVCacheMetricsLifecycleTest, RetainedOwnerCannotKeepCallbackAliveAfterStop) {
    auto            manager        = makeManager();
    auto            retained_owner = manager;
    CallbackBarrier callback;
    manager->metrics_reporter_thread_ = std::thread([&callback]() { callback.report(); });
    EXPECT_TRUE(callback.waitEntered());
    auto stopping = std::async(std::launch::async, [&manager]() { manager->stopMetricsReporting(); });
    EXPECT_TRUE(waitStopRequested(*manager));
    // The callback is held at a barrier: this is an ordering assertion, not a sleep.
    EXPECT_EQ(stopping.wait_for(0s), std::future_status::timeout);
    EXPECT_FALSE(callback.completed.load());
    callback.release();
    EXPECT_EQ(stopping.wait_for(5s), std::future_status::ready);
    stopping.get();
    callback.dependency_alive = false;
    EXPECT_TRUE(callback.completed.load());
    EXPECT_TRUE(callback.dependency_was_alive.load());
    EXPECT_FALSE(manager->metrics_reporter_thread_.joinable());
    EXPECT_EQ(retained_owner.get(), manager.get());
    manager.reset();
    EXPECT_TRUE(retained_owner->stop_.load());
    retained_owner->stopMetricsReporting();
}

TEST(KVCacheMetricsLifecycleTest, ConcurrentStopCallersJoinOnlyOnce) {
    auto            manager = makeManager();
    CallbackBarrier callback;
    manager->metrics_reporter_thread_ = std::thread([&callback]() { callback.report(); });
    EXPECT_TRUE(callback.waitEntered());
    auto first  = std::async(std::launch::async, [&manager]() { manager->stopMetricsReporting(); });
    auto second = std::async(std::launch::async, [&manager]() { manager->stopMetricsReporting(); });
    EXPECT_TRUE(waitStopRequested(*manager));
    EXPECT_EQ(first.wait_for(0s), std::future_status::timeout);
    EXPECT_EQ(second.wait_for(0s), std::future_status::timeout);
    callback.release();
    EXPECT_EQ(first.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(second.wait_for(5s), std::future_status::ready);
    first.get();
    second.get();
    EXPECT_FALSE(manager->metrics_reporter_thread_.joinable());
}

TEST(KVCacheMetricsLifecycleTest, RealIdleReportLoopStopsWithoutDependencies) {
    auto manager                      = makeManager();
    manager->metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, manager.get());
    manager->stopMetricsReporting();
    EXPECT_TRUE(manager->stop_.load());
    EXPECT_FALSE(manager->metrics_reporter_thread_.joinable());
}

TEST(KVCacheMetricsLifecycleTest, DestructorIsTheSameJoinBackstop) {
    auto            manager           = makeManager();
    auto*           alive_during_join = manager.get();
    CallbackBarrier callback;
    manager->metrics_reporter_thread_ = std::thread([&callback]() { callback.report(); });
    EXPECT_TRUE(callback.waitEntered());
    auto destroying = std::async(std::launch::async, [owner = std::move(manager)]() mutable { owner.reset(); });
    EXPECT_TRUE(waitStopRequested(*alive_during_join));
    EXPECT_EQ(destroying.wait_for(0s), std::future_status::timeout);
    callback.release();
    EXPECT_EQ(destroying.wait_for(5s), std::future_status::ready);
    destroying.get();
    EXPECT_TRUE(callback.completed.load());
    EXPECT_TRUE(callback.dependency_was_alive.load());
}
}  // namespace rtp_llm::test
