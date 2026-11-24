#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockTimer.h"
#include <gtest/gtest.h>
#include <future>
#include <stdexcept>

namespace rtp_llm {
TEST(RemoteMockTimerTest, DelayNeverFiresBeforeDeadline) {
    using namespace std::chrono;
    auto             now = steady_clock::time_point{};
    RemoteMockTimer  timer(8, [&] { return now; }, false);
    std::vector<int> fired;
    for (int delay : {0, 5, 20}) {
        ASSERT_TRUE(timer.schedule(now + milliseconds(delay), [&, delay](bool cancelled) {
            EXPECT_FALSE(cancelled);
            fired.push_back(delay);
        }));
    }
    timer.pump();
    EXPECT_EQ(fired, std::vector<int>({0}));
    now += milliseconds(4);
    timer.pump();
    EXPECT_EQ(fired.size(), 1u);
    now += milliseconds(1);
    timer.pump();
    EXPECT_EQ(fired, std::vector<int>({0, 5}));
    now += milliseconds(14);
    timer.pump();
    EXPECT_EQ(fired.size(), 2u);
    now += milliseconds(1);
    timer.pump();
    EXPECT_EQ(fired, std::vector<int>({0, 5, 20}));
}
TEST(RemoteMockTimerTest, StopCancelsExactlyOnce) {
    auto            now = std::chrono::steady_clock::time_point{};
    RemoteMockTimer timer(2, [&] { return now; }, false);
    int             calls = 0;
    auto            cb    = [&](bool cancelled) {
        EXPECT_TRUE(cancelled);
        ++calls;
    };
    EXPECT_TRUE(timer.schedule(now + std::chrono::seconds(1), cb));
    EXPECT_TRUE(timer.schedule(now + std::chrono::seconds(2), cb));
    EXPECT_FALSE(timer.schedule(now, cb));
    timer.stop();
    timer.stop();
    timer.pump();
    EXPECT_EQ(calls, 2);
    EXPECT_FALSE(timer.schedule(now, cb));
    EXPECT_EQ(calls, 2);
}
TEST(RemoteMockTimerTest, CallbacksCanReenterAndExceptionsDoNotLoseWork) {
    auto            now = std::chrono::steady_clock::time_point{};
    RemoteMockTimer timer(4, [&] { return now; }, false);
    int             calls = 0;
    ASSERT_TRUE(timer.schedule(now, [&](bool) {
        ++calls;
        EXPECT_TRUE(timer.schedule(now, [&](bool) { ++calls; }));
    }));
    ASSERT_TRUE(timer.schedule(now, [](bool) { throw std::runtime_error("test"); }));
    timer.pump();
    timer.pump();
    EXPECT_EQ(calls, 2);
}
TEST(RemoteMockTimerTest, StopDrainsExtractedCallbackAndCancelsPending) {
    RemoteMockTimer    timer(4);
    std::promise<void> entered, release, cancelled;
    auto               release_future = release.get_future().share();
    ASSERT_TRUE(timer.schedule(RemoteMockTimer::Clock::now(), [&](bool is_cancelled) {
        EXPECT_FALSE(is_cancelled);
        entered.set_value();
        release_future.wait();
    }));
    auto entered_future = entered.get_future();
    EXPECT_EQ(entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(timer.schedule(RemoteMockTimer::Clock::now() + std::chrono::hours(1), [&](bool is_cancelled) {
        EXPECT_TRUE(is_cancelled);
        cancelled.set_value();
    }));
    auto stopped          = std::async(std::launch::async, [&] { timer.stop(); });
    auto cancelled_future = cancelled.get_future();
    EXPECT_EQ(cancelled_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    release.set_value();
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(2)), std::future_status::ready);
}
TEST(RemoteMockTimerTest, WorkerCallbackMayStopWithoutSelfJoin) {
    RemoteMockTimer    timer(4);
    std::promise<void> stopped;
    ASSERT_TRUE(timer.schedule(RemoteMockTimer::Clock::now(), [&](bool) {
        timer.stop();
        stopped.set_value();
    }));
    auto future = stopped.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    timer.stop();
}
TEST(RemoteMockTimerTest, WorkerUsesSteadyDeadline) {
    RemoteMockTimer                                     timer(4);
    std::promise<std::chrono::steady_clock::time_point> p;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
    ASSERT_TRUE(timer.schedule(deadline, [&](bool cancelled) {
        EXPECT_FALSE(cancelled);
        p.set_value(std::chrono::steady_clock::now());
    }));
    auto future = p.get_future();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_GE(future.get(), deadline);
}
}  // namespace rtp_llm
