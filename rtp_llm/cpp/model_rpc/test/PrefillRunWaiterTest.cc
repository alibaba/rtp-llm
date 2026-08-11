#include "rtp_llm/cpp/model_rpc/PrefillRunWaiter.h"

#include <gtest/gtest.h>

using namespace std::chrono_literals;

namespace rtp_llm {

TEST(PrefillRunWaiterTest, ConfiguredTimeoutLimitsLongerServerDeadline) {
    const auto steady_now      = PrefillRunSteadyClock::time_point(10s);
    const auto system_now      = PrefillRunSystemClock::time_point(20s);
    const auto server_deadline = system_now + 500ms;

    const auto deadline = makePrefillRunDeadline(steady_now, 200ms, system_now, server_deadline);

    EXPECT_EQ(deadline.value, steady_now + 200ms);
    EXPECT_FALSE(deadline.limited_by_server_context);
}

TEST(PrefillRunWaiterTest, ServerDeadlineLimitsConfiguredTimeout) {
    const auto steady_now      = PrefillRunSteadyClock::time_point(10s);
    const auto system_now      = PrefillRunSystemClock::time_point(20s);
    const auto server_deadline = system_now + 80ms;

    const auto deadline = makePrefillRunDeadline(steady_now, 200ms, system_now, server_deadline);

    EXPECT_EQ(deadline.value, steady_now + 80ms);
    EXPECT_TRUE(deadline.limited_by_server_context);
}

TEST(PrefillRunWaiterTest, ExpiredServerDeadlineDoesNotWait) {
    const auto steady_now = PrefillRunSteadyClock::time_point(10s);
    const auto system_now = PrefillRunSystemClock::time_point(20s);

    const auto deadline = makePrefillRunDeadline(steady_now, 200ms, system_now, system_now - 1ms);

    EXPECT_EQ(deadline.value, steady_now);
    EXPECT_TRUE(deadline.limited_by_server_context);
}

TEST(PrefillRunWaiterTest, MissingServerDeadlineUsesConfiguredTimeout) {
    const auto steady_now = PrefillRunSteadyClock::time_point(10s);
    const auto system_now = PrefillRunSystemClock::time_point(20s);

    const auto deadline = makePrefillRunDeadline(
        steady_now, 200ms, system_now, PrefillRunSystemClock::time_point::max());

    EXPECT_EQ(deadline.value, steady_now + 200ms);
    EXPECT_FALSE(deadline.limited_by_server_context);
}

TEST(PrefillRunWaiterTest, CancellationInterruptsQueuedWait) {
    auto now         = PrefillRunSteadyClock::time_point();
    bool cancelled   = false;
    int  wait_count  = 0;
    const auto result = waitForPrefillRun(
        []() { return false; },
        []() { return false; },
        [&cancelled]() { return cancelled; },
        now + 100ms,
        10ms,
        [&now]() { return now; },
        [&now, &cancelled, &wait_count](PrefillRunSteadyClock::time_point wakeup) {
            now = wakeup;
            ++wait_count;
            cancelled = true;
        });

    EXPECT_EQ(result, PrefillRunWaitResult::Cancelled);
    EXPECT_EQ(wait_count, 1);
}

TEST(PrefillRunWaiterTest, AbsoluteDeadlineBoundsAllPolls) {
    auto now        = PrefillRunSteadyClock::time_point();
    int  wait_count = 0;
    const auto result = waitForPrefillRun(
        []() { return false; },
        []() { return false; },
        []() { return false; },
        now + 25ms,
        10ms,
        [&now]() { return now; },
        [&now, &wait_count](PrefillRunSteadyClock::time_point wakeup) {
            now = wakeup;
            ++wait_count;
        });

    EXPECT_EQ(result, PrefillRunWaitResult::DeadlineExceeded);
    EXPECT_EQ(now, PrefillRunSteadyClock::time_point() + 25ms);
    EXPECT_EQ(wait_count, 3);
}

TEST(PrefillRunWaiterTest, ReachedDeadlineWinsOverCancellation) {
    const auto deadline = PrefillRunSteadyClock::time_point(25ms);
    const auto result   = waitForPrefillRun(
        []() { return false; },
        []() { return false; },
        []() { return true; },
        deadline,
        10ms,
        [deadline]() { return deadline; },
        [](PrefillRunSteadyClock::time_point) {});

    EXPECT_EQ(result, PrefillRunWaitResult::DeadlineExceeded);
}

TEST(PrefillRunWaiterTest, StreamErrorWinsWithoutSleeping) {
    int wait_count = 0;
    const auto result = waitForPrefillRun(
        []() { return true; },
        []() { return false; },
        []() { return true; },
        PrefillRunSteadyClock::time_point::max(),
        10ms,
        []() { return PrefillRunSteadyClock::time_point(); },
        [&wait_count](PrefillRunSteadyClock::time_point) { ++wait_count; });

    EXPECT_EQ(result, PrefillRunWaitResult::StreamError);
    EXPECT_EQ(wait_count, 0);
}

TEST(PrefillRunWaiterTest, ReadyStreamReturnsWithoutSleeping) {
    int wait_count = 0;
    const auto result = waitForPrefillRun(
        []() { return false; },
        []() { return true; },
        []() { return false; },
        PrefillRunSteadyClock::time_point::max(),
        10ms,
        []() { return PrefillRunSteadyClock::time_point(); },
        [&wait_count](PrefillRunSteadyClock::time_point) { ++wait_count; });

    EXPECT_EQ(result, PrefillRunWaitResult::Ready);
    EXPECT_EQ(wait_count, 0);
}

}  // namespace rtp_llm
