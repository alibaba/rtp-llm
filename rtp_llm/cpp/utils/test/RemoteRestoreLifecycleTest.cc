#include "rtp_llm/cpp/utils/RemoteRestoreLifecycle.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
using Clock = RemoteRestoreTiming::Clock;
using Us    = std::chrono::microseconds;

TEST(RemoteRestoreLifecycleTest, SeparatesGpuReadinessFromRootContextUpdateObservation) {
    RemoteRestoreLifecycle lifecycle;
    const auto             start = Clock::time_point{} + Us(1000);
    ASSERT_TRUE(lifecycle.adopt({start, start + Us(500)}));
    EXPECT_EQ(lifecycle.checkpointReadyUs(), 500);
    EXPECT_TRUE(lifecycle.pendingForward());
    EXPECT_EQ(lifecycle.completeContextForward(start + Us(1200)), 1200);
    EXPECT_FALSE(lifecycle.pendingForward());
    EXPECT_FALSE(lifecycle.completeContextForward(start + Us(1300)).has_value());
}

TEST(RemoteRestoreLifecycleTest, IncludesLookupAndAllocationWithoutLosingReadSubphase) {
    RemoteRestoreLifecycle lifecycle;
    const auto             start = Clock::time_point{} + Us(1000);
    ASSERT_TRUE(lifecycle.adopt({start + Us(200), start + Us(700)}, start));
    EXPECT_EQ(lifecycle.checkpointReadyUs(), 700);
    EXPECT_EQ(lifecycle.readToCheckpointUs(), 500);
    EXPECT_EQ(lifecycle.completeContextForward(start + Us(1200)), 1200);
    lifecycle.reset();
    EXPECT_FALSE(lifecycle.readToCheckpointUs().has_value());
    EXPECT_FALSE(lifecycle.adopt({start, start + Us(10)}, start + Us(1)));
    EXPECT_FALSE(lifecycle.pendingForward());
    ASSERT_TRUE(lifecycle.adopt({start + Us(2000), start + Us(2030)}, start + Us(1900)));
    EXPECT_EQ(lifecycle.checkpointReadyUs(), 130);
    EXPECT_EQ(lifecycle.readToCheckpointUs(), 30);
}

TEST(RemoteRestoreLifecycleTest, LoadedThenCancelledDoesNotBecomeActualForwardUse) {
    RemoteRestoreLifecycle lifecycle;
    const auto             start = Clock::time_point{};
    ASSERT_TRUE(lifecycle.adopt({start, start + Us(10)}));
    EXPECT_TRUE(lifecycle.checkpointReadyUs().has_value());
    lifecycle.reset();
    EXPECT_FALSE(lifecycle.pendingForward());
    EXPECT_FALSE(lifecycle.completeContextForward(start + Us(100)).has_value());
    EXPECT_FALSE(lifecycle.checkpointReadyUs().has_value());
}

TEST(RemoteRestoreLifecycleTest, InvalidOrderOrMissingReadDoesNotFabricateLatency) {
    RemoteRestoreLifecycle lifecycle;
    const auto             start = Clock::time_point{};
    EXPECT_FALSE(lifecycle.checkpointReadyUs().has_value());
    EXPECT_FALSE(lifecycle.completeContextForward(start).has_value());
    EXPECT_FALSE(lifecycle.adopt({start + Us(1), start}));
    EXPECT_FALSE(lifecycle.pendingForward());
    ASSERT_TRUE(lifecycle.adopt({start, start + Us(10)}));
    EXPECT_FALSE(lifecycle.completeContextForward(start + Us(9)).has_value());
    EXPECT_TRUE(lifecycle.pendingForward());
    EXPECT_EQ(lifecycle.completeContextForward(start + Us(10)), 10);
}

TEST(RemoteRestoreLifecycleTest, RetryDoesNotReuseThePreviousAttemptClock) {
    RemoteRestoreLifecycle lifecycle;
    const auto             start = Clock::time_point{};
    ASSERT_TRUE(lifecycle.adopt({start, start + Us(10)}));
    lifecycle.reset();
    ASSERT_TRUE(lifecycle.adopt({start + Us(200), start + Us(250)}));
    EXPECT_EQ(lifecycle.checkpointReadyUs(), 50);
    EXPECT_EQ(lifecycle.completeContextForward(start + Us(300)), 100);
}
}  // namespace
}  // namespace rtp_llm
