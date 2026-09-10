#include "rtp_llm/cpp/engine_base/stream/CacheScheduleMetrics.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
const SchedulerRoundContext first{42, 7};
const SchedulerRoundContext last{42, 9};

void prepare(CacheScheduleMetrics& metrics) {
    metrics.activate(42);
    metrics.enqueue(100);
    metrics.canRun(110, &first);
}

void load(CacheScheduleMetrics& metrics, int64_t ready = 150) {
    metrics.beginAttempt(true, true);
    metrics.loadingStart(120);
    metrics.observe(CacheLoadTerminalSnapshot{true, true, ready}, 170);
    metrics.loadingDone(180);
}

TEST(CacheScheduleMetricsTest, ExactBoundariesAndFreeze) {
    for (int64_t ready : {115, 150}) {  // Also cover completion before loading starts.
        CacheScheduleMetrics metrics;
        prepare(metrics);
        load(metrics, ready);
        metrics.canRun(190, &last);
        metrics.running(200, &last);
        metrics.beginAttempt(false, false);
        metrics.reset();
        metrics.running(500, nullptr);
        auto sample = metrics.takeReport();
        ASSERT_TRUE(sample);
        EXPECT_TRUE(sample->has_async_cache_dependency);
        EXPECT_EQ(sample->enqueue_to_canrun_us, 10);
        EXPECT_EQ(sample->canrun_to_running_us, 90);
        EXPECT_EQ(sample->loading_latency_us, 60);
        EXPECT_EQ(sample->load_done_to_running_us, 20);
        EXPECT_EQ(sample->ready_wait_us, ready == 115 ? 50 : 20);
        EXPECT_EQ(sample->schedule_rounds, 3);
        EXPECT_FALSE(metrics.takeReport());
    }
}

TEST(CacheScheduleMetricsTest, DirectAndSameRoundHaveNoLoadingValues) {
    CacheScheduleMetrics metrics;
    metrics.activate(42);
    metrics.enqueue(100);
    metrics.canRun(100, &first);
    metrics.beginAttempt(false, true);
    metrics.running(100, &first);
    auto sample = metrics.takeReport();
    EXPECT_FALSE(sample->has_async_cache_dependency);
    EXPECT_EQ(sample->enqueue_to_canrun_us, 0);
    EXPECT_EQ(sample->canrun_to_running_us, 0);
    EXPECT_EQ(sample->loading_latency_us, 0);
    EXPECT_EQ(sample->load_done_to_running_us, 0);
    EXPECT_EQ(sample->ready_wait_us, 0);
    EXPECT_EQ(sample->schedule_rounds, 1);
}

TEST(CacheScheduleMetricsTest, RetryAndResetPreserveFirstBoundariesAndDependency) {
    for (bool reset : {false, true}) {
        CacheScheduleMetrics metrics;
        prepare(metrics);
        load(metrics);
        if (reset) {
            metrics.reset();
            metrics.enqueue(185);
            metrics.canRun(190, &last);
        }
        metrics.beginAttempt(false, true);
        metrics.running(200, &last);
        auto sample = metrics.takeReport();
        EXPECT_TRUE(sample->has_async_cache_dependency);
        EXPECT_EQ(sample->enqueue_to_canrun_us, 10);
        EXPECT_EQ(sample->canrun_to_running_us, 90);
        EXPECT_EQ(sample->schedule_rounds, 3);
        EXPECT_EQ(sample->loading_latency_us, 0);
        EXPECT_EQ(sample->load_done_to_running_us, 0);
        EXPECT_EQ(sample->ready_wait_us, 0);
        EXPECT_FALSE(metrics.takeReport());
    }
}

TEST(CacheScheduleMetricsTest, FailedOrUnsupportedContextOnlyInvalidatesLoading) {
    for (bool supported : {false, true}) {
        CacheScheduleMetrics metrics;
        prepare(metrics);
        metrics.beginAttempt(true, true);
        metrics.loadingStart(120);
        metrics.observe(supported ? std::make_optional(CacheLoadTerminalSnapshot{true, false, 150}) : std::nullopt,
                        170);
        metrics.loadingDone(180);
        metrics.running(200, &last);
        auto sample = metrics.takeReport();
        EXPECT_EQ(sample->canrun_to_running_us, 90);
        EXPECT_EQ(sample->schedule_rounds, 3);
        EXPECT_EQ(sample->loading_latency_us, 0);
        EXPECT_EQ(sample->load_done_to_running_us, 0);
        EXPECT_EQ(sample->ready_wait_us, 0);
    }
}

TEST(CacheScheduleMetricsTest, CancellationReportsCompletedIntervalsWithoutInventingRunning) {
    CacheScheduleMetrics metrics;
    prepare(metrics);
    load(metrics);
    auto sample = metrics.takeReport();
    EXPECT_EQ(sample->enqueue_to_canrun_us, 10);
    EXPECT_EQ(sample->loading_latency_us, 60);
    EXPECT_EQ(sample->ready_wait_us, 20);
    EXPECT_EQ(sample->canrun_to_running_us, 0);
    EXPECT_EQ(sample->load_done_to_running_us, 0);
    EXPECT_EQ(sample->schedule_rounds, 0);
}

TEST(CacheScheduleMetricsTest, MissingForeignReversedRoundsOnlySuppressRounds) {
    const SchedulerRoundContext foreign{43, 9}, reversed{42, 6}, zero{42, 0};
    for (auto round : {static_cast<const SchedulerRoundContext*>(nullptr), &foreign, &reversed, &zero}) {
        CacheScheduleMetrics metrics;
        prepare(metrics);
        metrics.running(200, round);
        auto sample = metrics.takeReport();
        EXPECT_EQ(sample->schedule_rounds, 0);
        EXPECT_EQ(sample->enqueue_to_canrun_us, 10);
        EXPECT_EQ(sample->canrun_to_running_us, 90);
    }
}

TEST(CacheScheduleMetricsTest, InvalidOrMissingTimesOnlySuppressDependentValues) {
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode);
        CacheScheduleMetrics metrics;
        metrics.activate(42);
        metrics.enqueue(mode == 0 ? 120 : 100);  // Reversed enqueue/CanRun.
        metrics.canRun(110, &first);
        metrics.beginAttempt(true, true);
        if (mode != 1) {  // Missing loading start.
            metrics.loadingStart(120);
        }
        metrics.observe(CacheLoadTerminalSnapshot{true, true, mode == 3 ? 171 : 150}, 170);
        if (mode != 2) {  // Missing loading done.
            metrics.loadingDone(180);
        }
        metrics.running(200, &last);
        const auto sample = metrics.takeReport();
        EXPECT_EQ(sample->enqueue_to_canrun_us, mode == 0 ? 0 : 10);
        EXPECT_EQ(sample->canrun_to_running_us, 90);
        EXPECT_EQ(sample->loading_latency_us, mode == 1 || mode == 2 ? 0 : 60);
        EXPECT_EQ(sample->load_done_to_running_us, mode == 2 ? 0 : 20);
        EXPECT_EQ(sample->ready_wait_us, mode == 1 || mode == 3 ? 0 : 20);
    }
}

TEST(CacheScheduleMetricsTest, CopiesAndInactiveObjectsDoNotOwnReports) {
    CacheScheduleMetrics owner;
    prepare(owner);
    CacheScheduleMetrics copy(owner), assigned;
    assigned = owner;
    EXPECT_FALSE(copy.active());
    EXPECT_FALSE(copy.takeReport());
    EXPECT_FALSE(assigned.takeReport());
    EXPECT_TRUE(owner.takeReport());
}
}  // namespace
}  // namespace rtp_llm
