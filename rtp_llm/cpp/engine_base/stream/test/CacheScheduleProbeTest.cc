#include "rtp_llm/cpp/engine_base/stream/CacheScheduleProbe.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
const SchedulerRoundContext first{42, 7};
const SchedulerRoundContext last{42, 9};

void direct(CacheScheduleProbe& probe, const SchedulerRoundContext* end = &first) {
    probe.activate(42);
    probe.enqueue(100);
    probe.canRun(100, &first);
    probe.beginAttempt({true}, true);
    probe.running(100, end);
}

TEST(CacheScheduleProbeTest, ExactBoundariesAndZeroValues) {
    CacheScheduleProbe probe;
    probe.activate(42);
    probe.enqueue(100);
    probe.canRun(110, &first);
    probe.beginAttempt({true, true}, true);
    probe.loadingStart(120);
    probe.observe(CacheLoadTerminalSnapshot{{true, true}, true, true, 150}, 170);
    probe.loadingDone(180);
    probe.canRun(190, &last);  // clearCanRun/re-admission must not overwrite Tc or Rc.
    probe.running(200, &last);
    auto sample = probe.takeReport();
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->sample_status, CacheSampleStatus::SINGLE_PASS);
    EXPECT_EQ(sample->dependency, CacheDependency::DATA);
    EXPECT_EQ(sample->enqueue_to_canrun_us, 10);
    EXPECT_EQ(sample->canrun_to_running_us, 90);
    EXPECT_EQ(sample->loading_latency_us, 60);
    EXPECT_EQ(sample->load_done_to_running_us, 20);
    EXPECT_EQ(sample->ready_wait_us, 20);
    EXPECT_EQ(sample->schedule_rounds, 3);
    EXPECT_FALSE(probe.takeReport());

    CacheScheduleProbe zero;
    direct(zero);
    sample = zero.takeReport();
    ASSERT_TRUE(sample->numeric_valid);
    EXPECT_FALSE(sample->loading_entered);
    EXPECT_EQ(sample->dependency, CacheDependency::NONE);
    EXPECT_EQ(sample->enqueue_to_canrun_us, 0);
    EXPECT_EQ(sample->canrun_to_running_us, 0);
    EXPECT_EQ(sample->loading_latency_us, 0);
    EXPECT_EQ(sample->load_done_to_running_us, 0);
    EXPECT_EQ(sample->ready_wait_us, 0);
    EXPECT_EQ(sample->schedule_rounds, 1);
}

TEST(CacheScheduleProbeTest, ContextCanCompleteBeforeLoadingStarts) {
    CacheScheduleProbe probe;
    probe.activate(42);
    probe.enqueue(100);
    probe.canRun(110, &first);
    probe.beginAttempt({false, false, true}, true);
    probe.loadingStart(120);
    probe.observe(CacheLoadTerminalSnapshot{{true, false, true}, true, true, 115}, 170);
    probe.loadingDone(180);
    probe.running(200, &last);
    auto sample = probe.takeReport();
    ASSERT_TRUE(sample->numeric_valid);
    EXPECT_EQ(sample->dependency, CacheDependency::LOOKUP_ONLY);
    EXPECT_EQ(sample->ready_wait_us, 50);
}

TEST(CacheScheduleProbeTest, RetryKeepsDataEvidenceAndOnlyCounts) {
    CacheScheduleProbe probe;
    probe.activate(42);
    probe.enqueue(100);
    probe.canRun(110, &first);
    probe.beginAttempt({false, true, false, true}, false);
    probe.beginAttempt({true}, true);
    probe.running(200, &last);
    auto sample = probe.takeReport();
    EXPECT_EQ(sample->dependency, CacheDependency::DATA);
    EXPECT_EQ(sample->sample_status, CacheSampleStatus::RECOVERED);
    EXPECT_FALSE(sample->numeric_valid);
}

TEST(CacheScheduleProbeTest, FailedPrefillLoadFallbackIsRecovered) {
    CacheScheduleProbe probe;
    probe.activate(42);
    probe.beginAttempt({true, true}, true);
    probe.observe(CacheLoadTerminalSnapshot{{true, true, false, true}, true, false, 150}, 170);
    probe.running(200, &last);
    EXPECT_EQ(probe.takeReport()->sample_status, CacheSampleStatus::RECOVERED);
}

TEST(CacheScheduleProbeTest, CancellationRetainsUnknownOrKnownData) {
    for (bool data : {false, true}) {
        CacheScheduleProbe probe;
        probe.activate(42);
        probe.beginAttempt({false, data, true}, true);
        auto sample = probe.takeReport();
        EXPECT_EQ(sample->dependency, data ? CacheDependency::DATA : CacheDependency::UNKNOWN);
        EXPECT_EQ(sample->sample_status, CacheSampleStatus::EXCLUDED);
        EXPECT_FALSE(sample->numeric_valid);
    }
}

TEST(CacheScheduleProbeTest, FrozenSnapshotIgnoresDecodeErrorsAndReset) {
    CacheScheduleProbe probe;
    direct(probe);
    probe.beginAttempt({false, true, true, true}, false);
    probe.reset();
    probe.running(500, nullptr);
    auto sample = probe.takeReport();
    EXPECT_TRUE(sample->numeric_valid);
    EXPECT_EQ(sample->dependency, CacheDependency::NONE);
    probe.reset();
    probe.activate(43);
    EXPECT_FALSE(probe.takeReport());
}

TEST(CacheScheduleProbeTest, ResetBeforeRunningCannotCreateAnotherSample) {
    CacheScheduleProbe probe;
    probe.activate(42);
    probe.beginAttempt({true}, false);
    probe.reset();
    direct(probe);
    EXPECT_EQ(probe.takeReport()->sample_status, CacheSampleStatus::EXCLUDED);
    EXPECT_FALSE(probe.takeReport());
}

TEST(CacheScheduleProbeTest, CopiesAndInactiveObjectsDoNotOwnReports) {
    CacheScheduleProbe owner;
    direct(owner);
    CacheScheduleProbe copy(owner);
    CacheScheduleProbe assigned;
    assigned = owner;
    EXPECT_FALSE(copy.active());
    EXPECT_FALSE(copy.takeReport());
    EXPECT_FALSE(assigned.takeReport());
    EXPECT_TRUE(owner.takeReport());
}

TEST(CacheScheduleProbeTest, MissingForeignOrReversedRoundsAreExcluded) {
    const SchedulerRoundContext foreign{43, 9}, reversed{42, 6};
    for (auto round : {static_cast<const SchedulerRoundContext*>(nullptr), &foreign, &reversed}) {
        CacheScheduleProbe probe;
        direct(probe, round);
        EXPECT_EQ(probe.takeReport()->sample_status, CacheSampleStatus::EXCLUDED);
    }
    CacheScheduleProbe missing_canrun;
    missing_canrun.activate(42);
    missing_canrun.enqueue(100);
    missing_canrun.canRun(100, nullptr);
    missing_canrun.canRun(100, &first);
    missing_canrun.beginAttempt({true}, true);
    missing_canrun.running(100, &first);
    EXPECT_FALSE(missing_canrun.takeReport()->numeric_valid);
}

TEST(CacheScheduleProbeTest, ClockRegressionAndUnsupportedTerminalNeverBecomeZeros) {
    for (int mode = 0; mode < 3; ++mode) {
        CacheScheduleProbe probe;
        probe.activate(42);
        probe.enqueue(100);
        probe.canRun(mode == 0 ? 99 : 110, &first);
        probe.beginAttempt({true, true}, true);
        probe.loadingStart(120);
        if (mode == 1) {
            probe.observe(std::nullopt, 170);
        } else {
            probe.observe(CacheLoadTerminalSnapshot{{true, true}, true, true, mode == 2 ? 171 : 150}, 170);
        }
        probe.loadingDone(180);
        probe.running(200, &last);
        EXPECT_FALSE(probe.takeReport()->numeric_valid);
    }
}
}  // namespace
}  // namespace rtp_llm
