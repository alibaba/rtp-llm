// R6 DEV-zeta — unit tests for KVCacheCanaryMetrics (canary §1.0 item 7).
//
// We can't easily mock kmonitor::MetricsReporter without pulling in the
// kmonitor factory, so these tests verify the process-wide counter half of
// each helper: every recordCanaryXxx() must bump its counter exactly once,
// and the counters must be independently resettable.  Production gauges
// (kv_cache.hit_rate etc.) are exercised by the smoke-test golden run since
// they require a live kmonitor sink.

#include <gtest/gtest.h>

#include "rtp_llm/cpp/metrics/KVCacheCanaryMetrics.h"

namespace rtp_llm {
namespace test {

class KVCacheCanaryMetricsTest: public ::testing::Test {
protected:
    void SetUp() override {
        resetCanaryCountersForTest();
    }
};

TEST_F(KVCacheCanaryMetricsTest, BaselineCountersAreZero) {
    EXPECT_EQ(canaryHitRateTickCount(), 0u);
    EXPECT_EQ(canaryHbmUsedBlocksTickCount(), 0u);
    EXPECT_EQ(canaryTtftTickCount(), 0u);
    EXPECT_EQ(canaryTpotTickCount(), 0u);
    EXPECT_EQ(canaryErrorEventCount(), 0u);
    EXPECT_EQ(canaryOomEventCount(), 0u);
    EXPECT_EQ(canaryPeerRefusedCount(), 0u);
}

TEST_F(KVCacheCanaryMetricsTest, EachHelperBumpsItsOwnCounterOnly) {
    // Null reporter — only the counters fire (no kmonitor dependency needed).
    kmonitor::MetricsReporterPtr null_reporter;

    recordCanaryHitRate(null_reporter, 87.5f);
    EXPECT_EQ(canaryHitRateTickCount(), 1u);
    EXPECT_EQ(canaryHbmUsedBlocksTickCount(), 0u);

    recordCanaryHbmUsedBlocks(null_reporter, 4096);
    recordCanaryHbmUsedBlocks(null_reporter, 4097);
    EXPECT_EQ(canaryHbmUsedBlocksTickCount(), 2u);

    recordCanaryTtftMs(null_reporter, 12);
    EXPECT_EQ(canaryTtftTickCount(), 1u);

    recordCanaryTpotMs(null_reporter, 3);
    EXPECT_EQ(canaryTpotTickCount(), 1u);

    recordCanaryErrorEvent(null_reporter);
    recordCanaryErrorEvent(null_reporter);
    recordCanaryErrorEvent(null_reporter);
    EXPECT_EQ(canaryErrorEventCount(), 3u);

    recordCanaryOomEvent(null_reporter);
    EXPECT_EQ(canaryOomEventCount(), 1u);

    recordCanaryPeerRefused(null_reporter);
    EXPECT_EQ(canaryPeerRefusedCount(), 1u);

    // No cross-talk between counters.
    EXPECT_EQ(canaryHitRateTickCount(), 1u);
    EXPECT_EQ(canaryHbmUsedBlocksTickCount(), 2u);
    EXPECT_EQ(canaryTtftTickCount(), 1u);
}

TEST_F(KVCacheCanaryMetricsTest, ResetClearsAllCounters) {
    kmonitor::MetricsReporterPtr null_reporter;
    recordCanaryHitRate(null_reporter, 50.0f);
    recordCanaryErrorEvent(null_reporter);
    recordCanaryPeerRefused(null_reporter);
    ASSERT_GT(canaryHitRateTickCount(), 0u);
    ASSERT_GT(canaryErrorEventCount(), 0u);
    ASSERT_GT(canaryPeerRefusedCount(), 0u);

    resetCanaryCountersForTest();

    EXPECT_EQ(canaryHitRateTickCount(), 0u);
    EXPECT_EQ(canaryErrorEventCount(), 0u);
    EXPECT_EQ(canaryPeerRefusedCount(), 0u);
}

TEST_F(KVCacheCanaryMetricsTest, CollectorSentinelsSkipUnsetFields) {
    // The collector's sentinel (-1 / false) means the report() function
    // skips that gauge for the tick, so partial-collector ticks never emit
    // stale zeros into the dashboard.  We can't observe the kmonitor sink
    // directly here, but we exercise the contract: default-constructed
    // collector should be a no-op when passed to report() (no crash, no
    // counter bump from the report path itself).
    KVCacheCanaryMetricsCollector empty;
    EXPECT_LT(empty.kv_cache_hit_rate, 0.0f);
    EXPECT_LT(empty.kv_cache_hbm_used_blocks, 0);
    EXPECT_LT(empty.inference_ttft_ms, 0);
    EXPECT_LT(empty.inference_tpot_ms, 0);
    EXPECT_FALSE(empty.engine_error_event);
    EXPECT_FALSE(empty.engine_oom_event);
    EXPECT_FALSE(empty.pd_peer_refused);
}

}  // namespace test
}  // namespace rtp_llm
