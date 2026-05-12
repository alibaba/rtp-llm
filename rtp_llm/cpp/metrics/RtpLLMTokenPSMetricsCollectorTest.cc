#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsLongPrefillByExecutionTime) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(256000, 256000, 0, 256000, 10 * 1000 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 25600.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 25600.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 256000.0, 1e-6);
    EXPECT_TRUE(collector.hasContextTPS());
    EXPECT_TRUE(collector.hasContextTPSWithCache());
    EXPECT_TRUE(collector.hasTotalTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergesShortPrefillsByExecutionTime) {
    RtpLLMTokenPSMetricsCollector collector;

    for (int i = 0; i < 10; ++i) {
        collector.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    }

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 10000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsContextTpsWithCacheIncludingReuseTokens) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsTimeWeightedTps) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    second.addTokenSize(9000, 9000, 0, 9000, 900 * 1000);
    merged.merge(&first);
    merged.merge(&second);

    EXPECT_NEAR(merged.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(merged.contextTPSWithCache(), 10000.0, 1e-6);
    EXPECT_NEAR(merged.totalTPS(), 10000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalAsTokenCounts) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 10, 1010, 100 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 10.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 1010.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalWhenExecutionTimeIsZero) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1000, 2, 1002, 0);

    EXPECT_FALSE(collector.hasContextTPS());
    EXPECT_FALSE(collector.hasContextTPSWithCache());
    EXPECT_TRUE(collector.hasGenerateTPS());
    EXPECT_TRUE(collector.hasTotalTPS());
    EXPECT_NEAR(collector.generateTPS(), 2.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 1002.0, 1e-6);
}

}  // namespace rtp_llm
