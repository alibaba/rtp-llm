#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>

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

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsPriorityMetrics) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(400, 600, 4, 404, 40 * 1000);
    first.addPriorityTokenSize(30, 400, 600, 4, 404, 40 * 1000);
    second.addTokenSize(600, 900, 6, 606, 60 * 1000);
    second.addPriorityTokenSize(50, 600, 900, 6, 606, 60 * 1000);
    merged.merge(&first);
    merged.merge(&second);

    auto priority_collectors = merged.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).contextTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextTPS(), 6000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).totalTPS(), 404.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).totalTPS(), 606.0, 1e-6);
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

TEST(RtpLLMTokenPSMetricsCollectorTest, MarksEmptyIdleWindowForZeroReport) {
    RtpLLMTokenPSMetricsCollector collector;

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());

    collector.markIdleWindow();

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_TRUE(collector.reportZeroTPS());
    EXPECT_NEAR(collector.contextTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 0.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 0.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, DoesNotMarkNonEmptyWindowAsIdle) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 2, 1002, 100 * 1000);
    collector.markIdleWindow();

    EXPECT_TRUE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsIdleZeroOnlyForEmptyMetrics) {
    RtpLLMTokenPSMetricsCollector idle;
    RtpLLMTokenPSMetricsCollector merged;

    idle.markIdleWindow();
    merged.merge(&idle);

    EXPECT_FALSE(merged.hasMetrics());
    EXPECT_TRUE(merged.reportZeroTPS());

    RtpLLMTokenPSMetricsCollector non_empty;
    non_empty.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    merged.merge(&non_empty);

    EXPECT_TRUE(merged.hasMetrics());
    EXPECT_FALSE(merged.reportZeroTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsContextWallTpsByReportWindow) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);
    collector.setReportWindowUs(200 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
    EXPECT_NEAR(collector.contextWallTPS(), 5000.0, 1e-6);
    EXPECT_NEAR(collector.contextWallTPSWithCache(), 7500.0, 1e-6);
    EXPECT_EQ(collector.reportWindowUs(), 200 * 1000);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallTpsUsesMergedTokens) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);
    second.addTokenSize(3000, 4500, 0, 3000, 100 * 1000);
    merged.merge(&first);
    merged.merge(&second);
    merged.setReportWindowUs(1 * 1000 * 1000);

    EXPECT_NEAR(merged.contextWallTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(merged.contextWallTPSWithCache(), 6000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGlobalAndPriorityMetrics) {
    RtpLLMTokenPSMetricsCollector collector;
    collector.addTokenSize(1000, 1500, 10, 1010, 100 * 1000);
    collector.addPriorityTokenSize(30, 400, 600, 4, 404, 100 * 1000);
    collector.addPriorityTokenSize(50, 600, 900, 6, 606, 100 * 1000);
    collector.setReportWindowUs(200 * 1000);

    auto priority_collectors = collector.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).contextTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextTPS(), 6000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).contextWallTPS(), 2000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextWallTPS(), 3000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS(), 4.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).generateTPS(), 6.0, 1e-6);

    EXPECT_NEAR(priority_collectors.at(30).contextTPS() + priority_collectors.at(50).contextTPS(),
                collector.contextTPS(),
                1e-6);
    EXPECT_NEAR(priority_collectors.at(30).contextWallTPS() + priority_collectors.at(50).contextWallTPS(),
                collector.contextWallTPS(),
                1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS() + priority_collectors.at(50).generateTPS(),
                collector.generateTPS(),
                1e-6);

    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter(nullptr);
    tps_reporter.report(&collector);

    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter(
        nullptr);
    wall_tps_reporter.report(&collector);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, AddTokenSizeByPriorityMatchesUntaggedTotals) {
    struct TokenCounts {
        int64_t context            = 0;
        int64_t context_with_cache = 0;
        int64_t generate           = 0;
        int64_t total              = 0;
    };
    // Mirrors the MTP decode path: accepted tokens bucketed per priority with
    // generate == total and no context contribution; bucket sums must equal
    // the untagged totals reported through addTokenSize.
    std::map<int32_t, TokenCounts> counts_by_priority;
    counts_by_priority[30] = {0, 0, 7, 7};
    counts_by_priority[50] = {0, 0, 3, 3};

    RtpLLMTokenPSMetricsCollector collector;
    const int64_t                 total_accepted = 10;
    collector.addTokenSize(0, 0, total_accepted, total_accepted, 100 * 1000);
    collector.addTokenSizeByPriority(counts_by_priority, 100 * 1000);

    auto priority_collectors = collector.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS(), 7.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).generateTPS(), 3.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS() + priority_collectors.at(50).generateTPS(),
                collector.generateTPS(),
                1e-6);
    EXPECT_NEAR(
        priority_collectors.at(30).totalTPS() + priority_collectors.at(50).totalTPS(), collector.totalTPS(), 1e-6);
    EXPECT_FALSE(priority_collectors.at(30).hasContextTPS());
    EXPECT_FALSE(priority_collectors.at(50).hasContextTPS());
}

}  // namespace rtp_llm
