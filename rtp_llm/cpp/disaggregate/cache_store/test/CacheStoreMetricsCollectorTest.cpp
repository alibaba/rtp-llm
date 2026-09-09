#include "gtest/gtest.h"

#include <unistd.h>
#include <vector>

#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

namespace rtp_llm {

class CacheStoreMetricsCollectorTest: public CacheStoreTestBase {};

TEST_F(CacheStoreMetricsCollectorTest, testStoreMetrics) {
    auto collector = std::make_shared<CacheStoreStoreMetricsCollector>(nullptr, 1, 1024);
    collector.reset();

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    collector      = std::make_shared<CacheStoreStoreMetricsCollector>(reporter, 1, 1024);

    usleep(10);
    collector->markTaskRun();
    usleep(10);
    collector->markEventSyncDone();
    usleep(10);
    collector->markEnd(true);

    collector.reset();
}

TEST_F(CacheStoreMetricsCollectorTest, testClientLoadMetrics) {
    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1024);
    collector.reset();

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    collector      = std::make_shared<CacheStoreClientLoadMetricsCollector>(reporter, 1, 1024);

    usleep(10);
    collector->markTaskRun();
    usleep(10);
    collector->markRequestCallBegin();
    usleep(10);
    collector->markRequestCallEnd(1111);
    usleep(10);
    collector->markEnd(true);

    collector.reset();
}

TEST_F(CacheStoreMetricsCollectorTest, testServerLoadMetricsReportsCollectedValues) {
    RtpLLMCacheStoreLoadServerMetricsCollector reported_metrics;
    size_t                                     report_count = 0;
    auto                                       collector =
        std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 2, 2048, 123, [&](const auto& metrics) {
            reported_metrics = metrics;
            ++report_count;
        });

    usleep(1000);
    collector->markAllBlocksReady();
    collector->setWriteInfo(1, 111, 1111);
    collector->setWriteInfo(2, 222, 2222);
    usleep(1000);
    collector->markEnd(false);

    collector.reset();

    EXPECT_EQ(1, report_count);
    EXPECT_FALSE(reported_metrics.success);
    EXPECT_EQ(2, reported_metrics.block_count);
    EXPECT_EQ(2048, reported_metrics.total_block_size);
    EXPECT_EQ(123, reported_metrics.request_send_cost_us);
    EXPECT_GT(reported_metrics.latency_us, 0);
    EXPECT_GT(reported_metrics.all_block_ready_latency_us, 0);
    EXPECT_GT(reported_metrics.transfer_gap_latency_us, 0);
    EXPECT_EQ((std::vector<int64_t>{1, 2}), reported_metrics.write_block_count);
    EXPECT_EQ((std::vector<int64_t>{111, 222}), reported_metrics.write_total_block_size);
    EXPECT_EQ((std::vector<int64_t>{1111, 2222}), reported_metrics.write_latency_us);
}

TEST_F(CacheStoreMetricsCollectorTest, testServerLoadMetricsWithoutAllBlocksReady) {
    RtpLLMCacheStoreLoadServerMetricsCollector reported_metrics;
    size_t                                     report_count = 0;
    auto                                       collector =
        std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 1024, 123, [&](const auto& metrics) {
            reported_metrics = metrics;
            ++report_count;
        });

    usleep(1000);
    collector->markEnd(true);
    collector.reset();

    EXPECT_EQ(1, report_count);
    EXPECT_TRUE(reported_metrics.success);
    EXPECT_GT(reported_metrics.latency_us, 0);
    EXPECT_EQ(0, reported_metrics.all_block_ready_latency_us);
    EXPECT_EQ(0, reported_metrics.transfer_gap_latency_us);
}

TEST_F(CacheStoreMetricsCollectorTest, testServerLoadMetricsReportsThroughMetricsReporter) {
    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(reporter, 1, 1024, 123);

    collector->markAllBlocksReady();
    collector->setWriteInfo(1, 1024, 1111);
    collector->markEnd(true);

    EXPECT_NO_THROW(collector.reset());
}

}  // namespace rtp_llm
