#include "gtest/gtest.h"

#include <unistd.h>

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

TEST_F(CacheStoreMetricsCollectorTest, testServerLoadMetrics) {
    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 1024, 123);
    collector.reset();

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    collector      = std::make_shared<CacheStoreServerLoadMetricsCollector>(reporter, 1, 1024, 123);
    usleep(10);
    collector->markFirstBlockReady();
    usleep(10);
    collector->markAllBlocksReady();
    collector->setWriteInfo(1, 111, 1111);
    usleep(10);
    collector->markEnd(true);

    collector.reset();
}

}  // namespace rtp_llm
