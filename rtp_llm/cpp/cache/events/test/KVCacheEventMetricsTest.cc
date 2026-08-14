#include "rtp_llm/cpp/cache/events/KVCacheEventMetrics.h"

#include <gtest/gtest.h>
#include <limits>

namespace rtp_llm::test {

TEST(KVCacheEventMetricsTest, MapsDistinctPublisherStatusValuesWithoutReordering) {
    PublisherStatus status;
    status.state                   = PublisherState::CIRCUIT_OPEN;
    status.queue_size              = 1;
    status.queue_high_watermark    = 2;
    status.accepted_count          = 3;
    status.dropped_count           = 4;
    status.request_failure_count   = 5;
    status.overflow_recovery_count = 6;
    status.snapshot_attempt_count  = 7;
    status.snapshot_commit_count   = 8;

    const auto collector = makeKVCacheEventMetricsCollector(status, /*cache_insert_rejected_count=*/9);
    EXPECT_EQ(8, collector.publisher_state);
    EXPECT_EQ(1, collector.queue_size);
    EXPECT_EQ(2, collector.queue_high_watermark);
    EXPECT_EQ(3, collector.accepted_count);
    EXPECT_EQ(4, collector.dropped_count);
    EXPECT_EQ(5, collector.request_failure_count);
    EXPECT_EQ(6, collector.overflow_recovery_count);
    EXPECT_EQ(7, collector.snapshot_attempt_count);
    EXPECT_EQ(8, collector.snapshot_commit_count);
    EXPECT_EQ(9, collector.cache_insert_rejected_count);
    EXPECT_FALSE(collector.suppress_publisher_details);
}

TEST(KVCacheEventMetricsTest, DisabledStateSuppressesOnlyInactiveDetailSeries) {
    const auto collector = makeKVCacheEventMetricsCollector(PublisherStatus{});

    EXPECT_EQ(0, collector.publisher_state);
    EXPECT_EQ(0, collector.cache_insert_rejected_count);
    EXPECT_TRUE(collector.suppress_publisher_details);
}

TEST(KVCacheEventMetricsTest, GatedStateRetainsDiagnosticDetailSeries) {
    PublisherStatus status;
    status.state         = PublisherState::GATED;
    status.dropped_count = 1;

    const auto collector = makeKVCacheEventMetricsCollector(status);
    EXPECT_EQ(9, collector.publisher_state);
    EXPECT_EQ(1, collector.dropped_count);
    EXPECT_FALSE(collector.suppress_publisher_details);
}

TEST(KVCacheEventMetricsTest, SaturatesCountersAtSignedMetricLimit) {
    PublisherStatus status;
    status.state                   = PublisherState::READY;
    status.queue_size              = std::numeric_limits<size_t>::max();
    status.queue_high_watermark    = std::numeric_limits<size_t>::max();
    status.accepted_count          = std::numeric_limits<uint64_t>::max();
    status.dropped_count           = std::numeric_limits<uint64_t>::max();
    status.request_failure_count   = std::numeric_limits<uint64_t>::max();
    status.overflow_recovery_count = std::numeric_limits<uint64_t>::max();
    status.snapshot_attempt_count  = std::numeric_limits<uint64_t>::max();
    status.snapshot_commit_count   = std::numeric_limits<uint64_t>::max();

    const auto collector = makeKVCacheEventMetricsCollector(status, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.queue_size);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.queue_high_watermark);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.accepted_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.dropped_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.request_failure_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.overflow_recovery_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.snapshot_attempt_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.snapshot_commit_count);
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), collector.cache_insert_rejected_count);
}

}  // namespace rtp_llm::test
