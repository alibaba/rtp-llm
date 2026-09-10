#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include <gtest/gtest.h>
#include <map>
#include <string>

#include "kmonitor/client/core/MetricsCollector.h"
#include "kmonitor/client/core/MetricsConfig.h"

namespace rtp_llm {
namespace {

TEST(RtpLLMCacheScheduleMetricsTest, NativeKmonitorOnlyReceivesPositiveSamples) {
    kmonitor::MetricsConfig config;
    auto                    monitor = std::make_shared<kmonitor::KMonitor>("cache_schedule_test");
    monitor->SetConfig(&config);
    monitor->SetServiceName("");
    kmonitor::MetricsTags     identity(std::map<std::string, std::string>{{"server_id", "3"}, {"rank_id", "2"}});
    kmonitor::MetricsReporter reporter(monitor, "", identity);
    kmonitor::MetricsTags     timeout("timeout", "true");
    for (int value : {-2, -1, 0, 10}) {
        RtpLLMStreamMetricsCollector collector;
        collector.not_streaming_qps     = false;
        collector.cache_schedule_target = true;
        // These legacy values must not leak into the classified series.
        collector.enqueue_to_canrun_us     = 999;
        collector.canrun_to_running_us     = 999;
        collector.loading_cache_latency_us = 999;
        collector.load_done_to_running_us  = 999;
        CacheScheduleSnapshot sample;
        sample.has_async_cache_dependency = true;
        sample.enqueue_to_canrun_us       = value;
        sample.canrun_to_running_us       = value;
        sample.loading_latency_us         = value;
        sample.load_done_to_running_us    = value;
        sample.ready_wait_us              = value;
        sample.schedule_rounds            = value == 0 ? 1 : 3;
        // Also exercise absent/empty target snapshots without leaking legacy values.
        if (value == -1) {
            collector.cache_schedule = CacheScheduleSnapshot{};
        } else if (value >= 0) {
            collector.cache_schedule = sample;
        }
        reporter.report<RtpLLMStreamMetrics>(&timeout, &collector);
    }
    kmonitor::MetricsCollector output;
    monitor->GetMetrics(&output, {kmonitor::NORMAL}, 1000);
    std::map<std::string, std::string> values;
    for (const auto* record : output.GetRecords().getRecords()) {
        if (record->Values().empty()) {
            continue;
        }
        EXPECT_EQ(record->Tags()->FindTag("server_id"), "3");
        EXPECT_EQ(record->Tags()->FindTag("rank_id"), "2");
        EXPECT_EQ(record->Tags()->FindTag("timeout"), "true");
        EXPECT_TRUE(record->Tags()->FindTag("cache_probe").empty());
        EXPECT_TRUE(record->Tags()->FindTag("sample_status").empty());
        EXPECT_EQ(record->Tags()->FindTag("cache_dependency"), "cache");
        EXPECT_TRUE(record->Tags()->FindTag("loading_entered").empty());
        for (const auto* value : record->Values()) {
            values[value->Name()] = value->Value();
        }
    }
    ASSERT_EQ(values.size(), 6);
    for (const auto* name : {"rtp_llm_stream_enqueue_to_canrun_us",
                             "rtp_llm_stream_canrun_to_running_us",
                             "rtp_llm_stream_loading_cache_latency_us",
                             "rtp_llm_stream_load_done_to_running_us",
                             "rtp_llm_stream_loading_cache_ready_wait_us"}) {
        // Native gauge serialization: average_sum_max_min_count.
        EXPECT_EQ(values[name], "10_10_10_10_1") << name;
    }
    EXPECT_EQ(values["rtp_llm_stream_canrun_to_running_schedule_rounds"], "2_4_3_1_2");
}

TEST(RtpLLMCacheScheduleMetricsTest, NonTargetStreamRetainsLegacyZeroFiltering) {
    kmonitor::MetricsConfig config;
    auto                    monitor = std::make_shared<kmonitor::KMonitor>("cache_schedule_non_target_test");
    monitor->SetConfig(&config);
    monitor->SetServiceName("");
    kmonitor::MetricsReporter reporter(monitor, "", kmonitor::MetricsTags());
    for (int value : {0, 10}) {
        RtpLLMStreamMetricsCollector collector;
        collector.not_streaming_qps    = false;
        collector.enqueue_to_canrun_us = value;
        reporter.report<RtpLLMStreamMetrics>(nullptr, &collector);
    }
    kmonitor::MetricsCollector output;
    monitor->GetMetrics(&output, {kmonitor::NORMAL}, 1000);
    size_t count = 0;
    for (const auto* record : output.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            ++count;
            EXPECT_EQ(value->Name(), "rtp_llm_stream_enqueue_to_canrun_us");
            EXPECT_EQ(value->Value(), "10_10_10_10_1");
            EXPECT_TRUE(record->Tags()->FindTag("cache_probe").empty());
        }
    }
    EXPECT_EQ(count, 1);
}
}  // namespace
}  // namespace rtp_llm
