#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsCollector.h"

namespace rtp_llm {
namespace {

TEST(RtpLLMCachePrepareMetricsTest, EmitsValuesAndStageTag) {
    auto monitor = std::make_shared<kmonitor::KMonitor>("cache_prepare_test", false);
    monitor->SetServiceName("");
    kmonitor::MetricsTags     base_tags("role", "prefill");
    kmonitor::MetricsReporter reporter(monitor, "", base_tags);

    kmonitor::MetricsTags                   stage_tags("stage", "coordinator_no_read_completion");
    RtpLLMCachePrepareStageMetricsCollector stage;
    stage.latency_us = 1234;
    ASSERT_TRUE((reporter.report<RtpLLMCachePrepareStageMetrics>(&stage_tags, &stage)));

    RtpLLMSchedulerCacheStallMetricsCollector exposed;
    exposed.cache_exposed_wait_us       = 35;
    exposed.cache_exposed_wait_total_us = 235;
    exposed.cache_exposed_wait_count    = 2;
    ASSERT_TRUE((reporter.report<RtpLLMSchedulerMetrics>(nullptr, &exposed)));

    RtpLLMEngineMetricsCollector engine;
    engine.step_latency_us     = 2000;
    engine.schedule_latency_us = 70;
    ASSERT_TRUE((reporter.report<RtpLLMEngineMetrics>(nullptr, &engine)));

    kmonitor::MetricsCollector collected;
    const auto                 now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    monitor->GetMetrics(&collected, {kmonitor::NORMAL}, now_ms);

    std::map<std::string, std::string> values;
    std::map<std::string, std::string> stages;
    std::map<std::string, std::string> roles;
    for (const auto* record : collected.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            values[value->Name()] = value->Value();
            stages[value->Name()] = record->Tags()->FindTag("stage");
            roles[value->Name()]  = record->Tags()->FindTag("role");
        }
    }
    auto gauge_sample = [&values](const std::string& name) {
        const auto& encoded = values.at(name);
        return encoded.substr(0, encoded.find('_'));
    };
    EXPECT_EQ(gauge_sample("rtp_llm_cache_prepare_stage_latency_us"), "1234");
    EXPECT_EQ(stages.at("rtp_llm_cache_prepare_stage_latency_us"), "coordinator_no_read_completion");
    EXPECT_EQ(roles.at("rtp_llm_cache_prepare_stage_latency_us"), "prefill");
    EXPECT_EQ(gauge_sample("rtp_llm_scheduler_cache_exposed_wait_us"), "35");
    EXPECT_EQ(gauge_sample("rtp_llm_scheduler_cache_exposed_wait_total_us"), "235");
    EXPECT_EQ(gauge_sample("rtp_llm_scheduler_cache_exposed_wait_count"), "2");
    EXPECT_EQ(gauge_sample("rtp_llm_schedule_latency_us"), "70");
    EXPECT_TRUE(values.count("rtp_llm_cache_prepare_stage_event_qps"));
}

}  // namespace
}  // namespace rtp_llm
