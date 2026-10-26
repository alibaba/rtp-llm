#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace rtp_llm {
namespace {

class RtpLLMEplbMetricsTest: public ::testing::Test {
protected:
    using MetricKey = std::tuple<std::string, std::string, std::string>;

    void SetUp() override {
        monitor_ = std::make_shared<kmonitor::KMonitor>("eplb_report_test");
        monitor_->SetServiceName("");
        reporter_ = std::make_shared<kmonitor::MetricsReporter>(monitor_, "", kmonitor::MetricsTags());
        ASSERT_NE(reporter_->getMetricsGroup<RtpLLmEplbMetrics>(), nullptr);
    }

    void report(RtpLLmEplbMetricsCollector& collector, const kmonitor::MetricsTags* tags = nullptr) {
        reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(tags, &collector);
    }

    void expectSnapshot(const std::map<MetricKey, std::vector<double>>& expected_samples) {
        kmonitor::MetricsCollector snapshot;
        monitor_->GetMetrics(&snapshot, {kmonitor::NORMAL}, 10000);
        std::map<MetricKey, std::string> actual;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                EXPECT_TRUE(record->Tags()->FindTag("ep_rank").empty());
                EXPECT_TRUE(record->Tags()->FindTag("layer").empty());
                MetricKey key{
                    value->Name(), record->Tags()->FindTag("model_id"), record->Tags()->FindTag("model_type")};
                EXPECT_TRUE(actual.emplace(key, value->Value()).second);
            }
        }
        ASSERT_EQ(actual.size(), expected_samples.size()) << ::testing::PrintToString(actual);
        for (const auto& [key, samples] : expected_samples) {
            SCOPED_TRACE(::testing::PrintToString(key));
            const auto metric = actual.find(key);
            ASSERT_NE(metric, actual.end());
            kmonitor::MinMaxCalculator expected;
            for (double sample : samples) {
                expected.Add(sample);
            }
            EXPECT_EQ(metric->second, expected.ToString());
        }
    }

    kmonitor::KMonitorPtr        monitor_;
    kmonitor::MetricsReporterPtr reporter_;
};

TEST_F(RtpLLMEplbMetricsTest, SkipsUnsetMetrics) {
    RtpLLmEplbMetricsCollector collector;
    report(collector);
    expectSnapshot({});
}

TEST_F(RtpLLMEplbMetricsTest, ReportsStepLatencyIncludingZeroWithoutLegacyMetrics) {
    RtpLLmEplbMetricsCollector collector;
    collector.step_latency_us = 0;
    report(collector);
    collector.step_latency_us = 37;
    report(collector);

    expectSnapshot({{MetricKey{"rtp_llm_eplb_step_latency_us", "", ""}, {0.0, 37.0}}});
}

TEST_F(RtpLLMEplbMetricsTest, ReportsModelIdentityWithoutLegacyMetrics) {
    RtpLLmEplbMetricsCollector target;
    target.moe_lb = 1.25;
    kmonitor::MetricsTags target_tags("model_id", "7");
    target_tags.AddTag("model_type", "target");
    report(target, &target_tags);

    RtpLLmEplbMetricsCollector draft;
    draft.moe_lb = 2.0;
    kmonitor::MetricsTags draft_tags("model_id", "8");
    draft_tags.AddTag("model_type", "draft");
    report(draft, &draft_tags);

    expectSnapshot(
        {{MetricKey{"rtp_llm_moe_lb", "7", "target"}, {1.25}}, {MetricKey{"rtp_llm_moe_lb", "8", "draft"}, {2.0}}});
}

}  // namespace
}  // namespace rtp_llm
