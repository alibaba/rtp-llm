#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/metrics/ServiceStatus.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>
#include <set>

namespace rtp_llm {

TEST(ServiceStatusTest, WallTpsKeepsRankAndSeparatesServingAndDrainSamples) {
    auto monitor = std::make_shared<kmonitor::KMonitor>("serving_test");
    monitor->SetServiceName("");
    kmonitor::MetricsReporter reporter(monitor, "", kmonitor::MetricsTags("dp_rank", "2"));
    kmonitor::MetricsTags     tags("priority", "7");
    tags.AddTag("is_serving", "true");

    RtpLLMTokenPSMetricsCollector active;
    active.addTokenSize(10, 10, 0, 10, 1000);
    active.setReportWindowUs(1000);
    reporter.report<RtpLLMWallClockTokenPSMetrics>(&tags, &active);
    markKmonServiceDraining();
    RtpLLMTokenPSMetricsCollector idle;
    idle.markIdleWindow();
    reporter.report<RtpLLMWallClockTokenPSMetrics>(&tags, &idle);

    kmonitor::MetricsCollector snapshot;
    monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, 10000);
    std::set<std::string> states;
    for (const auto* record : snapshot.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            if (value->Name() == "rtp_llm_context_wall_tps") {
                EXPECT_EQ(record->Tags()->FindTag("dp_rank"), "2");
                EXPECT_EQ(record->Tags()->FindTag("priority"), "7");
                states.insert(record->Tags()->FindTag("is_serving"));
            }
        }
    }
    EXPECT_EQ(states, (std::set<std::string>{"false", "true"}));
    EXPECT_EQ(tags.FindTag("is_serving"), "true");
}

}  // namespace rtp_llm
