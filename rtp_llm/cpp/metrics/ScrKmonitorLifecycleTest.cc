#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "autil/EnvUtil.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitorWorker.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "kmonitor/client/core/MetricsSystem.h"
#include "kmonitor/client/core/MetricsTags.h"
#include "kmonitor/client/sink/Sink.h"
#include "gtest/gtest.h"
#include <map>
#include <memory>
#include <set>
#include <string>

namespace rtp_llm {

class RecordingSink: public kmonitor::Sink {
public:
    explicit RecordingSink(const std::string& metricName): Sink("FlumeSink"), metricName_(metricName) {}

    bool Init() override {
        return true;
    }

    void PutMetrics(kmonitor::MetricsRecord* record) override {
        if (record->Name() == metricName_ && !record->Values().empty()) {
            sawMetric_ = true;
            tags_      = record->Tags()->GetTagsMap();
        }
    }

    void Flush() override {}

    bool sawMetric() const {
        return sawMetric_;
    }

    const std::map<std::string, std::string>& tags() const {
        return tags_;
    }

private:
    std::string                        metricName_;
    bool                               sawMetric_ = false;
    std::map<std::string, std::string> tags_;
};

class ScrKmonitorLifecycleTest: public ::testing::Test {
protected:
    void TearDown() override {
        stopKmonitorFactory();
    }
};

TEST_F(ScrKmonitorLifecycleTest, InactiveFactoryStaysInactive) {
    EXPECT_TRUE(resumeKmonitorAfterScr());
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
}

TEST_F(ScrKmonitorLifecycleTest, DeferredDefaultModeStartsAutomaticReporter) {
    autil::EnvGuard scrEnabled("RTPLLM_ENABLE_SCR", "1");
    autil::EnvGuard scrPhase("SCR_PHASE", "checkpoint");
    autil::EnvGuard manualMode("kmonitorManuallyMode", "false");

    ASSERT_TRUE(initKmonitorFactory());
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
    ASSERT_TRUE(resumeKmonitorAfterScr());

    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
}

TEST_F(ScrKmonitorLifecycleTest, FirstActivationPreservesMetricsAndUsesRestoredIdentity) {
    autil::EnvGuard scrEnabled("RTPLLM_ENABLE_SCR", "1");
    autil::EnvGuard scrPhase("SCR_PHASE", "checkpoint");
    autil::EnvGuard oldHost("HIPPO_SLAVE_IP", "10.0.0.1");
    autil::EnvGuard oldRequestedIp("RequestedIP", "10.1.0.1");
    autil::EnvGuard hippoRole("HIPPO_ROLE", "rtp_role");
    autil::EnvGuard serviceName("kmonitorServiceName", "scr_service");
    // Keep the sampling threads disabled so ManuallySnapshot observes exactly
    // the record produced below. Automatic-mode activation is covered above.
    autil::EnvGuard manualMode("kmonitorManuallyMode", "true");

    ASSERT_TRUE(initKmonitorFactory());
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
    auto* monitor = kmonitor::KMonitorFactory::GetKMonitor("scr_deferred");
    ASSERT_NE(monitor, nullptr);
    monitor->Register("retained", kmonitor::GAUGE);
    auto* retainedMetric = monitor->DeclareMetric("retained", nullptr);
    ASSERT_NE(retainedMetric, nullptr);
    monitor->Report(retainedMetric, 1.0);
    kmonitor::MetricsCollector seedCollector;
    monitor->GetMetrics(&seedCollector, {kmonitor::NORMAL}, 1);
    bool sawSeedMetric = false;
    for (auto* record : seedCollector.GetRecords().getRecords()) {
        if (record->Name() == "scr_service.retained" && !record->Values().empty()) {
            sawSeedMetric = true;
            EXPECT_EQ(record->Tags()->FindTag("hippo_slave_ip"), "10.0.0.1");
            EXPECT_EQ(record->Tags()->FindTag("host_ip"), "10.0.0.1");
            EXPECT_EQ(record->Tags()->FindTag("container_ip"), "10.1.0.1");
        }
    }
    EXPECT_TRUE(sawSeedMetric);

    autil::EnvGuard newHost("HIPPO_SLAVE_IP", "10.0.0.2");
    autil::EnvGuard newRequestedIp("RequestedIP", "10.1.0.2");
    ASSERT_TRUE(resumeKmonitorAfterScr());

    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetConfig()->sink_address(), "10.0.0.2:4141");
    EXPECT_EQ(monitor, kmonitor::KMonitorFactory::GetKMonitor("scr_deferred"));
    auto* system = kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem();
    auto sink = system->GetSink("FlumeSink");
    ASSERT_TRUE(resumeKmonitorAfterScr());
    EXPECT_EQ(system->GetSink("FlumeSink"), sink);
    system->Stop();
    auto recordingSink = std::make_shared<RecordingSink>("scr_service.retained");
    ASSERT_TRUE(recordingSink->Init());
    ASSERT_TRUE(system->AddSink(recordingSink));
    monitor->Report(retainedMetric, 2.0);
    system->ManuallySnapshot();
    ASSERT_TRUE(recordingSink->sawMetric());
    EXPECT_EQ(recordingSink->tags().at("hippo_slave_ip"), "10.0.0.2");
    EXPECT_EQ(recordingSink->tags().at("host_ip"), "10.0.0.2");
    EXPECT_EQ(recordingSink->tags().at("container_ip"), "10.1.0.2");
    for (const auto& tag : recordingSink->tags()) {
        EXPECT_NE(tag.second, "10.0.0.1");
        EXPECT_NE(tag.second, "10.1.0.1");
    }
    EXPECT_TRUE(monitor->UndeclareMetric(retainedMetric));
}

}  // namespace rtp_llm
