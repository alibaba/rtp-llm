#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "autil/EnvUtil.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitorWorker.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsSystem.h"
#include "kmonitor/client/core/MetricsTags.h"
#include "gtest/gtest.h"
#include <memory>

namespace rtp_llm {

class ScrKmonitorLifecycleTest: public ::testing::Test {
protected:
    void TearDown() override {
        kmonitor::KMonitorFactory::Shutdown();
    }
};

TEST_F(ScrKmonitorLifecycleTest, InactiveFactoryStaysInactive) {
    EXPECT_FALSE(pauseKmonitorForScr());
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
}

TEST_F(ScrKmonitorLifecycleTest, RetainsRegisteredMetricsAndReleasesOldSink) {
    kmonitor::MetricsConfig config;
    config.set_inited(true);
    config.set_manually_mode(false);
    config.set_sink_address("127.0.0.1:1");
    config.set_service_name("scr_lifecycle_test");
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    kmonitor::KMonitorFactory::Start();
    auto* monitor = kmonitor::KMonitorFactory::GetKMonitor("scr_lifecycle");
    ASSERT_NE(monitor, nullptr);
    monitor->Register("retained", kmonitor::GAUGE);
    auto* system = kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem();
    for (int cycle = 0; cycle < 2; ++cycle) {
        std::weak_ptr<kmonitor::Sink> old_sink = system->GetSink("FlumeSink");
        ASSERT_FALSE(old_sink.expired());
        ASSERT_TRUE(pauseKmonitorForScr());
        EXPECT_TRUE(old_sink.expired());
        EXPECT_EQ(monitor, kmonitor::KMonitorFactory::GetKMonitor("scr_lifecycle"));
        monitor->Report("retained", cycle);
        resumeKmonitorAfterScr();
        EXPECT_TRUE(system->Started());
        EXPECT_FALSE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
        EXPECT_EQ(monitor, kmonitor::KMonitorFactory::GetKMonitor("scr_lifecycle"));
    }
}

TEST_F(ScrKmonitorLifecycleTest, DefersTransportAndRefreshesSinkAfterScr) {
    autil::EnvGuard scrEnabled("RTPLLM_ENABLE_SCR", "1");
    autil::EnvGuard scrPhase("SCR_PHASE", "checkpoint");
    autil::EnvGuard oldHost("HIPPO_SLAVE_IP", "10.0.0.1");
    autil::EnvGuard oldRequestedIp("RequestedIP", "10.1.0.1");
    autil::EnvGuard hippoRole("HIPPO_ROLE", "rtp_role");

    ASSERT_TRUE(initKmonitorFactory());
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
    auto* monitor = kmonitor::KMonitorFactory::GetKMonitor("scr_deferred");
    ASSERT_NE(monitor, nullptr);
    monitor->Register("retained", kmonitor::GAUGE);
    EXPECT_TRUE(pauseKmonitorForScr());

    autil::EnvGuard newHost("HIPPO_SLAVE_IP", "10.0.0.2");
    autil::EnvGuard newRequestedIp("RequestedIP", "10.1.0.2");
    resumeKmonitorAfterScr();

    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetConfig()->sink_address(), "10.0.0.2:4141");
    EXPECT_EQ(monitor, kmonitor::KMonitorFactory::GetKMonitor("scr_deferred"));
    const auto tags = monitor->GetMetricsTags({})->GetTagsMap();
    EXPECT_EQ(tags.at("hippo_slave_ip"), "10.0.0.2");
    EXPECT_EQ(tags.at("host_ip"), "10.0.0.2");
    EXPECT_EQ(tags.at("container_ip"), "10.1.0.2");
}

}  // namespace rtp_llm
