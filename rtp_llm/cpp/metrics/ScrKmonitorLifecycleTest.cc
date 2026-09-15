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
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetConfig()->manually_mode());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->Started());
    EXPECT_FALSE(kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem()->GetSink("FlumeSink"));
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
    autil::EnvGuard hippoRole("HIPPO_ROLE", "seed_role");
    autil::EnvGuard hippoApp("HIPPO_APP", "seed_app");
    autil::EnvGuard hippoGroup("HIPPO_SERVICE_NAME", "seed_group");
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
    kmonitor::MetricsTags customTags;
    customTags.AddTag("custom", "kept");
    auto* retainedMetric = monitor->DeclareMetric("retained", &customTags);
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
    autil::EnvGuard newRole("HIPPO_ROLE", "restored_role");
    autil::EnvGuard newApp("HIPPO_APP", "restored_app");
    autil::EnvGuard newGroup("HIPPO_SERVICE_NAME", "restored_group");
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
    EXPECT_EQ(recordingSink->tags().at("hippo_role"), "restored_role");
    EXPECT_EQ(recordingSink->tags().at("hippo_app"), "restored_app");
    EXPECT_EQ(recordingSink->tags().at("hippo_group"), "restored_group");
    EXPECT_EQ(recordingSink->tags().at("hippo_cluster"), "restored_group");
    EXPECT_EQ(recordingSink->tags().at("custom"), "kept");
    for (const auto& tag : recordingSink->tags()) {
        EXPECT_NE(tag.second, "10.0.0.1");
        EXPECT_NE(tag.second, "10.1.0.1");
    }
    EXPECT_TRUE(monitor->UndeclareMetric(retainedMetric));
}

TEST_F(ScrKmonitorLifecycleTest, NormalStartupPreservesExplicitMetricTags) {
    autil::EnvGuard scrEnabled("RTPLLM_ENABLE_SCR", "0");
    autil::EnvGuard host("HIPPO_SLAVE_IP", "192.0.2.10");
    autil::EnvGuard ip("RequestedIP", "192.0.2.20");
    autil::EnvGuard role("HIPPO_ROLE", "normal_role");
    autil::EnvGuard service("kmonitorServiceName", "normal");
    autil::EnvGuard manual("kmonitorManuallyMode", "true");
    ASSERT_TRUE(initKmonitorFactory());
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    auto* monitor = kmonitor::KMonitorFactory::GetKMonitor("normal");
    monitor->Register("custom", kmonitor::GAUGE);
    kmonitor::MetricsTags explicitTags;
    explicitTags.AddTag("container_ip", "192.0.2.99");
    explicitTags.AddTag("hippo_role", "explicit_role");
    auto* metric = monitor->DeclareMetric("custom", &explicitTags);
    ASSERT_NE(metric, nullptr);
    auto* system = kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem();
    auto originalSink = system->GetSink("FlumeSink");
    ASSERT_TRUE(originalSink);
    ASSERT_TRUE(resumeKmonitorAfterScr());
    EXPECT_EQ(system->GetSink("FlumeSink"), originalSink);
    system->Stop();
    auto sink = std::make_shared<RecordingSink>("normal.custom");
    ASSERT_TRUE(system->AddSink(sink));
    monitor->Report(metric, 1.0);
    system->ManuallySnapshot();
    ASSERT_TRUE(sink->sawMetric());
    EXPECT_EQ(sink->tags().at("container_ip"), "192.0.2.99");
    EXPECT_EQ(sink->tags().at("hippo_role"), "explicit_role");
    EXPECT_TRUE(monitor->UndeclareMetric(metric));
}

TEST_F(ScrKmonitorLifecycleTest, RemovedRuntimeTagsDoNotSurviveInRetainedMetrics) {
    autil::EnvGuard scrEnabled("RTPLLM_ENABLE_SCR", "1");
    autil::EnvGuard phase("SCR_PHASE", "restore");
    autil::EnvGuard role("HIPPO_ROLE", "seed_role");
    autil::EnvGuard service("kmonitorServiceName", "scr_service");
    autil::EnvGuard manual("kmonitorManuallyMode", "true");
    ASSERT_TRUE(initKmonitorFactory());
    auto* monitor = kmonitor::KMonitorFactory::GetKMonitor("scr_removed");
    monitor->Register("retained", kmonitor::GAUGE);
    auto* metric = monitor->DeclareMetric("retained", nullptr);
    ASSERT_NE(metric, nullptr);
    ASSERT_EQ(unsetenv("HIPPO_ROLE"), 0);
    ASSERT_TRUE(resumeKmonitorAfterScr());
    auto* system = kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem();
    system->Stop();
    auto sink = std::make_shared<RecordingSink>("scr_service.retained");
    ASSERT_TRUE(system->AddSink(sink));
    monitor->Report(metric, 1.0);
    system->ManuallySnapshot();
    ASSERT_TRUE(sink->sawMetric());
    for (const auto& key : {"hippo_role", "hippo_group", "host_ip", "container_ip"}) {
        EXPECT_EQ(sink->tags().count(key), 0);
    }
    EXPECT_TRUE(monitor->UndeclareMetric(metric));
}

TEST_F(ScrKmonitorLifecycleTest, TagOverridesOnlyModifySelectedKeys) {
    const std::map<std::string, std::string> original = {
        {"identity", "seed"}, {"removed", "seed"}, {"custom", "kept"}};
    auto tags = std::make_shared<kmonitor::MetricsTags>(original);
    kmonitor::MetricsRecord record(nullptr, tags, 0);
    const std::map<std::string, std::string> fresh = {
        {"identity", "restored"}, {"added", "fresh"}, {"custom", "unexpected"}, {"unselected", "unexpected"}};
    const std::set<std::string> keys = {"identity", "removed", "added"};

    record.OverrideTags(fresh, keys);
    const std::map<std::string, std::string> expected = {
        {"identity", "restored"}, {"added", "fresh"}, {"custom", "kept"}};
    EXPECT_EQ(record.Tags()->GetTagsMap(), expected);
    EXPECT_EQ(tags->GetTagsMap(), original);
    auto refreshed = record.Tags();
    record.OverrideTags(fresh, keys);
    EXPECT_EQ(record.Tags(), refreshed);
    record.OverrideTags({{"unselected", "unexpected"}}, {});
    EXPECT_EQ(record.Tags(), refreshed);
}

}  // namespace rtp_llm
