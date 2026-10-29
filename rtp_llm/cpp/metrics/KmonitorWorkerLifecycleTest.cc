#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitorWorker.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsSystem.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

void configure(kmonitor::MetricsConfig& config) {
    config.set_inited(true);
    config.set_manually_mode(true);
    config.set_use_common_tag(false);
    config.set_service_name("lifecycle");
    config.set_tenant_name("test");
    config.set_sink_address("127.0.0.1:1");
}

size_t sample(kmonitor::KMonitorWorker& worker) {
    // Public API traverses the actual MetricsSystem source list. An additional
    // public KMonitor snapshot lets us inspect presence without private access.
    worker.getMetricsSystem()->ManuallySnapshot();
    kmonitor::MetricsCollector snapshot;
    for (const auto& entry : *worker.GetKMonitorMap()) {
        entry.second->GetMetrics(&snapshot, {kmonitor::NORMAL}, 10000);
    }
    return snapshot.GetRecords().getRecords().size();
}

class KmonitorWorkerLifecycleTest: public ::testing::Test {
protected:
    void SetUp() override {
        kmonitor::KMonitorFactory::Shutdown();
    }
    void TearDown() override {
        kmonitor::KMonitorFactory::Shutdown();
    }

    void startPayloadOnly() {
        kmonitor::MetricsConfig config;
        configure(config);
        ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
        kmonitor::KMonitorFactory::Start();
        ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
        EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 0u);
        // Exercise the real system's source list, not just the monitor map.
        EXPECT_EQ(sample(*kmonitor::KMonitorFactory::GetWorker()), 0u);
    }
};

TEST_F(KmonitorWorkerLifecycleTest, FullShutdownThenPayloadOnlyRestart) {
    for (int repeat = 0; repeat < 3; ++repeat) {
        kmonitor::MetricsConfig config;
        configure(config);
        ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
        kmonitor::KMonitorFactory::registerBuildInMetrics();
        kmonitor::KMonitorFactory::Start();
        ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
        ASSERT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 1u);
        kmonitor::KMonitorFactory::Shutdown();
        startPayloadOnly();
        kmonitor::KMonitorFactory::Shutdown();
    }
}

TEST_F(KmonitorWorkerLifecycleTest, InitOnlyBuiltinShutdownThenPayloadRestart) {
    kmonitor::MetricsConfig config;
    configure(config);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    kmonitor::MetricsTags tags("owner", "init-only");
    kmonitor::KMonitorFactory::registerBuildInMetrics(&tags);
    ASSERT_FALSE(kmonitor::KMonitorFactory::IsStarted());
    kmonitor::KMonitorFactory::Shutdown();
    startPayloadOnly();
}

TEST_F(KmonitorWorkerLifecycleTest, ReleaseBuiltinBeforeStartThenPayloadRestart) {
    kmonitor::MetricsConfig config;
    configure(config);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    kmonitor::MetricsTags tags("owner", "released");
    kmonitor::KMonitorFactory::registerBuildInMetrics(&tags);
    kmonitor::KMonitorFactory::ReleaseKMonitor("buildinmetric");
    kmonitor::KMonitorFactory::Start();
    EXPECT_EQ(sample(*kmonitor::KMonitorFactory::GetWorker()), 0u);
    kmonitor::KMonitorFactory::Shutdown();
    startPayloadOnly();
}

TEST_F(KmonitorWorkerLifecycleTest, InitOnlyWorkerShutdownRetiresActualSources) {
    kmonitor::MetricsConfig config;
    configure(config);
    kmonitor::KMonitorWorker worker;
    ASSERT_TRUE(worker.Init(config));
    worker.registerBuildInMetrics(nullptr);
    ASSERT_EQ(worker.GetKMonitorMap()->size(), 1u);
    worker.Shutdown();
    EXPECT_TRUE(worker.GetKMonitorMap()->empty());
    worker.Start();
    EXPECT_EQ(sample(worker), 0u);
}

TEST_F(KmonitorWorkerLifecycleTest, UnrelatedWorkerShutdownDoesNotRetireBuiltinOwner) {
    kmonitor::MetricsConfig config;
    configure(config);
    kmonitor::KMonitorWorker owner;
    ASSERT_TRUE(owner.Init(config));
    owner.registerBuildInMetrics(nullptr);
    {
        kmonitor::KMonitorWorker other;
        ASSERT_TRUE(other.Init(config));
        ASSERT_NE(other.GetKMonitor("other"), nullptr);
        other.Shutdown();
        EXPECT_TRUE(other.GetKMonitorMap()->empty());
    }
    owner.Start();
    EXPECT_GT(sample(owner), 0u);
}

TEST_F(KmonitorWorkerLifecycleTest, StartedWorkerCanRestartWithoutStaleSources) {
    kmonitor::MetricsConfig config;
    configure(config);
    kmonitor::KMonitorWorker worker;
    ASSERT_TRUE(worker.Init(config));
    worker.registerBuildInMetrics(nullptr);
    worker.Start();
    worker.Shutdown();
    ASSERT_TRUE(worker.GetKMonitorMap()->empty());
    worker.Start();
    EXPECT_EQ(sample(worker), 0u);
}

TEST_F(KmonitorWorkerLifecycleTest, DestroyingOneReporterDoesNotStopAnother) {
    startPayloadOnly();
    auto first  = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    auto second = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    ASSERT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), 2u);
    first.reset();
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    ASSERT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), 1u);
    second->report(37, "payload", kmonitor::QPS, nullptr, false);
    EXPECT_GT(sample(*kmonitor::KMonitorFactory::GetWorker()), 0u);
    second.reset();
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetKMonitorMap()->empty());
}

}  // namespace
}  // namespace rtp_llm
