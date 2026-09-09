#include "rtp_llm/cpp/metrics/ServiceStatus.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include <condition_variable>
#include <cstdio>

#include <cstdlib>
#include <optional>
#include <string>
#include <unistd.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

constexpr const char* STARTUP_WARMUP_HEALTH_GATE_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE";

class ServiceStatusTest: public ::testing::Test {
protected:
    void SetUp() override {
        const char* old_value = std::getenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV);
        if (old_value != nullptr) {
            old_gate_value_ = old_value;
        }
        ::unsetenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV);
        startKmonServiceStatus(nullptr);
    }

    void TearDown() override {
        stopKmonServiceStatus();
        if (old_gate_value_.has_value()) {
            ::setenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, old_gate_value_->c_str(), 1);
        } else {
            ::unsetenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV);
        }
    }

private:
    std::optional<std::string> old_gate_value_;
};

TEST_F(ServiceStatusTest, AddsCurrentServingStateWithoutDroppingCallerTags) {
    kmonitor::MetricsTags input("source", "test");
    input.AddTag(KMON_SERVICE_STATUS_TAG, "stale");

    auto stopped_tags = kmonTagsWithServiceStatus(&input);
    EXPECT_EQ(stopped_tags.FindTag("source"), "test");
    EXPECT_EQ(stopped_tags.FindTag(KMON_SERVICE_STATUS_TAG), "false");

    setKmonServiceServing(true);
    auto serving_tags = kmonTagsWithServiceStatus(&input);
    EXPECT_EQ(serving_tags.FindTag(KMON_SERVICE_STATUS_TAG), "true");
}

TEST_F(ServiceStatusTest, ScopedSuppressionIsNested) {
    setKmonServiceServing(true);
    EXPECT_TRUE(isKmonMetricReportingEnabled());
    {
        ScopedKmonMetricsSuppression outer;
        EXPECT_FALSE(isKmonMetricReportingEnabled());
        {
            ScopedKmonMetricsSuppression inner;
            EXPECT_FALSE(isKmonMetricReportingEnabled());
        }
        EXPECT_FALSE(isKmonMetricReportingEnabled());
    }
    EXPECT_TRUE(isKmonMetricReportingEnabled());
}

TEST_F(ServiceStatusTest, StartupWarmupGateSuppressesMetrics) {
    const auto missing_gate = "/tmp/rtp_llm_missing_warmup_gate_" + std::to_string(::getpid());
    ::setenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, missing_gate.c_str(), 1);
    startKmonServiceStatus(nullptr);
    setKmonServiceServing(true);

    EXPECT_FALSE(isKmonServiceServing());
    EXPECT_FALSE(isKmonMetricReportingEnabled());
}

TEST_F(ServiceStatusTest, ShutdownIsTerminalAndKeepsDrainingMetricsEnabled) {
    EXPECT_FALSE(isKmonMetricReportingEnabled());
    setKmonServiceServing(true);
    EXPECT_TRUE(isKmonServiceServing());
    setKmonServiceServing(false);
    setKmonServiceServing(true);
    reportKmonServiceStatus();
    EXPECT_FALSE(isKmonServiceServing());
    EXPECT_TRUE(isKmonMetricReportingEnabled());
}

TEST_F(ServiceStatusTest, ShutdownBeforeReadyNeverEnablesMetrics) {
    setKmonServiceServing(false);
    setKmonServiceServing(true);
    reportKmonServiceStatus();
    EXPECT_FALSE(isKmonServiceServing());
    EXPECT_FALSE(isKmonMetricReportingEnabled());
}

TEST_F(ServiceStatusTest, GateOpeningEnablesReportingOnlyUntilShutdown) {
    char gate_file[] = "/tmp/rtp_llm_kmon_gate_XXXXXX";
    int  fd          = ::mkstemp(gate_file);
    ASSERT_GE(fd, 0);
    ::close(fd);
    ::unlink(gate_file);
    ::setenv(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, gate_file, 1);
    startKmonServiceStatus(nullptr);
    setKmonServiceServing(true);
    EXPECT_FALSE(isKmonMetricReportingEnabled());
    FILE* file = ::fopen(gate_file, "w");
    ASSERT_NE(file, nullptr);
    ::fclose(file);
    reportKmonServiceStatus();
    EXPECT_TRUE(isKmonServiceServing());
    ::unlink(gate_file);
    reportKmonServiceStatus();
    EXPECT_TRUE(isKmonMetricReportingEnabled());
    setKmonServiceServing(false);
    reportKmonServiceStatus();
    EXPECT_FALSE(isKmonServiceServing());
    EXPECT_TRUE(isKmonMetricReportingEnabled());
}

TEST_F(ServiceStatusTest, ReportsHeartbeatWithoutSchedulerActivity) {
    kmonitor::MetricsConfig config;
    config.set_service_name("");
    auto monitor = std::make_shared<kmonitor::KMonitor>("service_status_test");
    monitor->SetConfig(&config);
    monitor->SetServiceName("");
    auto reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags("dp_rank", "0"));
    startKmonServiceStatus(reporter);
    auto wait_for_sample = [&](const std::string& serving) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            kmonitor::MetricsCollector samples;
            const auto                 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
            monitor->GetMetrics(&samples, {kmonitor::NORMAL}, now_ms);
            for (const auto* record : samples.GetRecords().getRecords()) {
                if (record->Tags()->FindTag(KMON_SERVICE_STATUS_TAG) != serving) {
                    continue;
                }
                for (const auto* value : record->Values()) {
                    if (value->Name() == KMON_SERVICE_STATUS_METRIC) {
                        EXPECT_EQ(record->Tags()->FindTag("dp_rank"), "0");
                        return true;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    setKmonServiceServing(true);
    EXPECT_TRUE(wait_for_sample("true"));
    EXPECT_TRUE(wait_for_sample("true"));
    setKmonServiceServing(false);
    EXPECT_TRUE(wait_for_sample("false"));
    EXPECT_TRUE(wait_for_sample("false"));
    stopKmonServiceStatus();
}

class RecordingTpsMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager*) override {
        return true;
    }
    void report(const kmonitor::MetricsTags*, RtpLLMTokenPSMetricsCollector* collector) {
        std::lock_guard<std::mutex> lock(mutex);
        if (collector->hasContextTPS()) {
            context_tps = collector->contextTPS();
        }
        ++reports;
        changed.notify_all();
    }
    std::mutex              mutex;
    std::condition_variable changed;
    double                  context_tps = 0;
    int                     reports     = 0;
};

template<template<typename, typename> class Reporter>
void checkWarmupSamplesAreDropped() {
    // A MetricsGroup with no registered SDK metrics needs no live KMonitor.
    auto  reporter = std::make_shared<kmonitor::MetricsReporter>(kmonitor::KMonitorPtr{}, "", kmonitor::MetricsTags{});
    auto* recorded = reporter->getMetricsGroup<RecordingTpsMetrics>();
    Reporter<RecordingTpsMetrics, RtpLLMTokenPSMetricsCollector> loop(reporter, 100);
    {
        std::unique_lock<std::mutex> lock(recorded->mutex);
        ASSERT_TRUE(recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return recorded->reports > 0; }));
    }
    RtpLLMTokenPSMetricsCollector warmup;
    warmup.addTokenSize(10000, 10000, 0, 10000, 1000000);
    loop.report(&warmup);
    setKmonServiceServing(true);
    RtpLLMTokenPSMetricsCollector real;
    real.addTokenSize(100, 100, 0, 100, 1000000);
    loop.report(&real);
    std::unique_lock<std::mutex> lock(recorded->mutex);
    ASSERT_TRUE(recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return recorded->context_tps > 0; }));
    EXPECT_DOUBLE_EQ(recorded->context_tps, 100.0);
}

TEST_F(ServiceStatusTest, ExecutionTpsDropsWarmupBeforeGateOpens) {
    checkWarmupSamplesAreDropped<MetricsLoopReporter>();
}

TEST_F(ServiceStatusTest, WallTpsDropsWarmupBeforeGateOpens) {
    // The wall-clock loop stays silent while startup is suppressed, so there
    // is no initial idle report to wait for. Collect before creating a thread.
    auto  reporter = std::make_shared<kmonitor::MetricsReporter>(kmonitor::KMonitorPtr{}, "", kmonitor::MetricsTags{});
    auto* recorded = reporter->getMetricsGroup<RecordingTpsMetrics>();
    WallClockMetricsLoopReporter<RecordingTpsMetrics, RtpLLMTokenPSMetricsCollector> loop(reporter, 100);
    RtpLLMTokenPSMetricsCollector                                                    warmup;
    warmup.addTokenSize(10000, 10000, 0, 10000, 1000000);
    loop.report(&warmup);
    setKmonServiceServing(true);
    RtpLLMTokenPSMetricsCollector real;
    real.addTokenSize(100, 100, 0, 100, 1000000);
    loop.report(&real);
    std::unique_lock<std::mutex> lock(recorded->mutex);
    ASSERT_TRUE(recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return recorded->context_tps > 0; }));
    EXPECT_DOUBLE_EQ(recorded->context_tps, 100.0);
}

}  // namespace
}  // namespace rtp_llm
