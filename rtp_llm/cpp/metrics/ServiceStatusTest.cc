#include "rtp_llm/cpp/metrics/ServiceStatus.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include <condition_variable>
#include <cstdio>

#include <cstdlib>
#include <optional>
#include <set>
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

TEST_F(ServiceStatusTest, ExplicitRestartResetsReportingWindowAndRepeatedStopIsSafe) {
    setKmonServiceServing(true);
    EXPECT_GT(kmonMetricReportingStartTime().time_since_epoch().count(), 0);
    stopKmonServiceStatus();
    stopKmonServiceStatus();
    EXPECT_FALSE(isKmonMetricReportingEnabled());
    startKmonServiceStatus(nullptr);
    EXPECT_EQ(kmonMetricReportingStartTime().time_since_epoch().count(), 0);
    EXPECT_FALSE(isKmonMetricReportingEnabled());
    setKmonServiceServing(true);
    EXPECT_TRUE(isKmonServiceServing());
}

TEST_F(ServiceStatusTest, DirectSdkMetricPathsHonorReadinessSuppressionAndDraining) {
    // Exercise real MetricsGroup::report methods and SDK snapshots, not only
    // the helper: dynamically constructed tags must also pass through the gate.
    auto report_and_check = [&](bool enabled, const std::string& serving) {
        auto monitor = std::make_shared<kmonitor::KMonitor>("service_status_metric_paths");
        monitor->SetServiceName("");
        auto reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
        kmonitor::MetricsTags tags("priority", "7");
        tags.AddTag(KMON_SERVICE_STATUS_TAG, "stale");

        RpcMetricsCollector rpc;
        rpc.error_qps  = true;
        rpc.error_code = ErrorCode::GENERATE_TIMEOUT;
        reporter->report<RpcMetrics, RpcMetricsCollector>(&tags, &rpc);
        RtpLLMCacheOperationMetricsCollector operation;
        operation.latency_us = 17;
        for (auto type : {RtpLLMCacheOperationMetricsCollector::OpType::MALLOC,
                          RtpLLMCacheOperationMetricsCollector::OpType::MALLOC_RETRY,
                          RtpLLMCacheOperationMetricsCollector::OpType::INSERT,
                          RtpLLMCacheOperationMetricsCollector::OpType::FREE}) {
            operation.operation_type = type;
            reporter->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(&tags, &operation);
        }
        RtpLLMCacheCopyErrorMetricsCollector copy_error;
        copy_error.copy_direction = "host_to_device";
        copy_error.error_type     = "copy_failed";
        reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheCopyErrorMetricsCollector>(&tags, &copy_error);
        RtpLLMCacheTransferMetricsCollector transfer;
        transfer.operation                = "load";
        transfer.source_tier              = "host";
        transfer.target_tier              = "device";
        transfer.descriptors_per_transfer = 3;
        transfer.latency_us               = 23;
        transfer.success                  = false;
        reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(&tags, &transfer);
        RtpLLMCacheReuseMetricsCollector reuse;
        reuse.report_load_metrics           = true;
        reuse.report_load_join              = true;
        reuse.report_load_join_wait         = true;
        reuse.report_match_to_ready_latency = true;
        reuse.load_attempted                = true;
        reuse.join_dependency_count         = 2;
        reuse.join_wait_latency_us          = 31;
        reuse.match_to_ready_latency_us     = 37;
        reporter->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(&tags, &reuse);
        RtpLLmEplbMetricsCollector eplb{};
        eplb.ep_rank            = 2;
        eplb.gpu_loads          = {19};
        eplb.update_layer_id    = 0;
        eplb.update_weights_qps = true;
        reporter->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(&tags, &eplb);
        reportKmonMetric(reporter, 41, "rtp_llm_model_inputs_log_us", kmonitor::GAUGE, &tags);
        reportKmonMetric(reporter, 1, "rtp_llm_cuda_graph_replay_oom_retry_qps", kmonitor::QPS);

        kmonitor::MetricsCollector snapshot;
        monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, 10000);
        std::set<std::string> names;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            EXPECT_EQ(record->Tags()->FindTag(KMON_SERVICE_STATUS_TAG), serving);
            for (const auto* value : record->Values()) {
                names.insert(value->Name());
                if (value->Name() == "rtp_llm_rpc_error_qps") {
                    EXPECT_EQ(record->Tags()->FindTag("priority"), "7");
                    EXPECT_EQ(record->Tags()->FindTag("error_code"), "603");
                    EXPECT_EQ(record->Tags()->FindTag("error_code_name"),
                              ErrorCodeToString(ErrorCode::GENERATE_TIMEOUT));
                } else if (value->Name() == "rtp_llm_kv_cache_memory_cache_copy_error_qps") {
                    EXPECT_EQ(record->Tags()->FindTag("copy_direction"), "host_to_device");
                    EXPECT_EQ(record->Tags()->FindTag("error_type"), "copy_failed");
                } else if (value->Name() == "rtp_llm_kv_cache_descriptors_per_transfer") {
                    EXPECT_EQ(record->Tags()->FindTag("operation"), "load");
                    EXPECT_EQ(record->Tags()->FindTag("source_tier"), "host");
                    EXPECT_EQ(record->Tags()->FindTag("target_tier"), "device");
                    kmonitor::MinMaxCalculator expected;
                    expected.Add(3);
                    EXPECT_EQ(value->Value(), expected.ToString());
                } else if (value->Name() == "rtp_llm_gpu_loads") {
                    EXPECT_EQ(record->Tags()->FindTag("ep_rank"), "2");
                    EXPECT_EQ(record->Tags()->FindTag("layer"), "0");
                    kmonitor::MinMaxCalculator expected;
                    expected.Add(19);
                    EXPECT_EQ(value->Value(), expected.ToString());
                }
            }
        }
        if (!enabled) {
            EXPECT_TRUE(names.empty());
        } else {
            for (const auto* name : {"rtp_llm_rpc_error_qps",
                                     "rtp_llm_kv_cache_malloc_qps",
                                     "rtp_llm_kv_cache_malloc_retry_qps",
                                     "rtp_llm_kv_cache_insert_qps",
                                     "rtp_llm_kv_cache_free_qps",
                                     "rtp_llm_kv_cache_memory_cache_copy_error_qps",
                                     "rtp_llm_kv_cache_descriptors_per_transfer",
                                     "rtp_llm_kv_cache_load_fail_qps",
                                     "rtp_llm_kv_cache_load_joined_request_qps",
                                     "rtp_llm_kv_cache_load_join_wait_latency_us",
                                     "rtp_llm_kv_cache_match_to_ready_latency_us",
                                     "rtp_llm_gpu_loads",
                                     "rtp_llm_update_layer_weights_qps",
                                     "rtp_llm_model_inputs_log_us",
                                     "rtp_llm_cuda_graph_replay_oom_retry_qps"}) {
                EXPECT_EQ(names.count(name), 1) << name;
            }
        }
        EXPECT_EQ(tags.FindTag(KMON_SERVICE_STATUS_TAG), "stale");
    };
    report_and_check(false, "");
    setKmonServiceServing(true);
    report_and_check(true, "true");
    {
        ScopedKmonMetricsSuppression warmup;
        report_and_check(false, "");
    }
    setKmonServiceServing(false);
    report_and_check(true, "false");
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
    RtpLLMTokenPSMetricsCollector                                warmup;
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
