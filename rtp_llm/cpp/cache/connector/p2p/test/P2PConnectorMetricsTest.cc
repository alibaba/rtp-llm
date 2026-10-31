#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {
namespace {

struct MetricSample {
    std::string value;
    std::string side;
    std::string reason;
    std::string stage;
    std::string success;
};

class P2PWritebackMetricsTest: public ::testing::Test {
protected:
    void SetUp() override {
        monitor_ = std::make_shared<kmonitor::KMonitor>("p2p_writeback_metrics_test");
        monitor_->SetServiceName("");
        reporter_ = std::make_shared<kmonitor::MetricsReporter>(monitor_, "", kmonitor::MetricsTags());
    }

    void snapshot() {
        samples_.clear();
        kmonitor::MetricsCollector collector;
        snapshot_time_ms_ += 10000;
        monitor_->GetMetrics(&collector, {kmonitor::NORMAL}, snapshot_time_ms_);
        for (const auto* record : collector.GetRecords().getRecords()) {
            const auto tags = record->Tags();
            for (const auto* metric : record->Values()) {
                samples_[metric->Name()].push_back({metric->Value(),
                                                    tags->FindTag("side"),
                                                    tags->FindTag("reason"),
                                                    tags->FindTag("stage"),
                                                    tags->FindTag("success")});
            }
        }
    }

    void expectGauge(const std::string& name, const std::vector<double>& expected_samples) {
        const auto found = samples_.find(name);
        ASSERT_NE(found, samples_.end()) << name;
        ASSERT_EQ(found->second.size(), 1) << name;
        kmonitor::MinMaxCalculator expected;
        for (const auto value : expected_samples) {
            expected.Add(value);
        }
        EXPECT_EQ(found->second.front().value, expected.ToString()) << name;
    }

    bool hasPositiveQps(const std::string& name) const {
        const auto found = samples_.find(name);
        if (found == samples_.end()) {
            return false;
        }
        for (const auto& sample : found->second) {
            if (std::stod(sample.value) > 0) {
                return true;
            }
        }
        return false;
    }

    kmonitor::KMonitorPtr                            monitor_;
    kmonitor::MetricsReporterPtr                     reporter_;
    std::map<std::string, std::vector<MetricSample>> samples_;
    int64_t                                          snapshot_time_ms_{0};
};

TEST_F(P2PWritebackMetricsTest, DecodeCompletionDoesNotReportDuplicateHoldTime) {
    WriteSchedulerMetricsCollector metrics;
    metrics.total_cost_time_us = 100;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    expectGauge("rtp_llm_p2p_writeback_cost_time_us", {100});
    EXPECT_EQ(samples_.count("rtp_llm_p2p_writeback_hold_time_us"), 0);
}

TEST_F(P2PWritebackMetricsTest, RepeatedCheckerInitDoesNotReplaceTheRunningReporter) {
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeSimpleMhaCacheConfig(2, 100, 4, DataType::TYPE_FP16, 1, 2).topologyPtr();
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(config, reporter_));
    EXPECT_FALSE(checker.init(config, nullptr));
    EXPECT_EQ(checker.metrics_reporter_, reporter_);
    checker.stop();
}

TEST_F(P2PWritebackMetricsTest, SkipPreservesReasonWithoutCompletingAnAttempt) {
    WriteSchedulerMetricsCollector metrics;
    metrics.skip_reason = "kv_reserve";
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    ASSERT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_skipped_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_failed_qps"));
    EXPECT_EQ(samples_.count("rtp_llm_p2p_writeback_cost_time_us"), 0);
    const auto& sample = samples_.at("rtp_llm_p2p_writeback_skipped_qps").front();
    EXPECT_EQ(sample.reason, "kv_reserve");
    EXPECT_EQ(sample.side, "decode");
}

TEST_F(P2PWritebackMetricsTest, CompletionReportsPublishedVolumeAndDistinctDurations) {
    WriteSchedulerMetricsCollector metrics;
    metrics.prefill               = true;
    metrics.submitted             = true;
    metrics.total_cost_time_us    = 100;
    metrics.hold_time_us          = 60;
    metrics.drain_time_us         = 10;
    metrics.planned_bytes         = 256;
    metrics.committed_token_count = 8;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_failed_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_skipped_qps"));
    EXPECT_EQ(samples_.count("rtp_llm_p2p_writeback_qps"), 0);
    expectGauge("rtp_llm_p2p_writeback_cost_time_us", {100});
    expectGauge("rtp_llm_p2p_writeback_hold_time_us", {60});
    expectGauge("rtp_llm_p2p_writeback_drain_time_us", {10});
    expectGauge("rtp_llm_p2p_writeback_committed_token_count", {8});
    EXPECT_EQ(samples_.at("rtp_llm_p2p_writeback_committed_token_count").front().side, "prefill");
}

TEST_F(P2PWritebackMetricsTest, PolicyRejectionsPreserveReasonAcrossStartWriteBackRpc) {
    for (const bool prefill : {true, false}) {
        for (const auto& error :
             {ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted"),
              ErrorInfo(ErrorCode::MALLOC_FAILED, "writeback would leave too few available KV blocks")}) {
            WriteSchedulerMetricsCollector metrics;
            metrics.prefill            = prefill;
            metrics.submitted          = !prefill;
            metrics.total_cost_time_us = 100;
            metrics.error =
                prefill ? error :
                          ErrorInfo(error.code(), "StartWriteBack peer=127.0.0.1:1234 key=test: " + error.ToString());
            reportWritebackMetrics(reporter_, metrics);
        }
    }
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_failed_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_skipped_qps"));
    const auto& rejected_samples = samples_.at("rtp_llm_p2p_writeback_rejected_qps");
    ASSERT_EQ(rejected_samples.size(), 4);
    for (const auto& sample : rejected_samples) {
        EXPECT_GT(std::stod(sample.value), 0);
        EXPECT_TRUE(sample.side == "prefill" || sample.side == "decode");
        EXPECT_TRUE(sample.reason == "prefix_evicted_or_demoted" || sample.reason == "kv_capacity");
    }
    EXPECT_EQ(samples_.at("rtp_llm_p2p_writeback_cost_time_us").size(), 4);
}

TEST_F(P2PWritebackMetricsTest, ExecutionErrorsAreFailuresNotPolicyRejections) {
    for (const auto& error :
         {ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback transfer deadline exceeded"),
          ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "StartWriteBack RPC failed"),
          ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source workers or plan mismatch"),
          ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback receive submission queue unavailable"),
          ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load failed"),
          ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED,
                    "writeback publication incomplete: final reuse endpoint unavailable"),
          ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted: unexpected failure"),
          ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "prefix_evicted_or_demoted")}) {
        WriteSchedulerMetricsCollector metrics;
        metrics.error = error;
        reportWritebackMetrics(reporter_, metrics);
    }
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_failed_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_skipped_qps"));
}

TEST_F(P2PWritebackMetricsTest, NoTransferIsASubsetOfSuccessOnly) {
    WriteSchedulerMetricsCollector metrics;
    metrics.no_transfer = true;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_no_transfer_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_failed_qps"));
    metrics.error = ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "prefix_evicted_or_demoted");
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_rejected_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_no_transfer_qps"));
}

TEST_F(P2PWritebackMetricsTest, FailedStageDoesNotReportSuccessfulTransferredBytes) {
    WritebackStageMetricsCollector metrics;
    metrics.prefill           = true;
    metrics.stage             = "transfer";
    metrics.error             = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "transfer deadline exceeded");
    metrics.transferred_bytes = 256;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    EXPECT_EQ(samples_.count("rtp_llm_p2p_writeback_transferred_bytes"), 0);
    expectGauge("rtp_llm_p2p_writeback_stage_cost_time_us", {0});
    const auto& sample = samples_.at("rtp_llm_p2p_writeback_stage_cost_time_us").front();
    EXPECT_EQ(sample.stage, "transfer");
    EXPECT_EQ(sample.success, "false");
}

TEST_F(P2PWritebackMetricsTest, SuccessfulStageReportsPayloadSize) {
    WritebackStageMetricsCollector metrics;
    metrics.prefill           = true;
    metrics.stage             = "transfer";
    metrics.cost_time_us      = 10;
    metrics.transferred_bytes = 128;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_transferred_bytes", {128});
    const auto& sample = samples_.at("rtp_llm_p2p_writeback_transferred_bytes").front();
    EXPECT_EQ(sample.side, "prefill");
    EXPECT_EQ(sample.success, "true");
}

TEST_F(P2PWritebackMetricsTest, IdleStatusReportsZerosWithoutRequestLabels) {
    WritebackStatusMetricsCollector metrics;
    reportWritebackMetrics(reporter_, metrics);
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_inflight_context_count", {0});
    expectGauge("rtp_llm_p2p_writeback_oldest_inflight_ms", {0});
    expectGauge("rtp_llm_p2p_writeback_retained_context_count", {0});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_bytes", {0});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_percent", {0});
    expectGauge("rtp_llm_p2p_writeback_oldest_kv_hold_ms", {0});
    EXPECT_TRUE(samples_.at("rtp_llm_p2p_writeback_retained_kv_bytes").front().reason.empty());
}

TEST_F(P2PWritebackMetricsTest, CacheReuseReportsOnlyWritebackBenefitIncludingZero) {
    RtpLLMCacheReuseMetricsCollector metrics;
    metrics.report_reuse_lengths = true;
    metrics.writeback_reuse_len = 4;
    reporter_->report<RtpLLMCacheReuseMetrics>(nullptr, &metrics);
    metrics.writeback_reuse_len = 0;
    reporter_->report<RtpLLMCacheReuseMetrics>(nullptr, &metrics);
    snapshot();
    expectGauge("rtp_llm_writeback_reuse_len", {4, 0});
    EXPECT_EQ(samples_.at("rtp_llm_writeback_reuse_len").front().side, "prefill");
    EXPECT_EQ(samples_.count("rtp_llm_actual_reuse_len"), 0);
    EXPECT_EQ(samples_.count("rtp_llm_reuse_len"), 0);
    EXPECT_EQ(samples_.count("rtp_llm_writeback_first_use_len"), 0);
}

TEST_F(P2PWritebackMetricsTest, ReuseSubphaseReportsDoNotEmitRequestLengths) {
    RtpLLMCacheReuseMetricsCollector metrics;
    metrics.writeback_reuse_len    = 4;
    metrics.report_load_join      = true;
    metrics.join_dependency_count = 1;
    reporter_->report<RtpLLMCacheReuseMetrics>(nullptr, &metrics);
    snapshot();
    EXPECT_EQ(samples_.count("rtp_llm_writeback_reuse_len"), 0);
    EXPECT_EQ(samples_.count("rtp_llm_actual_reuse_len"), 0);
    EXPECT_EQ(samples_.count("rtp_llm_reuse_len"), 0);
}

TEST_F(P2PWritebackMetricsTest, ResourceReleaseFailureDoesNotPrematurelyCompleteMetrics) {
    int  releases = 0;
    auto context  = std::make_shared<P2PConnectorAsyncWriteContext>(
        "metrics-release",
        currentTimeMs() + 10000,
        nullptr,
        100,
        100,
        [&] {
            if (++releases == 1) {
                throw std::runtime_error("release not ready");
            }
        },
        reporter_);
    context->finishWithoutTransfer();
    EXPECT_TRUE(context->resourceHoldPending());
    snapshot();
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    context->checkDone();
    EXPECT_FALSE(context->resourceHoldPending());
    snapshot();
    EXPECT_TRUE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
    context->checkDone();
    snapshot();
    EXPECT_FALSE(hasPositiveQps("rtp_llm_p2p_writeback_success_qps"));
}

TEST_F(P2PWritebackMetricsTest, CheckerReportsCancelledHoldsAndClearsAfterPhysicalStop) {
    const auto config   = test::makeSimpleMhaCacheConfig(2, 100, 4, DataType::TYPE_FP16, 1, 2);
    auto       resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config.topologyPtr());
    resource->mutableBlockIds("default").assign({1, 2});
    P2PConnectorAsyncWriteContextChecker checker;
    checker.accepting_        = true;
    checker.metrics_reporter_ = reporter_;
    ASSERT_TRUE(checker.retained_kv_.init(config.topologyPtr(), 30, false, reporter_));
    std::vector<std::shared_ptr<P2PConnectorAsyncWriteContext>> contexts;
    for (const auto& key : {"metrics-hold-1", "metrics-hold-2"}) {
        auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
            key, currentTimeMs() + 10000, nullptr, 100, 100, std::function<void()>{}, reporter_, resource);
        ASSERT_TRUE(checker.addContext(context));
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        context->cancel();
        contexts.push_back(context);
    }
    checker.checkOnce();
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_inflight_context_count", {2});
    expectGauge("rtp_llm_p2p_writeback_retained_context_count", {2});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_bytes",
                {static_cast<double>(2 * config.topology().blockSizeBytesForGroup("default"))});
    for (const auto& context : contexts) {
        context->peer_stopped_     = true;
        context->transfer_stopped_ = true;
    }
    checker.checkOnce();
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_inflight_context_count", {0});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_bytes", {0});
    expectGauge("rtp_llm_p2p_writeback_oldest_kv_hold_ms", {0});
}

TEST_F(P2PWritebackMetricsTest, PrefillStatusReportsTargetHoldsAndClearsAfterPhysicalStop) {
    const auto config   = test::makeSimpleMhaCacheConfig(2, 100, 4, DataType::TYPE_FP16, 1, 2);
    auto       resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config.topologyPtr());
    resource->mutableBlockIds("default").assign({1, 2});
    P2PSchedulerPrefillWrite scheduler({}, nullptr, reporter_, nullptr);
    ASSERT_TRUE(scheduler.retained_kv_.init(config.topologyPtr(), 30, true, reporter_));
    auto state       = std::make_shared<P2PSchedulerPrefillWrite::PrefillWriteRequest>(currentTimeMs() + 10000);
    state->cancelled = true;
    state->resource_start_time_us = currentTimeUs();
    scheduler.requests_.emplace("metrics-target", state);
    scheduler.retained_resources_.emplace("metrics-target", resource);
    scheduler.checkRetainedResources();
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_inflight_context_count", {1});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_bytes",
                {static_cast<double>(2 * config.topology().blockSizeBytesForGroup("default"))});
    EXPECT_EQ(samples_.at("rtp_llm_p2p_writeback_retained_kv_bytes").front().side, "prefill");
    state->stopped = true;
    scheduler.retained_resources_.clear();
    scheduler.checkRetainedResources();
    snapshot();
    expectGauge("rtp_llm_p2p_writeback_inflight_context_count", {0});
    expectGauge("rtp_llm_p2p_writeback_retained_kv_bytes", {0});
}

}  // namespace
}  // namespace rtp_llm
