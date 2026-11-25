#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <numeric>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class ScopedServingMetrics {
public:
    ScopedServingMetrics() {
        const char* value = std::getenv("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE");
        if (value) {
            old_gate_ = value;
        }
        ::unsetenv("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE");
        startKmonServiceStatus(nullptr);
        setKmonServiceServing(true);
    }
    ~ScopedServingMetrics() {
        stopKmonServiceStatus();
        if (old_gate_) {
            ::setenv("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE", old_gate_->c_str(), 1);
        }
    }

private:
    std::optional<std::string> old_gate_;
};

class RtpLLMTokenPSMetricsReportTest: public ::testing::Test {
protected:
    void SetUp() override {
        monitor_ = std::make_shared<kmonitor::KMonitor>("token_ps_report_test");
        monitor_->SetServiceName("");
        reporter_ = std::make_shared<kmonitor::MetricsReporter>(monitor_, "", kmonitor::MetricsTags());
        metrics_  = reporter_->getMetricsGroup<RtpLLMTokenPSMetrics>();
        ASSERT_NE(metrics_, nullptr);
    }

    void report(RtpLLMTokenPSMetricsCollector& collector, const std::string& priority = "30") {
        kmonitor::MetricsTags tags("priority", priority);
        metrics_->report(&tags, &collector);
    }

    void expectSnapshot(const std::map<std::string, std::vector<double>>& expected_samples,
                        const std::string&                                priority = "30") {
        kmonitor::MetricsCollector snapshot;
        snapshot_time_ms_ += 10000;
        monitor_->GetMetrics(&snapshot, {kmonitor::NORMAL}, snapshot_time_ms_);
        std::map<std::string, std::string> actual;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                EXPECT_EQ(record->Tags()->FindTag("priority"), priority);
                EXPECT_TRUE(actual.emplace(value->Name(), value->Value()).second);
            }
        }
        ASSERT_EQ(actual.size(), expected_samples.size()) << ::testing::PrintToString(actual);
        for (const auto& [name, samples] : expected_samples) {
            SCOPED_TRACE(name);
            ASSERT_FALSE(samples.empty());
            const auto metric = actual.find(name);
            ASSERT_NE(metric, actual.end());
            // Let Kmonitor encode the expected samples instead of duplicating its snapshot format.
            kmonitor::MinMaxCalculator expected;
            for (double sample : samples) {
                expected.Add(sample);
            }
            EXPECT_EQ(metric->second, expected.ToString());
        }
    }

private:
    ScopedServingMetrics         serving_metrics_;
    kmonitor::KMonitorPtr        monitor_;
    kmonitor::MetricsReporterPtr reporter_;
    RtpLLMTokenPSMetrics*        metrics_          = nullptr;
    int64_t                      snapshot_time_ms_ = 0;
};

struct TokenCountReport {
    std::string priority;
    int64_t     context_tokens;
    int64_t     context_tokens_with_cache;
};

class RecordingTokenCountMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager*) override {
        return true;
    }

    void report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector) {
        std::lock_guard<std::mutex> lock(mutex_);
        TokenCountReport            report{
            tags->FindTag("priority"), collector->contextTokens(), collector->contextTokensWithCache()};
        last_report_was_idle_ = collector->reportZeroTPS();
        if (last_report_was_idle_) {
            idle_report_ = report;
        } else {
            data_reports_.push_back(report);
        }
        cv_.notify_all();
    }

    bool waitForIdleAfter(size_t count, std::vector<TokenCountReport>& reports, TokenCountReport& idle_report) {
        std::unique_lock<std::mutex> lock(mutex_);
        // An idle report after the data proves that the real reporting loop drained its collector.
        if (!cv_.wait_for(lock, std::chrono::seconds(5), [&] {
                return data_reports_.size() >= count && last_report_was_idle_;
            })) {
            return false;
        }
        reports.swap(data_reports_);
        idle_report = idle_report_;
        return true;
    }

private:
    std::mutex                    mutex_;
    std::condition_variable       cv_;
    std::vector<TokenCountReport> data_reports_;
    TokenCountReport              idle_report_{};
    bool                          last_report_was_idle_ = false;
};

template<typename Reporter>
void checkReporterResetsTokenCounts(bool wall_clock = false) {
    ScopedServingMetrics serving_metrics;
    auto                 monitor = std::make_shared<kmonitor::KMonitor>("token_count_reset_test");
    auto  metrics_reporter       = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    auto* recording_metrics      = metrics_reporter->getMetricsGroup<RecordingTokenCountMetrics>();
    ASSERT_NE(recording_metrics, nullptr);
    Reporter reporter(metrics_reporter, 1);

    using CountsByPriority = std::map<std::string, std::pair<int64_t, int64_t>>;
    auto expect_window     = [&](const CountsByPriority& expected) {
        std::vector<TokenCountReport> reports;
        TokenCountReport              idle_report;
        ASSERT_TRUE(recording_metrics->waitForIdleAfter(expected.size(), reports, idle_report))
            << "Reporter did not drain its data and emit an idle window";
        ASSERT_EQ(reports.size(), expected.size());
        CountsByPriority actual;
        for (const auto& report : reports) {
            EXPECT_TRUE(
                actual.emplace(report.priority, std::make_pair(report.context_tokens, report.context_tokens_with_cache))
                    .second);
        }
        EXPECT_EQ(actual, expected);
        if (!wall_clock) {
            EXPECT_EQ(idle_report.priority, "0");
        }
        EXPECT_EQ(idle_report.context_tokens, 0);
        EXPECT_EQ(idle_report.context_tokens_with_cache, 0);
    };

    RtpLLMTokenPSMetricsCollector first;
    first.addTokenSize(400, 600, 0, 400, 1000000);
    first.addPriorityTokenSize(30, 100, 150, 0, 100, 1000000);
    first.addPriorityTokenSize(50, 300, 450, 0, 300, 1000000);
    reporter.report(&first);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"30", {100, 150}}, {"50", {300, 450}}}));

    // Reusing one priority must not retain either its old counts or the absent priority's bucket.
    RtpLLMTokenPSMetricsCollector second;
    second.addTokenSize(20, 30, 0, 20, 1000000);
    second.addPriorityTokenSize(50, 20, 30, 0, 20, 1000000);
    reporter.report(&second);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"50", {20, 30}}}));

    // Exercise the untagged path after reporting priority buckets.
    RtpLLMTokenPSMetricsCollector untagged;
    untagged.addTokenSize(10, 15, 0, 10, 1000000);
    reporter.report(&untagged);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"0", {10, 15}}}));

    RtpLLMTokenPSMetricsCollector cache_only;
    cache_only.addTokenSize(0, 25, 0, 0, 1000000);
    cache_only.addPriorityTokenSize(70, 0, 25, 0, 0, 1000000);
    reporter.report(&cache_only);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"70", {0, 25}}}));
}

}  // namespace

TEST_F(RtpLLMTokenPSMetricsReportTest, DoesNotRepeatContextMetricsAfterUntimedBatch) {
    RtpLLMTokenPSMetricsCollector prefill;
    prefill.addTokenSize(100, 150, 0, 100, 1000000);
    report(prefill);
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {100}},
                                            {"rtp_llm_context_tokens_with_cache", {150}},
                                            {"rtp_llm_context_tps", {100}},
                                            {"rtp_llm_context_tps_with_cache", {150}},
                                            {"rtp_llm_total_tps", {100}}}));

    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        // Production prefill also counts its executed tokens in total.
        RtpLLMTokenPSMetricsCollector untimed;
        untimed.addTokenSize(20, 30, 0, 20, execute_time_us);
        untimed.markIdleWindow();
        report(untimed);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_total_tps", {20}}}));
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
    }

    RtpLLMTokenPSMetricsCollector idle;
    idle.markIdleWindow();
    report(idle, "0");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {0}},
                                            {"rtp_llm_context_tokens_with_cache", {0}},
                                            {"rtp_llm_context_tps", {0}},
                                            {"rtp_llm_context_tps_with_cache", {0}},
                                            {"rtp_llm_generate_tps", {0}},
                                            {"rtp_llm_total_tps", {0}}},
                                           "0"));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, ReportsCacheOnlyCountsOnlyWithValidTiming) {
    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        RtpLLMTokenPSMetricsCollector untimed;
        untimed.addTokenSize(0, 25, 0, 0, execute_time_us);
        report(untimed);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
    }

    RtpLLMTokenPSMetricsCollector timed;
    timed.addTokenSize(0, 25, 0, 0, 1000000);
    timed.markIdleWindow();
    report(timed);
    ASSERT_NO_FATAL_FAILURE(
        expectSnapshot({{"rtp_llm_context_tokens_with_cache", {25}}, {"rtp_llm_context_tps_with_cache", {25}}}));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, KeepsDecodeCountsWithoutContextMetrics) {
    for (int64_t execute_time_us : {1000000, 0, -1}) {
        SCOPED_TRACE(execute_time_us);
        RtpLLMTokenPSMetricsCollector decode;
        decode.addTokenSize(0, 0, 7, 7, execute_time_us);
        report(decode);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_generate_tps", {7}}, {"rtp_llm_total_tps", {7}}}));
    }

    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, DoesNotAddZeroTpsSamplesWithinSnapshot) {
    RtpLLMTokenPSMetricsCollector prefill;
    prefill.addTokenSize(100, 150, 0, 100, 1000000);
    report(prefill);
    RtpLLMTokenPSMetricsCollector untimed;
    untimed.addTokenSize(20, 30, 0, 0, 0);
    report(untimed);
    RtpLLMTokenPSMetricsCollector decode;
    decode.addTokenSize(0, 0, 7, 7, 1000000);
    report(decode);

    // Several reports share one snapshot: missing TPS must not dilute its valid samples with zeros.
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {100}},
                                            {"rtp_llm_context_tokens_with_cache", {150}},
                                            {"rtp_llm_context_tps", {100}},
                                            {"rtp_llm_context_tps_with_cache", {150}},
                                            {"rtp_llm_generate_tps", {7}},
                                            {"rtp_llm_total_tps", {100, 7}}}));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, PriorityTokenCountsUseTheSameSamplesAsTps) {
    RtpLLMTokenPSMetricsCollector collector;
    collector.addTokenSize(400, 600, 7, 407, 200000);
    collector.addPriorityTokenSize(30, 400, 600, 0, 400, 200000);
    collector.addPriorityTokenSize(50, 0, 0, 7, 7, 200000);
    collector.addTokenSize(100, 150, 0, 100, 0);
    collector.addPriorityTokenSize(70, 100, 150, 0, 100, 0);

    auto priorities = collector.priorityCollectorsForReport();
    ASSERT_EQ(priorities.size(), 3);
    report(priorities.at(30), "30");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {400}},
                                            {"rtp_llm_context_tokens_with_cache", {600}},
                                            {"rtp_llm_context_tps", {2000}},
                                            {"rtp_llm_context_tps_with_cache", {3000}},
                                            {"rtp_llm_total_tps", {400}}}));

    // All priorities share the valid global execution window, including buckets with no valid context work.
    report(priorities.at(50), "50");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {0}},
                                            {"rtp_llm_context_tokens_with_cache", {0}},
                                            {"rtp_llm_context_tps", {0}},
                                            {"rtp_llm_context_tps_with_cache", {0}},
                                            {"rtp_llm_generate_tps", {7}},
                                            {"rtp_llm_total_tps", {7}}},
                                           "50"));
    report(priorities.at(70), "70");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {0}},
                                            {"rtp_llm_context_tokens_with_cache", {0}},
                                            {"rtp_llm_context_tps", {0}},
                                            {"rtp_llm_context_tps_with_cache", {0}},
                                            {"rtp_llm_total_tps", {100}}},
                                           "70"));

    report(collector, "0");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens", {400}},
                                            {"rtp_llm_context_tokens_with_cache", {600}},
                                            {"rtp_llm_context_tps", {2000}},
                                            {"rtp_llm_context_tps_with_cache", {3000}},
                                            {"rtp_llm_generate_tps", {7}},
                                            {"rtp_llm_total_tps", {507}}},
                                           "0"));
}

TEST(RtpLLMTokenPSMetricsCollectorTest, LoopReporterResetsTokenCountsAfterReport) {
    checkReporterResetsTokenCounts<MetricsLoopReporter<RecordingTokenCountMetrics, RtpLLMTokenPSMetricsCollector>>();
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallClockReporterResetsTokenCountsAfterReport) {
    checkReporterResetsTokenCounts<
        WallClockMetricsLoopReporter<RecordingTokenCountMetrics, RtpLLMTokenPSMetricsCollector>>(true);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsLongPrefillByExecutionTime) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(256000, 256000, 0, 256000, 10 * 1000 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 25600.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 25600.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 256000.0, 1e-6);
    EXPECT_TRUE(collector.hasContextTPS());
    EXPECT_TRUE(collector.hasContextTPSWithCache());
    EXPECT_TRUE(collector.hasTotalTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergesShortPrefillsByExecutionTime) {
    RtpLLMTokenPSMetricsCollector collector;

    for (int i = 0; i < 10; ++i) {
        collector.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    }

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 10000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsContextTpsWithCacheIncludingReuseTokens) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsTimeWeightedTps) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    second.addTokenSize(9000, 9000, 0, 9000, 900 * 1000);
    merged.merge(&first);
    merged.merge(&second);

    EXPECT_NEAR(merged.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(merged.contextTPSWithCache(), 10000.0, 1e-6);
    EXPECT_NEAR(merged.totalTPS(), 10000.0, 1e-6);
    EXPECT_EQ(merged.contextTokens(), 10000);
    EXPECT_EQ(merged.contextTokensWithCache(), 10000);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsPriorityMetrics) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(400, 600, 4, 404, 40 * 1000);
    first.addPriorityTokenSize(30, 400, 600, 4, 404, 40 * 1000);
    second.addTokenSize(600, 900, 6, 606, 60 * 1000);
    second.addPriorityTokenSize(50, 600, 900, 6, 606, 60 * 1000);
    merged.merge(&first);
    merged.merge(&second);

    auto priority_collectors = merged.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).contextTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextTPS(), 6000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).totalTPS(), 404.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).totalTPS(), 606.0, 1e-6);
    EXPECT_EQ(priority_collectors.at(30).contextTokens(), 400);
    EXPECT_EQ(priority_collectors.at(50).contextTokens(), 600);
    EXPECT_EQ(priority_collectors.at(30).contextTokensWithCache(), 600);
    EXPECT_EQ(priority_collectors.at(50).contextTokensWithCache(), 900);
    EXPECT_EQ(merged.contextTokens(), 1000);
    EXPECT_EQ(merged.contextTokensWithCache(), 1500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalAsTokenCounts) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 10, 1010, 100 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 10.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 1010.0, 1e-6);
    EXPECT_EQ(collector.contextTokens(), 1000);
    EXPECT_EQ(collector.contextTokensWithCache(), 1500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalWhenExecutionTimeIsInvalid) {
    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        RtpLLMTokenPSMetricsCollector collector;
        collector.addTokenSize(1000, 1500, 2, 1002, execute_time_us);
        collector.addPriorityTokenSize(30, 1000, 1500, 2, 1002, execute_time_us);

        auto priorities = collector.priorityCollectorsForReport();
        ASSERT_EQ(priorities.size(), 1);
        for (const auto* counts : {&collector, &priorities.at(30)}) {
            EXPECT_EQ(counts->contextTokens(), 0);
            EXPECT_EQ(counts->contextTokensWithCache(), 0);
            EXPECT_FALSE(counts->hasContextTPS());
            EXPECT_FALSE(counts->hasContextTPSWithCache());
            EXPECT_TRUE(counts->hasGenerateTPS());
            EXPECT_TRUE(counts->hasTotalTPS());
            EXPECT_NEAR(counts->generateTPS(), 2.0, 1e-6);
            EXPECT_NEAR(counts->totalTPS(), 1002.0, 1e-6);
        }
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MarksEmptyIdleWindowForZeroReport) {
    RtpLLMTokenPSMetricsCollector collector;

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());

    // Discarded context counts leave the window empty.
    collector.addTokenSize(1000, 1500, 0, 0, 0);
    collector.markIdleWindow();

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_TRUE(collector.reportZeroTPS());
    EXPECT_NEAR(collector.contextTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 0.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 0.0, 1e-6);
    EXPECT_EQ(collector.contextTokens(), 0);
    EXPECT_EQ(collector.contextTokensWithCache(), 0);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, DoesNotMarkNonEmptyWindowAsIdle) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 2, 1002, 100 * 1000);
    collector.markIdleWindow();

    EXPECT_TRUE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergeKeepsIdleZeroOnlyForEmptyMetrics) {
    RtpLLMTokenPSMetricsCollector idle;
    RtpLLMTokenPSMetricsCollector merged;

    idle.addTokenSize(1000, 1500, 0, 0, -1);
    idle.markIdleWindow();
    merged.merge(&idle);

    EXPECT_FALSE(merged.hasMetrics());
    EXPECT_TRUE(merged.reportZeroTPS());

    RtpLLMTokenPSMetricsCollector non_empty;
    non_empty.addTokenSize(1000, 1000, 0, 1000, 100 * 1000);
    merged.merge(&non_empty);

    EXPECT_TRUE(merged.hasMetrics());
    EXPECT_FALSE(merged.reportZeroTPS());
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ReportsContextWallTpsByReportWindow) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);
    collector.setReportWindowUs(200 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
    EXPECT_NEAR(collector.contextWallTPS(), 5000.0, 1e-6);
    EXPECT_NEAR(collector.contextWallTPSWithCache(), 7500.0, 1e-6);
    EXPECT_EQ(collector.reportWindowUs(), 200 * 1000);
    EXPECT_EQ(collector.contextTokens(), 1000);
    EXPECT_EQ(collector.contextTokensWithCache(), 1500);

    collector.setReportWindowUs(1000 * 1000);
    EXPECT_NEAR(collector.contextWallTPS(), 1000.0, 1e-6);
    EXPECT_NEAR(collector.contextWallTPSWithCache(), 1500.0, 1e-6);
    EXPECT_EQ(collector.contextTokens(), 1000);
    EXPECT_EQ(collector.contextTokensWithCache(), 1500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallTpsUsesMergedTokens) {
    RtpLLMTokenPSMetricsCollector first;
    RtpLLMTokenPSMetricsCollector second;
    RtpLLMTokenPSMetricsCollector merged;

    first.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);
    second.addTokenSize(3000, 4500, 0, 3000, 100 * 1000);
    merged.merge(&first);
    merged.merge(&second);
    merged.setReportWindowUs(1 * 1000 * 1000);

    EXPECT_NEAR(merged.contextWallTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(merged.contextWallTPSWithCache(), 6000.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGlobalAndPriorityMetrics) {
    RtpLLMTokenPSMetricsCollector collector;
    collector.addTokenSize(1000, 1500, 10, 1010, 100 * 1000);
    collector.addPriorityTokenSize(30, 400, 600, 4, 404, 100 * 1000);
    collector.addPriorityTokenSize(50, 600, 900, 6, 606, 100 * 1000);
    collector.setReportWindowUs(200 * 1000);

    auto priority_collectors = collector.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).contextTPS(), 4000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextTPS(), 6000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).contextWallTPS(), 2000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).contextWallTPS(), 3000.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS(), 4.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).generateTPS(), 6.0, 1e-6);

    EXPECT_NEAR(priority_collectors.at(30).contextTPS() + priority_collectors.at(50).contextTPS(),
                collector.contextTPS(),
                1e-6);
    EXPECT_NEAR(priority_collectors.at(30).contextWallTPS() + priority_collectors.at(50).contextWallTPS(),
                collector.contextWallTPS(),
                1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS() + priority_collectors.at(50).generateTPS(),
                collector.generateTPS(),
                1e-6);

    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter(nullptr);
    tps_reporter.report(&collector);

    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter(
        nullptr);
    wall_tps_reporter.report(&collector);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, AddTokenSizeByPriorityMatchesUntaggedTotals) {
    struct TokenCounts {
        int64_t context            = 0;
        int64_t context_with_cache = 0;
        int64_t generate           = 0;
        int64_t total              = 0;
    };
    // Mirrors the MTP decode path: accepted tokens bucketed per priority with
    // generate == total and no context contribution; bucket sums must equal
    // the untagged totals reported through addTokenSize.
    std::map<int32_t, TokenCounts> counts_by_priority;
    counts_by_priority[30] = {0, 0, 7, 7};
    counts_by_priority[50] = {0, 0, 3, 3};

    RtpLLMTokenPSMetricsCollector collector;
    const int64_t                 total_accepted = 10;
    collector.addTokenSize(0, 0, total_accepted, total_accepted, 100 * 1000);
    collector.addTokenSizeByPriority(counts_by_priority, 100 * 1000);

    auto priority_collectors = collector.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 2);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS(), 7.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(50).generateTPS(), 3.0, 1e-6);
    EXPECT_NEAR(priority_collectors.at(30).generateTPS() + priority_collectors.at(50).generateTPS(),
                collector.generateTPS(),
                1e-6);
    EXPECT_NEAR(
        priority_collectors.at(30).totalTPS() + priority_collectors.at(50).totalTPS(), collector.totalTPS(), 1e-6);
    EXPECT_FALSE(priority_collectors.at(30).hasContextTPS());
    EXPECT_FALSE(priority_collectors.at(50).hasContextTPS());
    EXPECT_EQ(collector.contextTokens(), 0);
    EXPECT_EQ(collector.contextTokensWithCache(), 0);
    for (const auto& [priority, priority_collector] : priority_collectors) {
        EXPECT_EQ(priority_collector.contextTokens(), 0);
        EXPECT_EQ(priority_collector.contextTokensWithCache(), 0);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergingUntimedCountsPreservesTimeValidatedTps) {
    RtpLLMTokenPSMetricsCollector timed;
    timed.addTokenSize(700, 900, 0, 700, 100 * 1000);
    timed.addPriorityTokenSize(30, 700, 900, 0, 700, 100 * 1000);
    RtpLLMTokenPSMetricsCollector untimed;
    untimed.addTokenSize(300, 500, 0, 300, 0);
    untimed.addPriorityTokenSize(30, 300, 500, 0, 300, 0);
    timed.merge(&untimed);
    timed.setReportWindowUs(2 * 1000 * 1000);

    auto priority_collectors = timed.priorityCollectorsForReport();
    ASSERT_EQ(priority_collectors.size(), 1);
    for (const auto* collector : {&timed, &priority_collectors.at(30)}) {
        EXPECT_EQ(collector->contextTokens(), 700);
        EXPECT_EQ(collector->contextTokensWithCache(), 900);
        EXPECT_DOUBLE_EQ(collector->contextTPS(), 7000.0);
        EXPECT_DOUBLE_EQ(collector->contextTPSWithCache(), 9000.0);
        EXPECT_DOUBLE_EQ(collector->contextWallTPS(), 350.0);
        EXPECT_DOUBLE_EQ(collector->contextWallTPSWithCache(), 450.0);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, IgnoresNonPositiveContextTokenCounts) {
    RtpLLMTokenPSMetricsCollector collector;
    for (int64_t token_num : {0, -1}) {
        collector.addTokenSize(token_num, token_num, 0, 0, 100 * 1000);
        collector.addPriorityTokenSize(30, token_num, token_num, 0, 0, 100 * 1000);
    }

    EXPECT_EQ(collector.contextTokens(), 0);
    EXPECT_EQ(collector.contextTokensWithCache(), 0);
    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_EQ(collector.priorityCollectorsForReport().at(30).contextTokens(), 0);
    EXPECT_EQ(collector.priorityCollectorsForReport().at(30).contextTokensWithCache(), 0);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallPrioritiesAreZeroFilledAcrossIdleAndUntaggedWindows) {
    std::set<int32_t>             known;
    RtpLLMTokenPSMetricsCollector first;
    first.addTokenSize(100, 150, 0, 100, 1000);
    first.addPriorityTokenSize(39, 100, 150, 0, 100, 1000);
    first.setReportWindowUs(1000000);
    EXPECT_DOUBLE_EQ(first.priorityCollectorsForReport(known).at(39).contextWallTPSWithCache(), 150);

    RtpLLMTokenPSMetricsCollector next;
    next.addTokenSize(0, 250, 0, 0, 1000);  // Fully cached prefill still contributes to with-cache TPS.
    next.addPriorityTokenSize(50, 0, 250, 0, 0, 1000);
    next.setReportWindowUs(2000000);
    auto buckets = next.priorityCollectorsForReport(known);
    ASSERT_EQ(buckets.size(), 2);
    EXPECT_DOUBLE_EQ(buckets.at(39).contextWallTPSWithCache(), 0);
    EXPECT_EQ(buckets.at(39).reportWindowUs(), 2000000);
    EXPECT_DOUBLE_EQ(buckets.at(50).contextWallTPSWithCache(), 125);

    RtpLLMTokenPSMetricsCollector idle;
    idle.markIdleWindow();
    idle.setReportWindowUs(3000000);
    buckets = idle.priorityCollectorsForReport(known);
    ASSERT_EQ(buckets.size(), 2);
    EXPECT_EQ(buckets.count(0), 0);
    for (const auto& [priority, bucket] : buckets) {
        EXPECT_TRUE(bucket.reportZeroTPS());
        EXPECT_DOUBLE_EQ(bucket.contextWallTPSWithCache(), 0);
        EXPECT_EQ(bucket.reportWindowUs(), 3000000);
    }
    RtpLLMTokenPSMetricsCollector untagged;
    untagged.addTokenSize(70, 90, 0, 70, 1000);
    untagged.setReportWindowUs(1000000);
    buckets = untagged.priorityCollectorsForReport(known);
    ASSERT_EQ(buckets.size(), 3);
    EXPECT_DOUBLE_EQ(buckets.at(0).contextWallTPSWithCache(), 90);
    EXPECT_DOUBLE_EQ(buckets.at(39).contextWallTPSWithCache(), 0);
    EXPECT_DOUBLE_EQ(buckets.at(50).contextWallTPSWithCache(), 0);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, OnlineMixedPriorityKmonitorAndDownsampleConserveTokens) {
    ScopedServingMetrics reporting;
    auto                        monitor = std::make_shared<kmonitor::KMonitor>("wall_priority_online_test");
    monitor->SetServiceName("");
    auto              reporter     = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    auto*             per_priority = reporter->getMetricsGroup<RtpLLMWallClockTokenPSMetrics>();
    std::set<int32_t> known;
    using Key = std::pair<std::string, std::string>;
    std::map<Key, std::vector<double>> expected;
    // Production public-pool priorities: uneven duty cycles, changing proportions, idle windows,
    // and multiple batches merged per window. Compare with independent input-token accounting.
    const std::vector<int32_t>                   priorities{39, 40, 49, 50};
    std::vector<double>                          total_samples;
    std::map<int32_t, std::vector<double>>       priority_samples;
    std::vector<double>                          oracle;
    double                                       sparse_sum = 0;
    std::map<int32_t, std::pair<double, size_t>> sparse;
    for (int tick = 0; tick < 180; ++tick) {
        SCOPED_TRACE(tick);
        RtpLLMTokenPSMetricsCollector window;
        std::map<int32_t, int64_t>    context_counts, cached_counts;
        int64_t                       all_context = 0, all_cached = 0;
        for (size_t i = 0; i < priorities.size(); ++i) {
            int priority = priorities[i];
            // All series are introduced at tick zero; later windows reproduce sparse reporting.
            bool active = tick == 0 || (tick % 17 != 0 && tick % (i + 2) == 0);
            for (int batch = 0; active && batch < 2; ++batch) {
                int64_t                       context = tick % 11 == 0 ? 0 : (i + 1) * 100 + batch * 13;
                int64_t                       cached  = context + 500 + tick * 3;
                RtpLLMTokenPSMetricsCollector step;
                step.addTokenSize(context, cached, 0, context, 1000);
                step.addPriorityTokenSize(priority, context, cached, 0, context, 1000);
                window.merge(&step);
                context_counts[priority] += context;
                cached_counts[priority] += cached;
                all_context += context;
                all_cached += cached;
            }
        }
        if (!window.hasMetrics()) {
            window.markIdleWindow();
        }
        window.setReportWindowUs(1000000);
        EXPECT_EQ(window.contextTokens(), all_context);
        auto buckets = window.priorityCollectorsForReport(known);
        ASSERT_EQ(buckets.size(), priorities.size());
        total_samples.push_back(window.contextWallTPSWithCache());
        oracle.push_back(all_cached);
        double sum = 0;
        for (auto& [priority, bucket] : buckets) {
            const auto            tag = std::to_string(priority);
            kmonitor::MetricsTags tags("priority", tag);
            per_priority->report(&tags, &bucket);
            expected[{tag, "rtp_llm_context_wall_tps"}].push_back(context_counts[priority]);
            expected[{tag, "rtp_llm_context_wall_tps_with_cache"}].push_back(cached_counts[priority]);
            expected[{tag, "rtp_llm_wall_tps_report_interval_us"}].push_back(1000000);
            priority_samples[priority].push_back(bucket.contextWallTPSWithCache());
            sum += bucket.contextWallTPSWithCache();
            if (cached_counts[priority]) {
                sparse[priority].first += cached_counts[priority];
                ++sparse[priority].second;
            }
        }
        EXPECT_DOUBLE_EQ(sum, all_cached);
        if ((tick + 1) % 10 == 0) {
            kmonitor::MetricsCollector snapshot;
            monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, (tick + 1) * 1000);
            std::map<Key, std::string> actual;
            for (const auto* record : snapshot.GetRecords().getRecords()) {
                for (const auto* value : record->Values()) {
                    EXPECT_EQ(record->Tags()->FindTag(KMON_SERVICE_STATUS_TAG), "true");
                    EXPECT_TRUE(
                        actual.emplace(Key{record->Tags()->FindTag("priority"), value->Name()}, value->Value()).second);
                }
            }
            ASSERT_EQ(actual.size(), expected.size());
            for (const auto& [key, calculator] : expected) {
                ASSERT_EQ(actual.count(key), 1);
                if (key.second == "rtp_llm_wall_tps_report_interval_us") {
                    kmonitor::MinMaxCalculator expected_interval;
                    for (double value : calculator)
                        expected_interval.Add(value);
                    EXPECT_EQ(actual.at(key), expected_interval.ToString());
                } else {
                    const double expected_tps = std::accumulate(calculator.begin(), calculator.end(), 0.0) / 10.0;
                    EXPECT_NEAR(std::stod(actual.at(key)), expected_tps, 1e-6) << key.first << ":" << key.second;
                }
            }
            expected.clear();
        }
    }
    // Reproduce sum(avg(priority)) at the dashboard's 10s / 20s / 60s resolutions.
    for (int width : {10, 20, 60}) {
        for (int begin = 0; begin < 180; begin += width) {
            double bucket_sum = 0, total_sum = 0, expected_sum = 0;
            for (int tick = begin; tick < begin + width; ++tick) {
                total_sum += total_samples[tick];
                expected_sum += oracle[tick];
                for (int priority : priorities) {
                    bucket_sum += priority_samples.at(priority)[tick];
                }
            }
            EXPECT_DOUBLE_EQ(bucket_sum / width, expected_sum / width);
            EXPECT_DOUBLE_EQ(total_sum / width, expected_sum / width);
        }
    }
    double actual_mean = 0;
    for (double count : oracle)
        actual_mean += count / oracle.size();
    for (const auto& [priority, samples] : sparse)
        sparse_sum += samples.first / samples.second;
    EXPECT_GT(sparse_sum, actual_mean * 1.5) << "Fixture must reproduce the old sparse-priority overcount";
}

namespace {
struct WallWindowRecord {
    std::string priority;
    int64_t     tokens;
    int64_t     window_us;
    double      tps;
};
class RecordingWallWindows: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager*) override {
        return true;
    }
    void report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector) {
        std::lock_guard<std::mutex> lock(mutex);
        if (collector->reportZeroTPS() && collector->reportWindowUs() == 0) {
            ++heartbeats;
            changed.notify_all();
            return;
        }
        records.push_back({tags->FindTag("priority"),
                           collector->contextTokensWithCache(),
                           collector->reportWindowUs(),
                           collector->contextWallTPSWithCache()});
        changed.notify_all();
    }
    std::mutex                    mutex;
    std::condition_variable       changed;
    std::vector<WallWindowRecord> records;
    size_t                        heartbeats = 0;
};
}  // namespace

TEST(RtpLLMTokenPSMetricsCollectorTest, WallLoopZeroFillsPrioritiesAndKeepsFullLongStepWindow) {
    ScopedServingMetrics reporting;
    auto  reporter = std::make_shared<kmonitor::MetricsReporter>(kmonitor::KMonitorPtr{}, "", kmonitor::MetricsTags{});
    auto* recorded = reporter->getMetricsGroup<RecordingWallWindows>();
    WallClockMetricsLoopReporter<RecordingWallWindows, RtpLLMTokenPSMetricsCollector> loop(reporter, 5);
    auto                                                                              guard = loop.makeActiveGuard();
    // Let any idle report taken before beginActive() finish before inspecting a fully active period.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard<std::mutex> lock(recorded->mutex);
        recorded->records.clear();
        recorded->heartbeats = 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    {
        std::lock_guard<std::mutex> lock(recorded->mutex);
        EXPECT_TRUE(recorded->records.empty()) << "An unfinished step must not drain or reset the collector";
        EXPECT_GT(recorded->heartbeats, 0) << "Active steps must keep QPS series alive without resetting their window";
    }
    RtpLLMTokenPSMetricsCollector batch;
    batch.addTokenSize(100, 150, 0, 100, 1000);
    batch.addPriorityTokenSize(39, 100, 150, 0, 100, 1000);
    loop.report(&batch);
    {
        std::unique_lock<std::mutex> lock(recorded->mutex);
        ASSERT_TRUE(
            recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return !recorded->records.empty(); }));
        ASSERT_EQ(recorded->records.size(), 1);
        const auto& first = recorded->records[0];
        EXPECT_EQ(first.priority, "39");
        EXPECT_EQ(first.tokens, 150);
        EXPECT_GE(first.window_us, 60000);
        recorded->records.clear();
    }
    // Alternate priorities in distinct real reporting-loop windows.
    for (int priority : {50, 39}) {
        RtpLLMTokenPSMetricsCollector next;
        next.addTokenSize(200, 300, 0, 200, 1000);
        next.addPriorityTokenSize(priority, 200, 300, 0, 200, 1000);
        loop.report(&next);
        std::unique_lock<std::mutex> lock(recorded->mutex);
        ASSERT_TRUE(
            recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return recorded->records.size() >= 2; }));
        ASSERT_EQ(recorded->records.size(), 2);
        int64_t sum = 0;
        for (size_t i = 0; i < 2; ++i) {
            const auto& bucket = recorded->records[i];
            EXPECT_EQ(bucket.priority, i == 0 ? "39" : "50");
            EXPECT_EQ(bucket.window_us, recorded->records[0].window_us);
            EXPECT_EQ(bucket.tokens, bucket.priority == std::to_string(priority) ? 300 : 0);
            sum += bucket.tokens;
        }
        EXPECT_EQ(sum, 300);
        recorded->records.clear();
    }
    guard = {};
    {
        std::unique_lock<std::mutex> lock(recorded->mutex);
        ASSERT_TRUE(
            recorded->changed.wait_for(lock, std::chrono::seconds(5), [&] { return recorded->records.size() >= 2; }));
        EXPECT_EQ(recorded->records[0].priority, "39");
        EXPECT_EQ(recorded->records[1].priority, "50");
        EXPECT_EQ(recorded->records[0].tokens, 0);
        EXPECT_EQ(recorded->records[1].tokens, 0);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KmonitorRatesConserveTokensWithUnequalWindowsAndLongSteps) {
    ScopedServingMetrics reporting;
    for (bool single_priority : {true, false}) {
        SCOPED_TRACE(single_priority);
        auto monitor = std::make_shared<kmonitor::KMonitor>("unequal_wall_windows_test");
        monitor->SetServiceName("");
        auto              reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
        auto*             per_priority = reporter->getMetricsGroup<RtpLLMWallClockTokenPSMetrics>();
        std::set<int32_t> known;
        auto              send = [&](int64_t window_us, int64_t tokens, int priority) {
            RtpLLMTokenPSMetricsCollector window;
            window.addTokenSize(tokens, tokens * 2, 0, tokens, 1000);
            window.addPriorityTokenSize(priority, tokens, tokens * 2, 0, tokens, 1000);
            window.setReportWindowUs(window_us);
            auto buckets = window.priorityCollectorsForReport(known);
            for (auto& [p, bucket] : buckets) {
                kmonitor::MetricsTags tags("priority", std::to_string(p));
                per_priority->report(&tags, &bucket);
            }
        };
        auto snapshot = [&](int64_t time_ms, double expected_tps) {
            kmonitor::MetricsCollector snapshot;
            monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, time_ms);
            double priority_sum = 0, priority_cache_sum = 0;
            int    rate_count = 0;
            for (const auto* record : snapshot.GetRecords().getRecords()) {
                for (const auto* value : record->Values()) {
                    if (value->Name() == "rtp_llm_context_wall_tps") {
                        priority_sum += std::stod(value->Value());
                        ++rate_count;
                    } else if (value->Name() == "rtp_llm_context_wall_tps_with_cache") {
                        priority_cache_sum += std::stod(value->Value());
                        ++rate_count;
                    }
                }
            }
            EXPECT_EQ(rate_count, known.size() * 2);
            EXPECT_DOUBLE_EQ(priority_sum, expected_tps);
            EXPECT_DOUBLE_EQ(priority_cache_sum, expected_tps * 2);
        };
        // Both zhipu (one priority) and the public pool (multiple priorities):
        // old avg(900/1s, 900/9s) = 500 TPS; correct (900+900)/10s = 180 TPS.
        send(1000000, 900, 50);
        send(9000000, 900, single_priority ? 50 : 39);
        ASSERT_NO_FATAL_FAILURE(snapshot(10000, 180));
        // A step spans the next entire SDK period: no completed tokens and no reporter update.
        ASSERT_NO_FATAL_FAILURE(snapshot(20000, 0));
        send(20000000, 3600, 50);
        ASSERT_NO_FATAL_FAILURE(snapshot(30000, 360));
        // 180*10 + 0*10 + 360*10 = all 5400 completed tokens; no stale rates or duplicates.
        ASSERT_NO_FATAL_FAILURE(snapshot(40000, 0));
    }
}

namespace {
using WallTokenTotals = std::map<std::pair<std::string, std::string>, double>;

// Integrate actual SDK output, independently of the collector's rate calculations.
void accumulateWallSnapshot(const kmonitor::KMonitorPtr& monitor, int64_t& time_ms, WallTokenTotals& totals) {
    time_ms += 10000;
    kmonitor::MetricsCollector snapshot;
    monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, time_ms);
    for (const auto* record : snapshot.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            if (value->Name() == "rtp_llm_context_wall_tps" || value->Name() == "rtp_llm_context_wall_tps_with_cache") {
                totals[{record->Tags()->FindTag("priority"), value->Name()}] += std::stod(value->Value()) * 10;
            }
        }
    }
}
}  // namespace

TEST(RtpLLMTokenPSMetricsCollectorTest, ConcurrentWallLoopAndSdkSnapshotsConserveEveryPriority) {
    ScopedServingMetrics reporting;
    auto                        monitor = std::make_shared<kmonitor::KMonitor>("wall_concurrent_test");
    monitor->SetServiceName("");
    auto reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> loop(reporter, 1);
    auto                     guard = loop.makeActiveGuard();
    WallTokenTotals          expected, actual;
    int64_t                  time_ms = 0;
    std::atomic<int>         finished{0};
    std::vector<std::thread> producers;
    // Each writer alternates priorities, includes cache-only batches, and merges concurrently
    // with the real reporting loop and SDK snapshot/reset. The oracle uses input counts only.
    for (int writer = 0; writer < 4; ++writer) {
        for (int i = 0; i < 100; ++i) {
            auto priority = std::to_string(39 + (writer + i) % 4);
            expected[{priority, "rtp_llm_context_wall_tps"}] += i % 7 ? (writer + 1) * 10 : 0;
            expected[{priority, "rtp_llm_context_wall_tps_with_cache"}] += (writer + 1) * 30;
        }
        producers.emplace_back([&, writer] {
            for (int i = 0; i < 100; ++i) {
                const int                     priority = 39 + (writer + i) % 4;
                const int                     context  = i % 7 ? (writer + 1) * 10 : 0;
                const int                     cached   = (writer + 1) * 30;
                RtpLLMTokenPSMetricsCollector batch;
                batch.addTokenSize(context, cached, 0, context, 1000);
                batch.addPriorityTokenSize(priority, context, cached, 0, context, 1000);
                loop.report(&batch);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            ++finished;
        });
    }
    while (finished.load() < 4) {
        accumulateWallSnapshot(monitor, time_ms, actual);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (auto& producer : producers)
        producer.join();
    guard                  = {};
    const auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto       all_arrived = [&] {
        for (const auto& [key, count] : expected) {
            if (actual[key] < count)
                return false;
        }
        return true;
    };
    while (!all_arrived() && std::chrono::steady_clock::now() < deadline) {
        accumulateWallSnapshot(monitor, time_ms, actual);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Continue through idle snapshots to catch replay of the last completed batch.
    for (int i = 0; i < 10; ++i) {
        accumulateWallSnapshot(monitor, time_ms, actual);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (const auto& [key, count] : actual)
        EXPECT_DOUBLE_EQ(count, expected[key]) << key.first << ':' << key.second;
    for (const auto& [key, count] : expected)
        EXPECT_DOUBLE_EQ(actual[key], count) << key.first << ':' << key.second;
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallQpsRecoversAfterSdkSeriesRecyclingAndLatePriority) {
    ScopedServingMetrics reporting;
    auto                        monitor = std::make_shared<kmonitor::KMonitor>("wall_recycle_test");
    monitor->SetServiceName("");
    auto              reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    auto*             metrics  = reporter->getMetricsGroup<RtpLLMWallClockTokenPSMetrics>();
    std::set<int32_t> known;
    auto              send = [&](int priority, int64_t tokens, int64_t cached) {
        RtpLLMTokenPSMetricsCollector window;
        window.addTokenSize(tokens, cached, 0, tokens, 1000);
        window.addPriorityTokenSize(priority, tokens, cached, 0, tokens, 1000);
        for (auto& [p, bucket] : window.priorityCollectorsForReport(known)) {
            kmonitor::MetricsTags tags("priority", std::to_string(p));
            metrics->report(&tags, &bucket);
        }
    };
    WallTokenTotals actual;
    int64_t         time_ms = 0;
    send(50, 100, 200);
    // Exceed the SDK's 20 untouched snapshots: a very long active step can let a series expire.
    for (int i = 0; i < 30; ++i)
        accumulateWallSnapshot(monitor, time_ms, actual);
    EXPECT_DOUBLE_EQ((actual[{"50", "rtp_llm_context_wall_tps"}]), 100);
    EXPECT_DOUBLE_EQ((actual[{"50", "rtp_llm_context_wall_tps_with_cache"}]), 200);
    send(39, 0, 300);  // A previously unseen, cache-only priority starts after the gap.
    accumulateWallSnapshot(monitor, time_ms, actual);
    send(50, 400, 800);  // The recycled priority returns without losing or replaying its tokens.
    for (int i = 0; i < 30; ++i)
        accumulateWallSnapshot(monitor, time_ms, actual);
    EXPECT_DOUBLE_EQ((actual[{"50", "rtp_llm_context_wall_tps"}]), 500);
    EXPECT_DOUBLE_EQ((actual[{"50", "rtp_llm_context_wall_tps_with_cache"}]), 1000);
    EXPECT_DOUBLE_EQ((actual[{"39", "rtp_llm_context_wall_tps"}]), 0);
    EXPECT_DOUBLE_EQ((actual[{"39", "rtp_llm_context_wall_tps_with_cache"}]), 300);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ActiveHeartbeatsPreventSdkRecyclingWithoutDilutingOrReplayingTokens) {
    ScopedServingMetrics reporting;
    auto                        monitor = std::make_shared<kmonitor::KMonitor>("wall_heartbeat_test");
    monitor->SetServiceName("");
    auto              reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    auto*             metrics  = reporter->getMetricsGroup<RtpLLMWallClockTokenPSMetrics>();
    std::set<int32_t> known;
    auto              report = [&](RtpLLMTokenPSMetricsCollector& window) {
        for (auto& [p, bucket] : window.priorityCollectorsForReport(known)) {
            kmonitor::MetricsTags tags("priority", std::to_string(p));
            metrics->report(&tags, &bucket);
        }
    };
    RtpLLMTokenPSMetricsCollector first;
    first.addTokenSize(100, 200, 0, 100, 1000);
    first.addPriorityTokenSize(50, 100, 200, 0, 100, 1000);
    first.setReportWindowUs(1000000);
    report(first);
    int64_t         time_ms = 0;
    WallTokenTotals integrated;
    accumulateWallSnapshot(monitor, time_ms, integrated);
    for (int i = 0; i < 30; ++i) {
        RtpLLMTokenPSMetricsCollector heartbeat;
        heartbeat.markIdleWindow();
        report(heartbeat);
        WallTokenTotals sample;
        accumulateWallSnapshot(monitor, time_ms, sample);
        ASSERT_EQ(sample.size(), 2) << "Zeros must remain present beyond the SDK recycling threshold";
        for (const auto& [key, value] : sample)
            EXPECT_DOUBLE_EQ(value, 0);
    }
    RtpLLMTokenPSMetricsCollector finished;
    finished.addTokenSize(3000, 6000, 0, 3000, 300000000);
    finished.addPriorityTokenSize(50, 3000, 6000, 0, 3000, 300000000);
    finished.setReportWindowUs(300000000);
    report(finished);
    // Heartbeats in the same SDK period must add zero, not replace/dilute completed tokens.
    RtpLLMTokenPSMetricsCollector idle;
    idle.markIdleWindow();
    report(idle);
    accumulateWallSnapshot(monitor, time_ms, integrated);
    accumulateWallSnapshot(monitor, time_ms, integrated);
    EXPECT_DOUBLE_EQ((integrated[{"50", "rtp_llm_context_wall_tps"}]), 3100);
    EXPECT_DOUBLE_EQ((integrated[{"50", "rtp_llm_context_wall_tps_with_cache"}]), 6200);
}

}  // namespace rtp_llm
