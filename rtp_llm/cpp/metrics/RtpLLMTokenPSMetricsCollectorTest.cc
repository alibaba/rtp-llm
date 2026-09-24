#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

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
                        const std::string&                             priority = "30") {
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
    kmonitor::KMonitorPtr        monitor_;
    kmonitor::MetricsReporterPtr reporter_;
    RtpLLMTokenPSMetrics*        metrics_          = nullptr;
    int64_t                     snapshot_time_ms_ = 0;
};

struct TokenDeltaReport {
    std::string priority;
    int64_t     context_tokens;
    int64_t     context_tokens_with_cache;
};

class RecordingTokenDeltaMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager*) override {
        return true;
    }

    void report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector) {
        std::lock_guard<std::mutex> lock(mutex_);
        TokenDeltaReport           report{
            tags->FindTag("priority"), collector->contextTokensDelta(), collector->contextTokensWithCacheDelta()};
        last_report_was_idle_ = collector->reportZeroTPS();
        if (last_report_was_idle_) {
            idle_report_ = report;
        } else {
            data_reports_.push_back(report);
        }
        cv_.notify_all();
    }

    bool waitForIdleAfter(size_t count, std::vector<TokenDeltaReport>& reports, TokenDeltaReport& idle_report) {
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
    std::vector<TokenDeltaReport> data_reports_;
    TokenDeltaReport              idle_report_{};
    bool                          last_report_was_idle_ = false;
};

template<typename Reporter>
void checkReporterResetsTokenDeltas() {
    auto  monitor           = std::make_shared<kmonitor::KMonitor>("token_delta_reset_test");
    auto  metrics_reporter  = std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags());
    auto* recording_metrics = metrics_reporter->getMetricsGroup<RecordingTokenDeltaMetrics>();
    ASSERT_NE(recording_metrics, nullptr);
    Reporter reporter(metrics_reporter, 1);

    using CountsByPriority = std::map<std::string, std::pair<int64_t, int64_t>>;
    auto expect_window = [&](const CountsByPriority& expected) {
        std::vector<TokenDeltaReport> reports;
        TokenDeltaReport              idle_report;
        ASSERT_TRUE(recording_metrics->waitForIdleAfter(expected.size(), reports, idle_report))
            << "Reporter did not drain its data and emit an idle window";
        ASSERT_EQ(reports.size(), expected.size());
        CountsByPriority actual;
        for (const auto& report : reports) {
            EXPECT_TRUE(actual.emplace(report.priority,
                                       std::make_pair(report.context_tokens, report.context_tokens_with_cache))
                            .second);
        }
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(idle_report.priority, "0");
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

    // Exercise the untagged path with no other metric making this collector non-empty.
    RtpLLMTokenPSMetricsCollector untimed;
    untimed.addTokenSize(10, 15, 0, 0, 0);
    reporter.report(&untimed);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"0", {10, 15}}}));

    RtpLLMTokenPSMetricsCollector cache_only;
    cache_only.addTokenSize(0, 25, 0, 0, 0);
    cache_only.addPriorityTokenSize(70, 0, 25, 0, 0, 0);
    reporter.report(&cache_only);
    ASSERT_NO_FATAL_FAILURE(expect_window({{"70", {0, 25}}}));
}

}  // namespace

TEST_F(RtpLLMTokenPSMetricsReportTest, DoesNotRepeatTpsAfterUntimedTokenDeltas) {
    RtpLLMTokenPSMetricsCollector prefill;
    prefill.addTokenSize(100, 150, 0, 100, 1000000);
    report(prefill);
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_delta", {100}},
                                           {"rtp_llm_context_tokens_with_cache_delta", {150}},
                                           {"rtp_llm_context_tps", {100}},
                                           {"rtp_llm_context_tps_with_cache", {150}},
                                           {"rtp_llm_total_tps", {100}}}));

    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        // Deliberately omit total to exercise the delta-only collector API boundary.
        RtpLLMTokenPSMetricsCollector untimed;
        untimed.addTokenSize(20, 30, 0, 0, execute_time_us);
        untimed.markIdleWindow();
        report(untimed);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_delta", {20}},
                                               {"rtp_llm_context_tokens_with_cache_delta", {30}}}));
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
    }

    RtpLLMTokenPSMetricsCollector idle;
    idle.markIdleWindow();
    report(idle, "0");
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_delta", {0}},
                                           {"rtp_llm_context_tokens_with_cache_delta", {0}},
                                           {"rtp_llm_context_tps", {0}},
                                           {"rtp_llm_context_tps_with_cache", {0}},
                                           {"rtp_llm_generate_tps", {0}},
                                           {"rtp_llm_total_tps", {0}}},
                                          "0"));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, ReportsCacheOnlyDeltaWithoutInventingTpsSamples) {
    RtpLLMTokenPSMetricsCollector cache_only;
    cache_only.addTokenSize(0, 25, 0, 0, 0);
    cache_only.markIdleWindow();
    report(cache_only);
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_with_cache_delta", {25}}}));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST_F(RtpLLMTokenPSMetricsReportTest, KeepsDecodeAndUntimedTotalCountsWithoutContextTps) {
    for (int64_t execute_time_us : {1000000, 0, -1}) {
        SCOPED_TRACE(execute_time_us);
        RtpLLMTokenPSMetricsCollector decode;
        decode.addTokenSize(0, 0, 7, 7, execute_time_us);
        report(decode);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_generate_tps", {7}}, {"rtp_llm_total_tps", {7}}}));
    }

    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        // Production prefill counts the same executed tokens in context and total.
        RtpLLMTokenPSMetricsCollector untimed;
        untimed.addTokenSize(20, 30, 0, 20, execute_time_us);
        report(untimed);
        ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_delta", {20}},
                                               {"rtp_llm_context_tokens_with_cache_delta", {30}},
                                               {"rtp_llm_total_tps", {20}}}));
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
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({{"rtp_llm_context_tokens_delta", {100, 20}},
                                           {"rtp_llm_context_tokens_with_cache_delta", {150, 30}},
                                           {"rtp_llm_context_tps", {100}},
                                           {"rtp_llm_context_tps_with_cache", {150}},
                                           {"rtp_llm_generate_tps", {7}},
                                           {"rtp_llm_total_tps", {100, 7}}}));
    ASSERT_NO_FATAL_FAILURE(expectSnapshot({}));
}

TEST(RtpLLMTokenPSMetricsCollectorTest, LoopReporterResetsTokenDeltasAfterReport) {
    checkReporterResetsTokenDeltas<MetricsLoopReporter<RecordingTokenDeltaMetrics, RtpLLMTokenPSMetricsCollector>>();
}

TEST(RtpLLMTokenPSMetricsCollectorTest, WallClockReporterResetsTokenDeltasAfterReport) {
    checkReporterResetsTokenDeltas<
        WallClockMetricsLoopReporter<RecordingTokenDeltaMetrics, RtpLLMTokenPSMetricsCollector>>();
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
    EXPECT_EQ(merged.contextTokensDelta(), 10000);
    EXPECT_EQ(merged.contextTokensWithCacheDelta(), 10000);
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
    EXPECT_EQ(priority_collectors.at(30).contextTokensDelta(), 400);
    EXPECT_EQ(priority_collectors.at(50).contextTokensDelta(), 600);
    EXPECT_EQ(priority_collectors.at(30).contextTokensWithCacheDelta(), 600);
    EXPECT_EQ(priority_collectors.at(50).contextTokensWithCacheDelta(), 900);
    EXPECT_EQ(merged.contextTokensDelta(), 1000);
    EXPECT_EQ(merged.contextTokensWithCacheDelta(), 1500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalAsTokenCounts) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1500, 10, 1010, 100 * 1000);

    EXPECT_NEAR(collector.contextTPS(), 10000.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 15000.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 10.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 1010.0, 1e-6);
    EXPECT_EQ(collector.contextTokensDelta(), 1000);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 1500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsGenerateAndTotalWhenExecutionTimeIsZero) {
    RtpLLMTokenPSMetricsCollector collector;

    collector.addTokenSize(1000, 1000, 2, 1002, 0);

    EXPECT_FALSE(collector.hasContextTPS());
    EXPECT_FALSE(collector.hasContextTPSWithCache());
    EXPECT_TRUE(collector.hasGenerateTPS());
    EXPECT_TRUE(collector.hasTotalTPS());
    EXPECT_NEAR(collector.generateTPS(), 2.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 1002.0, 1e-6);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MarksEmptyIdleWindowForZeroReport) {
    RtpLLMTokenPSMetricsCollector collector;

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());

    collector.markIdleWindow();

    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_TRUE(collector.reportZeroTPS());
    EXPECT_NEAR(collector.contextTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.contextTPSWithCache(), 0.0, 1e-6);
    EXPECT_NEAR(collector.generateTPS(), 0.0, 1e-6);
    EXPECT_NEAR(collector.totalTPS(), 0.0, 1e-6);
    EXPECT_EQ(collector.contextTokensDelta(), 0);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 0);
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
    EXPECT_EQ(collector.contextTokensDelta(), 0);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 0);
    for (const auto& [priority, priority_collector] : priority_collectors) {
        EXPECT_EQ(priority_collector.contextTokensDelta(), 0);
        EXPECT_EQ(priority_collector.contextTokensWithCacheDelta(), 0);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, ContextTokenDeltasDoNotDependOnExecutionOrReportWindow) {
    RtpLLMTokenPSMetricsCollector collector;
    collector.addTokenSize(1000, 1500, 0, 1000, 100 * 1000);
    collector.addTokenSize(3000, 4500, 0, 3000, 900 * 1000);
    collector.setReportWindowUs(2 * 1000 * 1000);

    EXPECT_EQ(collector.contextTokensDelta(), 4000);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 6000);
    EXPECT_DOUBLE_EQ(collector.contextTPS(), 4000.0);
    EXPECT_DOUBLE_EQ(collector.contextTPSWithCache(), 6000.0);
    EXPECT_DOUBLE_EQ(collector.contextWallTPS(), 2000.0);
    EXPECT_DOUBLE_EQ(collector.contextWallTPSWithCache(), 3000.0);

    collector.setReportWindowUs(10 * 1000 * 1000);
    EXPECT_EQ(collector.contextTokensDelta(), 4000);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 6000);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsContextTokenDeltasWithoutValidTiming) {
    for (int64_t execute_time_us : {0, -1}) {
        SCOPED_TRACE(execute_time_us);
        RtpLLMTokenPSMetricsCollector collector;
        // No other metric can make this window non-empty.
        collector.addTokenSize(1000, 1500, 0, 0, execute_time_us);
        collector.addPriorityTokenSize(30, 1000, 1500, 0, 0, execute_time_us);
        collector.markIdleWindow();

        EXPECT_EQ(collector.contextTokensDelta(), 1000);
        EXPECT_EQ(collector.contextTokensWithCacheDelta(), 1500);
        EXPECT_TRUE(collector.hasContextTokensDelta());
        EXPECT_TRUE(collector.hasContextTokensWithCacheDelta());
        EXPECT_TRUE(collector.hasMetrics());
        EXPECT_FALSE(collector.reportZeroTPS());
        EXPECT_FALSE(collector.hasContextTPS());
        EXPECT_FALSE(collector.hasContextTPSWithCache());

        auto priority_collectors = collector.priorityCollectorsForReport();
        ASSERT_EQ(priority_collectors.size(), 1);
        EXPECT_EQ(priority_collectors.at(30).contextTokensDelta(), 1000);
        EXPECT_EQ(priority_collectors.at(30).contextTokensWithCacheDelta(), 1500);
        EXPECT_FALSE(priority_collectors.at(30).hasContextTPS());
        EXPECT_FALSE(priority_collectors.at(30).hasContextTPSWithCache());

        RtpLLMTokenPSMetricsCollector merged;
        merged.markIdleWindow();
        merged.merge(&collector);
        EXPECT_TRUE(merged.hasMetrics());
        EXPECT_FALSE(merged.reportZeroTPS());
        EXPECT_EQ(merged.contextTokensDelta(), 1000);
        EXPECT_EQ(merged.contextTokensWithCacheDelta(), 1500);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, MergingUntimedDeltasPreservesTimeValidatedTps) {
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
        EXPECT_EQ(collector->contextTokensDelta(), 1000);
        EXPECT_EQ(collector->contextTokensWithCacheDelta(), 1400);
        EXPECT_DOUBLE_EQ(collector->contextTPS(), 7000.0);
        EXPECT_DOUBLE_EQ(collector->contextTPSWithCache(), 9000.0);
        EXPECT_DOUBLE_EQ(collector->contextWallTPS(), 350.0);
        EXPECT_DOUBLE_EQ(collector->contextWallTPSWithCache(), 450.0);
    }
}

TEST(RtpLLMTokenPSMetricsCollectorTest, KeepsCacheOnlyDeltaWithoutValidTiming) {
    RtpLLMTokenPSMetricsCollector collector;
    collector.addTokenSize(0, 500, 0, 0, 0);
    collector.addPriorityTokenSize(30, 0, 500, 0, 0, 0);
    collector.markIdleWindow();

    EXPECT_EQ(collector.contextTokensDelta(), 0);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 500);
    EXPECT_FALSE(collector.hasContextTokensDelta());
    EXPECT_TRUE(collector.hasContextTokensWithCacheDelta());
    EXPECT_TRUE(collector.hasMetrics());
    EXPECT_FALSE(collector.reportZeroTPS());
    EXPECT_EQ(collector.priorityCollectorsForReport().at(30).contextTokensWithCacheDelta(), 500);
}

TEST(RtpLLMTokenPSMetricsCollectorTest, IgnoresNonPositiveContextTokenCounts) {
    RtpLLMTokenPSMetricsCollector collector;
    for (int64_t token_num : {0, -1}) {
        collector.addTokenSize(token_num, token_num, 0, 0, 100 * 1000);
        collector.addPriorityTokenSize(30, token_num, token_num, 0, 0, 100 * 1000);
    }

    EXPECT_EQ(collector.contextTokensDelta(), 0);
    EXPECT_EQ(collector.contextTokensWithCacheDelta(), 0);
    EXPECT_FALSE(collector.hasMetrics());
    EXPECT_EQ(collector.priorityCollectorsForReport().at(30).contextTokensDelta(), 0);
    EXPECT_EQ(collector.priorityCollectorsForReport().at(30).contextTokensWithCacheDelta(), 0);
}

}  // namespace rtp_llm
