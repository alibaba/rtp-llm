#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class RtpLLMSpeculativeMetricsTest: public ::testing::Test {
protected:
    using Samples   = std::map<std::string, std::vector<double>>;
    using MetricKey = std::pair<std::string, std::string>;

    void SetUp() override {
        monitor_ = std::make_shared<kmonitor::KMonitor>("speculative_report_test");
        monitor_->SetServiceName("");
    }

    void report(RtpLLMSpeculativeEngineMetricsCollector& collector, const std::string& dp_rank = "0") {
        auto& reporter = reporters_[dp_rank];
        if (!reporter) {
            reporter =
                std::make_shared<kmonitor::MetricsReporter>(monitor_, "", kmonitor::MetricsTags("dp_rank", dp_rank));
        }
        reporter->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(nullptr, &collector);
    }

    static Samples latencySamples(double step_latency_us = 0) {
        return {{"rtp_llm_sp_step_latency_us", {step_latency_us}},
                {"rtp_llm_sp_propose_step_latency_us", {0}},
                {"rtp_llm_sp_score_step_latency_us", {0}},
                {"rtp_llm_sp_speculative_sampler_latency_us", {0}}};
    }

    void expectSnapshot(const std::map<std::string, Samples>& expected_by_rank) {
        kmonitor::MetricsCollector snapshot;
        snapshot_time_ms_ += 10000;
        monitor_->GetMetrics(&snapshot, {kmonitor::NORMAL}, snapshot_time_ms_);
        std::map<MetricKey, std::string> actual;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                MetricKey key{record->Tags()->FindTag("dp_rank"), value->Name()};
                EXPECT_TRUE(actual.emplace(key, value->Value()).second);
            }
        }
        size_t expected_count = 0;
        for (const auto& [rank, metrics] : expected_by_rank) {
            expected_count += metrics.size();
            for (const auto& [name, samples] : metrics) {
                SCOPED_TRACE(rank + ":" + name);
                ASSERT_FALSE(samples.empty());
                const auto metric = actual.find(MetricKey{rank, name});
                ASSERT_NE(metric, actual.end());
                kmonitor::MinMaxCalculator expected;
                for (double sample : samples) {
                    expected.Add(sample);
                }
                EXPECT_EQ(metric->second, expected.ToString());
            }
        }
        EXPECT_EQ(actual.size(), expected_count) << ::testing::PrintToString(actual);
    }

private:
    kmonitor::KMonitorPtr                               monitor_;
    std::map<std::string, kmonitor::MetricsReporterPtr> reporters_;
    int64_t                                             snapshot_time_ms_ = 0;
};

TEST_F(RtpLLMSpeculativeMetricsTest, SkipsUnsetDecisionAndEmptyAcceptance) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.total_verify_draft_token_num = 0;
    report(collector);
    expectSnapshot({{"0", latencySamples()}});
    expectSnapshot({});
}

TEST_F(RtpLLMSpeculativeMetricsTest, ReportsZeroDecisionBeforeAcceptanceIsReady) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.verify_budget           = 0;
    collector.decided_draft_token_num = 0;
    report(collector);

    auto expected = latencySamples();
    expected.insert({{"rtp_llm_sp_verify_budget", {0}},
                     {"rtp_llm_sp_decided_draft_token_num", {0}},
                     {"rtp_llm_sp_acceptance_metrics_dropped", {0}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, UsesCompletedBudgetInsteadOfProposalWidth) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.step_latency_us              = 120;
    collector.spec_steps                   = 5;
    collector.total_stream_num             = 2;
    collector.total_propose_token_num      = 10;
    collector.total_accepted_token_num     = 6;
    collector.verify_budget                = 3;
    collector.decided_draft_token_num      = 6;
    collector.total_verify_draft_token_num = 6;
    report(collector);

    auto expected = latencySamples(120);
    expected.insert({{"rtp_llm_sp_verify_budget", {3}},
                     {"rtp_llm_sp_decided_draft_token_num", {6}},
                     {"rtp_llm_sp_acceptance_metrics_dropped", {0}},
                     {"rtp_llm_sp_total_verify_draft_token_num", {6}},
                     {"rtp_llm_sp_total_propose_token_num", {10}},
                     {"rtp_llm_sp_total_accepted_token_num", {6}},
                     {"rtp_llm_sp_avg_accept_token_num", {3}},
                     {"rtp_llm_sp_avg_accept_rate", {0.75}},
                     {"rtp_llm_sp_avg_fix_accept_rate", {2.0 / 3.0}},
                     {"rtp_llm_sp_estimate_tpot_us", {40}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, ReportsFullWidthAcceptance) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.step_latency_us              = 120;
    collector.spec_steps                   = 5;
    collector.total_stream_num             = 2;
    collector.total_propose_token_num      = 10;
    collector.total_accepted_token_num     = 12;
    collector.verify_budget                = 5;
    collector.decided_draft_token_num      = 10;
    collector.total_verify_draft_token_num = 10;
    report(collector);

    auto expected = latencySamples(120);
    expected.insert({{"rtp_llm_sp_verify_budget", {5}},
                     {"rtp_llm_sp_decided_draft_token_num", {10}},
                     {"rtp_llm_sp_acceptance_metrics_dropped", {0}},
                     {"rtp_llm_sp_total_verify_draft_token_num", {10}},
                     {"rtp_llm_sp_total_propose_token_num", {10}},
                     {"rtp_llm_sp_total_accepted_token_num", {12}},
                     {"rtp_llm_sp_avg_accept_token_num", {6}},
                     {"rtp_llm_sp_avg_accept_rate", {1}},
                     {"rtp_llm_sp_avg_fix_accept_rate", {1}},
                     {"rtp_llm_sp_estimate_tpot_us", {20}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, ZeroCompletedBudgetDoesNotEmitUndefinedDraftAcceptanceRate) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.spec_steps                   = 5;
    collector.total_stream_num             = 2;
    collector.total_propose_token_num      = 10;
    collector.total_accepted_token_num     = 2;
    collector.total_verify_draft_token_num = 0;
    report(collector);

    auto expected = latencySamples();
    expected.insert({{"rtp_llm_sp_total_verify_draft_token_num", {0}},
                     {"rtp_llm_sp_total_propose_token_num", {10}},
                     {"rtp_llm_sp_total_accepted_token_num", {2}},
                     {"rtp_llm_sp_avg_accept_token_num", {1}},
                     {"rtp_llm_sp_avg_accept_rate", {1}},
                     {"rtp_llm_sp_estimate_tpot_us", {0}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, CurrentDecisionDoesNotChangeAggregatedCompletedDenominator) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.spec_steps = 5;
    // Two completed rounds: (B=1, V=0, A=1) and (B=2, V=8, A=7).
    collector.total_stream_num             = 3;
    collector.total_propose_token_num      = 15;
    collector.total_accepted_token_num     = 8;
    collector.total_verify_draft_token_num = 8;
    // The current round has a different batch and full-width budget.
    collector.verify_budget              = 5;
    collector.decided_draft_token_num    = 20;
    collector.acceptance_metrics_dropped = 1;
    report(collector);

    auto expected = latencySamples();
    expected.insert({{"rtp_llm_sp_verify_budget", {5}},
                     {"rtp_llm_sp_decided_draft_token_num", {20}},
                     {"rtp_llm_sp_acceptance_metrics_dropped", {1}},
                     {"rtp_llm_sp_total_verify_draft_token_num", {8}},
                     {"rtp_llm_sp_total_propose_token_num", {15}},
                     {"rtp_llm_sp_total_accepted_token_num", {8}},
                     {"rtp_llm_sp_avg_accept_token_num", {8.0 / 3.0}},
                     {"rtp_llm_sp_avg_accept_rate", {8.0 / 11.0}},
                     {"rtp_llm_sp_avg_fix_accept_rate", {0.625}},
                     {"rtp_llm_sp_estimate_tpot_us", {0}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, LegacyCallersKeepConfiguredWidthDenominator) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.step_latency_us          = 120;
    collector.spec_steps               = 5;
    collector.total_stream_num         = 2;
    collector.total_propose_token_num  = 10;
    collector.total_accepted_token_num = 6;
    report(collector);

    auto expected = latencySamples(120);
    expected.insert({{"rtp_llm_sp_total_propose_token_num", {10}},
                     {"rtp_llm_sp_total_accepted_token_num", {6}},
                     {"rtp_llm_sp_avg_accept_token_num", {3}},
                     {"rtp_llm_sp_avg_accept_rate", {0.5}},
                     {"rtp_llm_sp_avg_fix_accept_rate", {0.4}},
                     {"rtp_llm_sp_estimate_tpot_us", {40}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, ZeroGeneratedDraftsDoNotSuppressCompletedAcceptance) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    collector.total_stream_num         = 2;
    collector.total_accepted_token_num = 2;
    report(collector);

    auto expected = latencySamples();
    expected.insert({{"rtp_llm_sp_total_propose_token_num", {0}},
                     {"rtp_llm_sp_total_accepted_token_num", {2}},
                     {"rtp_llm_sp_avg_accept_token_num", {1}},
                     {"rtp_llm_sp_avg_accept_rate", {1}},
                     {"rtp_llm_sp_estimate_tpot_us", {0}}});
    expectSnapshot({{"0", expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, DifferentDPRanksKeepIndependentBudgetsAndAcceptance) {
    RtpLLMSpeculativeEngineMetricsCollector zero;
    zero.spec_steps                   = 5;
    zero.verify_budget                = 0;
    zero.decided_draft_token_num      = 0;
    zero.total_stream_num             = 1;
    zero.total_propose_token_num      = 5;
    zero.total_accepted_token_num     = 1;
    zero.total_verify_draft_token_num = 0;
    report(zero, "0");

    RtpLLMSpeculativeEngineMetricsCollector bounded;
    bounded.spec_steps                   = 5;
    bounded.verify_budget                = 3;
    bounded.decided_draft_token_num      = 6;
    bounded.total_stream_num             = 2;
    bounded.total_propose_token_num      = 10;
    bounded.total_accepted_token_num     = 6;
    bounded.total_verify_draft_token_num = 6;
    report(bounded, "1");

    auto zero_expected = latencySamples();
    zero_expected.insert({{"rtp_llm_sp_verify_budget", {0}},
                          {"rtp_llm_sp_decided_draft_token_num", {0}},
                          {"rtp_llm_sp_acceptance_metrics_dropped", {0}},
                          {"rtp_llm_sp_total_verify_draft_token_num", {0}},
                          {"rtp_llm_sp_total_propose_token_num", {5}},
                          {"rtp_llm_sp_total_accepted_token_num", {1}},
                          {"rtp_llm_sp_avg_accept_token_num", {1}},
                          {"rtp_llm_sp_avg_accept_rate", {1}},
                          {"rtp_llm_sp_estimate_tpot_us", {0}}});
    auto bounded_expected = latencySamples();
    bounded_expected.insert({{"rtp_llm_sp_verify_budget", {3}},
                             {"rtp_llm_sp_decided_draft_token_num", {6}},
                             {"rtp_llm_sp_acceptance_metrics_dropped", {0}},
                             {"rtp_llm_sp_total_verify_draft_token_num", {6}},
                             {"rtp_llm_sp_total_propose_token_num", {10}},
                             {"rtp_llm_sp_total_accepted_token_num", {6}},
                             {"rtp_llm_sp_avg_accept_token_num", {3}},
                             {"rtp_llm_sp_avg_accept_rate", {0.75}},
                             {"rtp_llm_sp_avg_fix_accept_rate", {2.0 / 3.0}},
                             {"rtp_llm_sp_estimate_tpot_us", {0}}});
    expectSnapshot({{"0", zero_expected}, {"1", bounded_expected}});
}

TEST_F(RtpLLMSpeculativeMetricsTest, GaugesRetainZeroBudgetAndDroppedSampleValues) {
    RtpLLMSpeculativeEngineMetricsCollector collector;
    for (int64_t budget : {0, 3, 5}) {
        collector.verify_budget              = budget;
        collector.decided_draft_token_num    = 2 * budget;
        collector.acceptance_metrics_dropped = budget == 3 ? 1 : 0;
        report(collector);
    }

    Samples expected{{"rtp_llm_sp_step_latency_us", {0, 0, 0}},
                     {"rtp_llm_sp_propose_step_latency_us", {0, 0, 0}},
                     {"rtp_llm_sp_score_step_latency_us", {0, 0, 0}},
                     {"rtp_llm_sp_speculative_sampler_latency_us", {0, 0, 0}},
                     {"rtp_llm_sp_verify_budget", {0, 3, 5}},
                     {"rtp_llm_sp_decided_draft_token_num", {0, 6, 10}},
                     {"rtp_llm_sp_acceptance_metrics_dropped", {0, 1, 0}}};
    expectSnapshot({{"0", expected}});
}

}  // namespace
}  // namespace rtp_llm
