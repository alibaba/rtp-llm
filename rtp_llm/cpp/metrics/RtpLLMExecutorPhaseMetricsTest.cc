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

// The executor round pipeline is dominated by phases that had no observable
// gauges: the per-stream non-forward gap (~135ms/stream on the cached-prefill
// deployment) could not be attributed between kv_cache_update, the round-end
// device sync, sampler and output dispatch. This test pins the full per-phase
// gauge set so the decomposition stays visible:
//   rtp_llm_kv_cache_update_us, rtp_llm_round_end_sync_us (new),
//   rtp_llm_dispatch_output_us (renamed; the old literal name carried a
//   trailing "_metric" suffix and never matched a query), plus the existing
//   gather/tp_sync/forward/sample gauges.
class RtpLLMExecutorPhaseMetricsTest: public ::testing::Test {
protected:
    // KMonitor snapshots encode every metric through MinMaxCalculator, so the
    // expected value must be encoded the same way (the pattern proven by
    // RtpLLMEplbMetricsTest / RtpLLMTokenPSMetricsCollectorTest).
    using Samples = std::map<std::string, std::string>;

    void SetUp() override {
        monitor_ = std::make_shared<kmonitor::KMonitor>("executor_phase_report_test");
        monitor_->SetServiceName("");
        reporter_ = std::make_shared<kmonitor::MetricsReporter>(monitor_, "", kmonitor::MetricsTags("dp_rank", "0"));
        ASSERT_NE(reporter_->getMetricsGroup<RtpLLMExecutorMetrics>(), nullptr);
    }

    void report(RtpLLMExecutorMetricsCollector& collector) {
        reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
    }

    Samples snapshot() {
        kmonitor::MetricsCollector collector;
        snapshot_time_ms_ += 10000;
        monitor_->GetMetrics(&collector, {kmonitor::NORMAL}, snapshot_time_ms_);
        Samples actual;
        for (const auto* record : collector.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                EXPECT_TRUE(actual.emplace(value->Name(), value->Value()).second)
                    << "duplicate metric in snapshot: " << value->Name();
            }
        }
        return actual;
    }

    void expectMetric(const Samples& actual, const std::string& name, double sample) {
        SCOPED_TRACE(name);
        const auto metric = actual.find(name);
        ASSERT_NE(metric, actual.end()) << "metric missing from snapshot: " << name;
        kmonitor::MinMaxCalculator expected;
        expected.Add(sample);
        EXPECT_EQ(metric->second, expected.ToString());
    }

    std::shared_ptr<kmonitor::KMonitor>        monitor_;
    std::shared_ptr<kmonitor::MetricsReporter> reporter_;
    int64_t                                    snapshot_time_ms_ = 0;
};

TEST_F(RtpLLMExecutorPhaseMetricsTest, ReportsAllRoundPhaseGauges) {
    RtpLLMExecutorMetricsCollector collector;
    collector.context_batch_size  = 2;
    collector.generate_batch_size = 0;
    collector.execute_token_size  = 28000;
    collector.max_seq_len         = 290000;
    // One instrumented prefill-only round: the phases that were previously
    // invisible or misnamed.
    collector.gather_model_input_us = 300;
    collector.tp_sync_input_us      = 550;
    collector.kv_cache_update_us    = 2100;
    collector.model_forward_us      = 159000;
    collector.sample_input_us       = 0;  // prefill-only rounds skip the sampler
    collector.dispatch_output_us    = 4200;
    collector.round_end_sync_us     = 96000;

    report(collector);
    const auto actual = snapshot();

    expectMetric(actual, "rtp_llm_gather_model_input_us", 300);
    expectMetric(actual, "rtp_llm_tp_sync_input_us", 550);
    expectMetric(actual, "rtp_llm_kv_cache_update_us", 2100);
    expectMetric(actual, "rtp_llm_model_forward_us", 159000);
    expectMetric(actual, "rtp_llm_sample_input_us", 0);
    expectMetric(actual, "rtp_llm_dispatch_output_us", 4200);
    expectMetric(actual, "rtp_llm_round_end_sync_us", 96000);
    expectMetric(actual, "rtp_llm_execute_token_size", 28000);
    expectMetric(actual, "rtp_llm_max_seq_len", 290000);
    // The legacy misregistered name must be gone: the dispatch gauge is only
    // published under the corrected name.
    EXPECT_EQ(actual.count("rtp_llm_dispatch_output_us_metric"), 0u);
}

TEST_F(RtpLLMExecutorPhaseMetricsTest, ZeroRoundStillReportsPhaseGauges) {
    // A default-constructed collector (e.g. skip_run or warmup rounds) must
    // still publish every phase gauge so the series never disappear between
    // reporting windows.
    RtpLLMExecutorMetricsCollector collector;
    report(collector);
    const auto actual = snapshot();
    for (const char* const name : {"rtp_llm_kv_cache_update_us", "rtp_llm_round_end_sync_us",
                                   "rtp_llm_dispatch_output_us", "rtp_llm_sample_input_us"}) {
        expectMetric(actual, name, 0.0);
    }
}

}  // namespace
}  // namespace rtp_llm
