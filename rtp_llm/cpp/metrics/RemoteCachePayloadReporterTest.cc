#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "autil/EnvUtil.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MutableMetric.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include <gtest/gtest.h>
#include <future>
#include <tuple>

namespace rtp_llm {
namespace {

class RemoteCachePayloadReporterTest: public ::testing::Test {
protected:
    autil::EnvGuard manual_{"kmonitorManuallyMode", "true"};
    void            SetUp() override {
        stopKmonitorFactory();
    }
    void TearDown() override {
        stopKmonitorFactory();
    }
};

TEST_F(RemoteCachePayloadReporterTest, DisabledAndMissingRootReporterDoNotStartFactory) {
    EXPECT_EQ(remoteCachePayloadReporter(false, 3, nullptr), nullptr);
    EXPECT_EQ(remoteCachePayloadReporter(true, 0, nullptr), nullptr);
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
}

TEST_F(RemoteCachePayloadReporterTest, ExistingRootReporterIsReusedWithoutFallback) {
    ASSERT_TRUE(initKmonitorFactory());
    auto       reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    const auto count    = kmonitor::KMonitorFactory::GetKMonitorMap()->size();
    EXPECT_EQ(remoteCachePayloadReporter(true, 0, reporter), reporter);
    EXPECT_EQ(remoteCachePayloadReporter(true, 1, reporter), reporter);
    EXPECT_EQ(remoteCachePayloadReporter(false, 1, reporter), nullptr);
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), count);
}

TEST_F(RemoteCachePayloadReporterTest, NonrootGetsOnlyPayloadReporterWithoutChangingGeneralPointer) {
    kmonitor::MetricsReporterPtr general;
    auto                         payload = remoteCachePayloadReporter(true, 3, general);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(general, nullptr);
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 0u);
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), 1u);
}

TEST_F(RemoteCachePayloadReporterTest, ConfiguredFactoryIsStartedWithoutReconfiguration) {
    kmonitor::MetricsConfig config;
    config.set_inited(true);
    config.set_manually_mode(true);
    config.set_use_common_tag(false);
    config.set_service_name("retained-service");
    config.set_sink_address("127.0.0.1:1");
    config.AddGlobalTag("retained", "tag");
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    auto* original = kmonitor::KMonitorFactory::GetConfig();
    auto  payload  = remoteCachePayloadReporter(true, 1, nullptr);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(kmonitor::KMonitorFactory::GetConfig(), original);
    EXPECT_EQ(original->service_name(), "retained-service");
    EXPECT_EQ(original->sink_address(), "127.0.0.1:1");
    EXPECT_EQ(original->global_tags()->FindTag("retained"), "tag");
    EXPECT_FALSE(original->use_common_tag());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 0u);
    EXPECT_TRUE(initKmonitorFactory());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetConfig(), original);
    EXPECT_EQ(original->service_name(), "retained-service");
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 1u);
}

TEST_F(RemoteCachePayloadReporterTest, StartedFactoryAndSinkSurviveAnotherBackendReporter) {
    auto first = remoteCachePayloadReporter(true, 1, nullptr);
    ASSERT_NE(first, nullptr);
    auto* config = kmonitor::KMonitorFactory::GetConfig();
    config->set_service_name("do-not-reinit");
    auto second = remoteCachePayloadReporter(true, 2, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(config, kmonitor::KMonitorFactory::GetConfig());
    EXPECT_EQ(config->service_name(), "do-not-reinit");
    first.reset();
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), 1u);
    RtpLLMRemoteCachePayloadMetricsCollector event;
    event.completed_payload_bytes = 37;
    second->report<RtpLLMRemoteCachePayloadMetrics>(nullptr, &event);
    ASSERT_NE(second->getMetricsGroup<RtpLLMRemoteCachePayloadMetrics>(), nullptr);
}

TEST_F(RemoteCachePayloadReporterTest, RepeatedFullThenPayloadFactoryRestart) {
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_TRUE(initKmonitorFactory());
        EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 1u);
        stopKmonitorFactory();
        {
            auto payload = remoteCachePayloadReporter(true, 1, nullptr);
            ASSERT_NE(payload, nullptr);
            EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 0u);
        }
        stopKmonitorFactory();
    }
}

TEST_F(RemoteCachePayloadReporterTest, ConcurrentNonrootReportersShareOneFactory) {
    std::vector<std::future<kmonitor::MetricsReporterPtr>> pending;
    for (int rank = 1; rank < 8; ++rank) {
        pending.push_back(
            std::async(std::launch::async, [rank] { return remoteCachePayloadReporter(true, rank, nullptr); }));
    }
    std::vector<kmonitor::MetricsReporterPtr> reporters;
    for (auto& future : pending) {
        reporters.push_back(future.get());
        ASSERT_NE(reporters.back(), nullptr);
    }
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->size(), 7u);
    EXPECT_EQ(kmonitor::KMonitorFactory::GetKMonitorMap()->count("buildinmetric"), 0u);
    reporters.clear();
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    EXPECT_TRUE(kmonitor::KMonitorFactory::GetKMonitorMap()->empty());
}

TEST_F(RemoteCachePayloadReporterTest, UnequalRankPayloadsCountOnceAndKeepSuccessfulPrefixBeforeFailure) {
    // These are reporter/SDK event semantics. Actual backend installation and
    // model CP execution are separate gates; do not multiply a root total by CP.
    for (const bool replicated : {false, true}) {
        for (int cp : {2, 4, 8}) {
            ASSERT_TRUE(initKmonitorFactory());
            auto    root         = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
            int64_t actual_sum   = 0;
            int64_t expected_sum = 0;
            for (int rank = 0; rank < cp; ++rank) {
                SCOPED_TRACE(::testing::PrintToString(std::make_tuple(replicated, cp, rank)));
                auto payload = remoteCachePayloadReporter(true, rank, rank == 0 ? root : nullptr);
                ASSERT_NE(payload, nullptr);
                kmonitor::MetricsTags tags("mode", "write");
                tags.AddTag("tp_rank", std::to_string(rank));
                tags.AddTag("dp_rank", "0");
                const int64_t first  = (replicated ? 100 : 20) + rank;
                const int64_t second = 7 + rank;
                for (int64_t bytes : {first, second}) {
                    RtpLLMRemoteCachePayloadMetricsCollector event;
                    event.completed_payload_bytes = bytes;
                    payload->report<RtpLLMRemoteCachePayloadMetrics>(&tags, &event);
                }
                RtpLLMRemoteCachePayloadMetricsCollector failure;
                failure.completion_unknown = true;
                payload->report<RtpLLMRemoteCachePayloadMetrics>(&tags, &failure);
                auto* group = payload->getMetricsGroup<RtpLLMRemoteCachePayloadMetrics>();
                ASSERT_NE(group, nullptr);
                const auto snapshot = [&](kmonitor::MutableMetric* metric) {
                    auto* series = metric->DeclareMetric(&tags);
                    if (!series) {
                        ADD_FAILURE() << "missing payload series";
                        return -1.0;
                    }
                    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
                    series->Snapshot(&record, /*period_ms=*/1000);
                    EXPECT_TRUE(metric->UndeclareMetric(series));
                    if (record.Values().size() != 1) {
                        ADD_FAILURE() << "unexpected payload sample count";
                        return -1.0;
                    }
                    return std::stod(record.Values().front()->Value());
                };
                const double completed = snapshot(group->completed_payload_bytes_qps_metric);
                EXPECT_DOUBLE_EQ(completed, first + second);
                EXPECT_DOUBLE_EQ(snapshot(group->completion_unknown_qps_metric), 1);
                actual_sum += static_cast<int64_t>(completed);
                expected_sum += first + second;
            }
            EXPECT_EQ(actual_sum, expected_sum);
            root.reset();
            stopKmonitorFactory();
        }
    }
}

}  // namespace
}  // namespace rtp_llm
