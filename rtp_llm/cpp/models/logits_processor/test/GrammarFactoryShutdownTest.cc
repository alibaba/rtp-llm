#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <xgrammar/tokenizer_info.h>
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

namespace rtp_llm {
namespace {

class GrammarFactoryShutdownTest: public ::testing::Test {
protected:
    void SetUp() override {
        // This lifecycle test has no telemetry network or GPU work.
        setenv("kmonitorManuallyMode", "true", 1);
        setenv("kmonitorEnableLogFileSink", "false", 1);
        ASSERT_TRUE(initKmonitorFactory());
    }
    void TearDown() override {
        LogitsProcessorFactory::shutdown();
        stopKmonitorFactory();
    }
    void initialize(const kmonitor::MetricsReporterPtr& reporter) {
        std::vector<std::string> vocab;
        for (int i = 0; i < 128; ++i) {
            vocab.emplace_back(1, static_cast<char>(i));
        }
        GrammarConfig grammar;
        grammar.num_workers = 2;
        grammar.tokenizer_info_json =
            xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
        LogitsProcessorFactory::init(ModelConfig{}, grammar, "", reporter);
    }
};

TEST_F(GrammarFactoryShutdownTest, DropsStaticBackendAndReporterBeforeFactoryShutdown) {
    auto reporter = std::make_shared<kmonitor::MetricsReporter>("", kmonitor::MetricsTags{});
    std::weak_ptr<kmonitor::MetricsReporter> reporter_lease = reporter;
    initialize(reporter);
    std::weak_ptr<XGrammarBackend> backend = LogitsProcessorFactory::grammarBackend();
    ASSERT_FALSE(backend.expired());
    reporter.reset();
    ASSERT_FALSE(reporter_lease.expired()) << "the process-wide grammar backend still owns the reporter";
    LogitsProcessorFactory::shutdown();
    EXPECT_TRUE(backend.expired());
    EXPECT_TRUE(reporter_lease.expired()) << "reporter must release KMonitor before factory deletes it";
    LogitsProcessorFactory::shutdown();
}

TEST_F(GrammarFactoryShutdownTest, NoBackendShutdownIsIdempotent) {
    LogitsProcessorFactory::shutdown();
    LogitsProcessorFactory::shutdown();
    EXPECT_EQ(LogitsProcessorFactory::grammarBackend(), nullptr);
}

TEST_F(GrammarFactoryShutdownTest, FreshEngineCanReinitializeAfterShutdown) {
    for (int i = 0; i < 3; ++i) {
        auto reporter = std::make_shared<kmonitor::MetricsReporter>("", kmonitor::MetricsTags{});
        std::weak_ptr<kmonitor::MetricsReporter> lease = reporter;
        initialize(reporter);
        reporter.reset();
        ASSERT_FALSE(lease.expired());
        LogitsProcessorFactory::shutdown();
        EXPECT_TRUE(lease.expired());
    }
}

}  // namespace
}  // namespace rtp_llm
