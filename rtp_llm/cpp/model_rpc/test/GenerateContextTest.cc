#include <chrono>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

namespace rtp_llm {

class GenerateContextTest: public ::testing::Test {
protected:
    class Context: public GenerateContext {
    public:
        Context(int64_t timeout_ms, kmonitor::MetricsReporterPtr& reporter):
            GenerateContext(1, timeout_ms, nullptr, reporter, nullptr) {}

        using GenerateContext::request_begin_time_;
    };

    kmonitor::MetricsReporterPtr reporter_;
};

TEST_F(GenerateContextTest, RetryResetDoesNotRenewRequestDeadline) {
    Context context(1000, reporter_);
    const auto deadline = context.request_deadline;
    context.reset();
    EXPECT_EQ(context.request_deadline, deadline);
    EXPECT_EQ(context.streamRpcDeadline(), deadline);
}

TEST_F(GenerateContextTest, TimeoutUpdateUsesOriginalRequestStart) {
    Context context(0, reporter_);
    context.request_begin_time_ = std::chrono::system_clock::now() - std::chrono::seconds(2);
    context.setRequestTimeoutMs(1000);
    EXPECT_EQ(context.request_deadline, context.request_begin_time_ + std::chrono::milliseconds(1000));
    EXPECT_TRUE(context.requestDeadlineExceeded());
}

TEST_F(GenerateContextTest, RpcDeadlineUsesEarlierLimit) {
    Context context(60000, reporter_);
    const auto before = std::chrono::system_clock::now();
    const auto deadline = context.streamRpcDeadline(10);
    const auto after = std::chrono::system_clock::now();
    ASSERT_TRUE(deadline.has_value());
    EXPECT_GE(*deadline, before + std::chrono::milliseconds(10));
    EXPECT_LE(*deadline, after + std::chrono::milliseconds(10));
    EXPECT_LT(*deadline, *context.request_deadline);

    context.request_deadline = before - std::chrono::milliseconds(1);
    EXPECT_EQ(context.streamRpcDeadline(60000), context.request_deadline);
}

TEST_F(GenerateContextTest, DisabledLimitsKeepOtherDeadline) {
    Context context(1000, reporter_);
    EXPECT_EQ(context.streamRpcDeadline(0), context.request_deadline);
    EXPECT_EQ(context.streamRpcDeadline(-1), context.request_deadline);
    context.setRequestTimeoutMs(0);
    EXPECT_FALSE(context.requestDeadlineExceeded());
    EXPECT_FALSE(context.streamRpcDeadline(0).has_value());
    EXPECT_TRUE(context.streamRpcDeadline(1000).has_value());
    context.setRequestTimeoutMs(-1);
    EXPECT_FALSE(context.streamRpcDeadline(-1).has_value());
}

}  // namespace rtp_llm
