#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/test/MockAsyncContext.h"

namespace rtp_llm {
namespace test {

TEST(AsyncContextTest, CompletedAsyncContext_SuccessStatus) {
    CompletedAsyncContext context(ErrorInfo::OkStatus());
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(AsyncContextTest, CompletedAsyncContext_FailureStatusPreservesErrorInfo) {
    CompletedAsyncContext context(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_FALSE(context.errorInfo().ok());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");
}

TEST(AsyncContextTest, FusedAsyncContext_DoneTrue_WhenEmptyOrAllDoneOrNull) {
    auto c1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto c2 = std::shared_ptr<AsyncContext>{nullptr};
    ON_CALL(*c1, done()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused_empty({});
    EXPECT_TRUE(fused_empty.done());

    FusedAsyncContext fused({c1, c2});
    EXPECT_TRUE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_DoneFalse_WhenAnyNotDone) {
    auto done_ctx     = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto not_done_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*not_done_ctx, done()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({done_ctx, not_done_ctx});
    EXPECT_FALSE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessTrue_WhenAllSuccessOrNull) {
    auto ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused({ok, nullptr});
    EXPECT_TRUE(fused.success());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessFalse_WhenAnyFail) {
    auto ok  = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto bad = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));
    ON_CALL(*bad, success()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({ok, bad});
    EXPECT_FALSE(fused.success());
}

}  // namespace test
}  // namespace rtp_llm
