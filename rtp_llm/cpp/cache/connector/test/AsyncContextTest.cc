#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"

namespace rtp_llm {
namespace test {

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

TEST(AsyncContextTest, FusedAsyncReadContext_DoneTrue_WhenMatchContextNull) {
    auto resource = std::shared_ptr<KVCacheResourceV1>{};  // not used by logic
    auto ctx      = std::make_shared<FusedAsyncReadContext>(nullptr, resource);
    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());  // fused_match_context_ is null => success() must be false
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneFalse_WhenMatchNotDone) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(false));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResourceV1>{});

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneTrue_WhenMatchDoneButFailed) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(false));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResourceV1>{});

    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneFalse_WhenMatchSuccessButReadNotSet) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResourceV1>{});

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_SuccessDependsOnReadContext) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto read_ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_ok, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_ok, success()).WillByDefault(testing::Return(true));
    auto read_fused = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_ok});

    auto ctx = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResourceV1>{});
    ctx->setFusedReadContext(read_fused);

    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());

    // Now make read context fail while done.
    auto read_bad = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_bad, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_bad, success()).WillByDefault(testing::Return(false));
    ctx->setFusedReadContext(std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_bad}));

    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

}  // namespace test
}  // namespace rtp_llm
