#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackAsyncContext.h"

#include <memory>

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(LoadBackAsyncContextTest, CompleteSingleWork) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(1);
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(context->done());

    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    context->waitDone();
}

TEST(LoadBackAsyncContextTest, CompleteAfterEveryTransfer) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(false));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST(LoadBackAsyncContextTest, CancellationWaitsForEveryTransfer) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(2);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST(LoadBackAsyncContextTest, RejectCompletionAfterTerminalState) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(1);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->completeOne(true));
}

TEST(LoadBackAsyncContextTest, TaskFailureCompletesAllPendingTransfers) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->completeOne(true));
    EXPECT_FALSE(context->onTaskFail());
}

TEST(LoadBackAsyncContextTest, TaskFailureCompletesCanceledContext) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(2);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_FALSE(context->success());
}

TEST(LoadBackAsyncContextTest, ZeroTransferContextIsImmediatelySuccessful) {
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(0);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->completeOne(true));
}

}  // namespace rtp_llm
