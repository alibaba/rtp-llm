#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <memory>

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(LoadAsyncContextTest, CompleteSingleWork) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(1);
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(context->done());

    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    context->waitDone();
}

TEST(LoadAsyncContextTest, CompleteAfterEveryTransfer) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(false));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST(LoadAsyncContextTest, CancellationWaitsForEveryTransfer) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(2);
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

TEST(LoadAsyncContextTest, RejectCompletionAfterTerminalState) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(1);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->completeOne(true));
}

TEST(LoadAsyncContextTest, TaskFailureCompletesAllPendingTransfers) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->completeOne(true));
    EXPECT_FALSE(context->onTaskFail());
}

TEST(LoadAsyncContextTest, TaskFailureCompletesCanceledContext) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(2);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_FALSE(context->success());
}

TEST(LoadAsyncContextTest, ZeroTransferContextIsImmediatelySuccessful) {
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(0);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->completeOne(true));
}

}  // namespace rtp_llm
