#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {

TEST(TransferBatchAsyncContextTest, WaitersWakeAfterCompletion) {
    auto context = std::make_shared<TransferBatchAsyncContext>();
    std::atomic<size_t> completed_waiters{0};
    std::thread first([&] {
        context->waitDone();
        ++completed_waiters;
    });
    std::thread second([&] {
        context->waitDone();
        ++completed_waiters;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(completed_waiters.load(), 0u);
    context->complete(ErrorInfo::OkStatus());
    first.join();
    second.join();

    EXPECT_EQ(completed_waiters.load(), 2u);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
}

TEST(TransferBatchAsyncContextTest, FailureIsVisibleAfterWait) {
    TransferBatchAsyncContext context;
    context.complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "bad descriptor"));
    context.waitDone();

    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
}

TEST(TransferBatchAsyncContextTest, FirstCompletionWins) {
    TransferBatchAsyncContext context;
    context.complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "first failure"));
    context.complete(ErrorInfo::OkStatus());

    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
}

TEST(TransferBatchAsyncContextTest, CompletionReleasesGuard) {
    auto                      guard      = std::make_shared<int>(1);
    std::weak_ptr<int>        weak_guard = guard;
    TransferBatchAsyncContext context(0, guard);
    guard.reset();
    ASSERT_FALSE(weak_guard.expired());

    context.complete(ErrorInfo::OkStatus());

    EXPECT_TRUE(weak_guard.expired());
}

TEST(TransferBatchAsyncContextTest, CallbackRegisteredBeforeCompletionRunsOnce) {
    TransferBatchAsyncContext context;
    std::atomic<size_t> callback_count{0};
    ErrorCode           callback_code = ErrorCode::INVALID_PARAMS;

    context.onDone([&](TransferResult error) {
        callback_code = error.status.code();
        ++callback_count;
    });
    context.complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "first completion"));
    context.complete(ErrorInfo::OkStatus());

    EXPECT_EQ(callback_count.load(), 1u);
    EXPECT_EQ(callback_code, ErrorCode::EXECUTION_EXCEPTION);
}

TEST(TransferBatchAsyncContextTest, CallbackRegisteredAfterCompletionRunsImmediately) {
    TransferBatchAsyncContext context;
    context.complete(ErrorInfo::OkStatus());

    size_t callback_count = 0;
    context.onDone([&](TransferResult error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    EXPECT_EQ(callback_count, 1u);
}

TEST(TransferBatchAsyncContextTest, CallbackCanReadContextWithoutDeadlock) {
    TransferBatchAsyncContext context;
    context.onDone([&](TransferResult error) {
        EXPECT_TRUE(error.ok());
        EXPECT_TRUE(context.done());
        EXPECT_TRUE(context.success());
    });

    context.complete(ErrorInfo::OkStatus());
}

TEST(TransferBatchAsyncContextTest, MultipleCallbacksAndWaiterObserveSameTerminalResult) {
    auto context = std::make_shared<TransferBatchAsyncContext>();
    std::atomic<size_t> callback_count{0};

    context->onDone([&](TransferResult error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });
    context->onDone([&](TransferResult error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    std::thread waiter([&] { context->waitDone(); });
    context->complete(ErrorInfo::OkStatus());
    context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "duplicate completion"));
    waiter.join();

    EXPECT_EQ(callback_count.load(), 2u);
    EXPECT_TRUE(context->success());
}

TEST(TransferBatchAsyncContextTest, ClassifiedResultReachesCallbacksBeforeAndAfterCompletion) {
    TransferBatchAsyncContext context;
    size_t                    calls    = 0;
    const auto                callback = [&](TransferResult result) {
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.copy_error, CacheCopyError::IO_FAILED);
        EXPECT_EQ(result.status.ToString(), "disk read failed");
        ++calls;
    };
    context.onDone(callback);
    context.complete({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk read failed"), CacheCopyError::IO_FAILED});
    context.onDone(callback);
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(context.result().copy_error, CacheCopyError::IO_FAILED);
}

}  // namespace
}  // namespace rtp_llm
