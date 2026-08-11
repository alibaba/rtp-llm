#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {

TEST(TransferBatchAsyncContextTest, FreshContextRemainsPendingUntilProducerCompletes) {
    TransferBatchAsyncContext context;

    EXPECT_FALSE(context.done());
    EXPECT_FALSE(context.success());
}

TEST(TransferBatchAsyncContextTest, WaitDoneBlocksUntilProducerCompletes) {
    TransferBatchAsyncContext context;
    std::atomic<bool>         wait_returned{false};
    std::promise<void>        wait_started_promise;
    std::future<void>         wait_started_future = wait_started_promise.get_future();
    std::thread waiter([&] {
        wait_started_promise.set_value();
        context.waitDone();
        wait_returned.store(true, std::memory_order_release);
    });

    const bool waiter_started = wait_started_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (waiter_started) {
        EXPECT_FALSE(wait_returned.load(std::memory_order_acquire));
    }

    context.complete(ErrorInfo::OkStatus());
    waiter.join();

    ASSERT_TRUE(waiter_started) << "waitDone() did not enter its coordinated wait before completion";
    EXPECT_TRUE(wait_returned.load(std::memory_order_acquire));
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
}

TEST(TransferBatchAsyncContextTest, FirstCompletionWinsAndPreservesFailure) {
    TransferBatchAsyncContext context;
    const ErrorInfo           failure(ErrorCode::EXECUTION_EXCEPTION, "batch transfer failed");

    context.complete(failure);
    context.complete(ErrorInfo::OkStatus());

    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_EQ(context.errorInfo().ToString(), "batch transfer failed");
}

TEST(TransferBatchAsyncContextTest, ReleasesCompletionGuardBeforeDoneBecomesVisible) {
    struct DestructionObserver {
        explicit DestructionObserver(std::atomic<bool>* destroyed): destroyed(destroyed) {}
        ~DestructionObserver() {
            destroyed->store(true, std::memory_order_release);
        }
        std::atomic<bool>* destroyed;
    };

    std::atomic<bool>         destroyed{false};
    auto                      guard = std::make_shared<DestructionObserver>(&destroyed);
    TransferBatchAsyncContext context(guard);
    guard.reset();

    context.complete(ErrorInfo::OkStatus());

    EXPECT_TRUE(context.done());
    EXPECT_TRUE(destroyed.load(std::memory_order_acquire));
}

TEST(TransferBatchAsyncContextTest, FirstCompletionWinsAndPreservesSuccess) {
    TransferBatchAsyncContext context;

    context.complete(ErrorInfo::OkStatus());
    context.complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "later batch transfer failure"));

    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(TransferBatchAsyncContextTest, RepeatedObservationsAndWaitsAreSafeAfterSuccess) {
    TransferBatchAsyncContext context;
    context.complete(ErrorInfo::OkStatus());

    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.done());
    context.waitDone();
    context.waitDone();
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
}

}  // namespace
}  // namespace rtp_llm
