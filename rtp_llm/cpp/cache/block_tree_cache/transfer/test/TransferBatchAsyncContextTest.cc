#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <limits>
#include <thread>
#include <stdexcept>

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

TEST(TransferBatchAsyncContextTest, BufferSafeFailureReachesCallbacksBeforeAndAfterCompletion) {
    TransferBatchAsyncContext context(2);
    EXPECT_FALSE(context.done());
    EXPECT_FALSE(context.result().buffer_safe);

    size_t     calls    = 0;
    const auto callback = [&](TransferResult result) {
        EXPECT_TRUE(result.buffer_safe);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status.code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(result.status.ToString(), "completed disk read failed");
        EXPECT_EQ(result.copy_error, CacheCopyError::IO_FAILED);
        EXPECT_EQ(result.block_errors, std::vector<CacheCopyError>(2, CacheCopyError::IO_FAILED));
        ++calls;
    };
    context.onDone(callback);
    EXPECT_EQ(calls, 0u);
    TransferResult failure(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "completed disk read failed"),
                           CacheCopyError::IO_FAILED);
    failure.buffer_safe = true;
    context.complete(std::move(failure));
    context.waitDone();
    EXPECT_EQ(calls, 1u);
    context.onDone(callback);
    context.complete(TransferStatus::OK);

    EXPECT_EQ(calls, 2u);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.result().buffer_safe);
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_EQ(context.errorInfo().ToString(), "completed disk read failed");
}

TEST(TransferBatchAsyncContextTest, ThrowingConsumersDoNotInterruptOtherRegisteredCallbacks) {
    TransferBatchAsyncContext context(2);
    size_t                    calls = 0;
    context.onDone([&](TransferResult) {
        ++calls;
        throw std::runtime_error("consumer");
    });
    context.onDone([&](TransferResult result) {
        ++calls;
        EXPECT_EQ(result.block_errors.size(), 2u);
    });
    context.onDone([&](TransferResult) {
        ++calls;
        throw 7;
    });
    context.onDone([&](TransferResult) { ++calls; });
    EXPECT_NO_THROW(context.complete(TransferStatus::OK));
    EXPECT_TRUE(context.done());
    EXPECT_EQ(calls, 4u);
    context.complete(TransferStatus::DEVICE_IO_ERROR);
    EXPECT_EQ(calls, 4u);
}

TEST(TransferBatchAsyncContextTest, LateCallbackKeepsRegistrationThreadAndExceptionSemantics) {
    TransferBatchAsyncContext context;
    context.complete(TransferStatus::OK);
    const auto registrar = std::this_thread::get_id();
    EXPECT_THROW(context.onDone([&](TransferResult) {
        EXPECT_EQ(std::this_thread::get_id(), registrar);
        throw std::runtime_error("late consumer");
    }),
                 std::runtime_error);
}

TEST(TransferBatchAsyncContextTest, LatePhysicalProofSurvivesEarlierBusinessCompletion) {
    size_t safe_calls = 0;
    auto proof = std::make_shared<TransferBufferSafety>([&] { ++safe_calls; });
    TransferBatchAsyncContext context(1, nullptr, proof);
    size_t business_calls = 0;
    context.onDone([&](TransferResult result) {
        ++business_calls;
        EXPECT_FALSE(result.buffer_safe);
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    });
    context.complete(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "observation failed"));
    EXPECT_EQ(safe_calls, 0u);

    TransferResult drained_failure(TransferStatus::DEVICE_IO_ERROR);
    drained_failure.buffer_safe = true;
    context.complete(drained_failure);
    context.complete(drained_failure);
    EXPECT_EQ(safe_calls, 1u);
    EXPECT_EQ(business_calls, 1u);
    EXPECT_FALSE(context.result().buffer_safe);
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
}

TEST(TransferBatchAsyncContextTest, ProofPrecedesBusinessCallbacksAndGuardRelease) {
    auto guard = std::make_shared<int>(1);
    std::weak_ptr<int> weak_guard = guard;
    size_t safe_calls = 0;
    TransferBatchAsyncContext* active = nullptr;
    auto proof = std::make_shared<TransferBufferSafety>([&] {
        ++safe_calls;
        EXPECT_FALSE(active->done());
        EXPECT_FALSE(weak_guard.expired());
    });
    TransferBatchAsyncContext context(1, guard, proof);
    active = &context;
    guard.reset();
    context.onDone([&](TransferResult) {
        EXPECT_EQ(safe_calls, 1u);
        throw std::runtime_error("business consumer");
    });
    TransferResult safe(TransferStatus::OK);
    safe.buffer_safe = true;
    EXPECT_NO_THROW(context.complete(std::move(safe)));
    EXPECT_EQ(safe_calls, 1u);
    EXPECT_TRUE(weak_guard.expired());
    EXPECT_THROW(context.onDone([](TransferResult) { throw std::runtime_error("late business consumer"); }),
                 std::runtime_error);
    EXPECT_EQ(safe_calls, 1u);
}

TEST(TransferBatchAsyncContextTest, PhysicalProofSurvivesResultExpansionFailure) {
    size_t safe_calls = 0;
    auto proof = std::make_shared<TransferBufferSafety>([&] { ++safe_calls; });
    TransferBatchAsyncContext context(std::numeric_limits<size_t>::max(), nullptr, proof);
    TransferResult safe(TransferStatus::OK);
    safe.buffer_safe = true;
    EXPECT_THROW(context.complete(std::move(safe)), std::length_error);
    EXPECT_EQ(safe_calls, 1u);
    EXPECT_FALSE(context.done());
}

TEST(CompletedTransferAsyncContextTest, ProofPrecedesResultExpansionAndLateConsumer) {
    size_t safe_calls = 0;
    auto proof = std::make_shared<TransferBufferSafety>([&] { ++safe_calls; });
    TransferResult safe(TransferStatus::INVALID_ARGS);
    safe.buffer_safe = true;
    EXPECT_THROW((void)CompletedTransferAsyncContext(safe, std::numeric_limits<size_t>::max(), proof),
                 std::length_error);
    EXPECT_EQ(safe_calls, 1u);

    auto second_proof = std::make_shared<TransferBufferSafety>([&] { ++safe_calls; });
    CompletedTransferAsyncContext context(std::move(safe), 1, second_proof);
    EXPECT_EQ(safe_calls, 2u);
    EXPECT_THROW(context.onDone([&](TransferResult result) {
        EXPECT_TRUE(result.buffer_safe);
        EXPECT_EQ(safe_calls, 2u);
        throw std::runtime_error("completed consumer");
    }), std::runtime_error);
    EXPECT_EQ(safe_calls, 2u);
}

TEST(CompletedTransferAsyncContextTest, UnprovenResultDoesNotPublishSafety) {
    size_t safe_calls = 0;
    auto proof = std::make_shared<TransferBufferSafety>([&] { ++safe_calls; });
    CompletedTransferAsyncContext context(TransferStatus::OK, 1, proof);
    EXPECT_TRUE(context.success());
    EXPECT_EQ(safe_calls, 0u);
    proof->report(true);
    EXPECT_EQ(safe_calls, 1u);
}

}  // namespace
}  // namespace rtp_llm
