#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

TEST(MemoryAsyncContextTest, WaitBlocksUntilComplete) {
    MemoryAsyncContext context(nullptr);
    std::promise<void> waiter_started;
    auto               waiter = std::async(std::launch::async, [&]() {
        waiter_started.set_value();
        context.waitDone();
    });

    waiter_started.get_future().wait();
    EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
    EXPECT_FALSE(context.done());

    context.complete(true);
    EXPECT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
}

TEST(MemoryAsyncContextTest, CompletePublishesSuccessAndFailure) {
    MemoryAsyncContext succeeded(nullptr);
    succeeded.complete(true);
    EXPECT_TRUE(succeeded.done());
    EXPECT_TRUE(succeeded.success());

    MemoryAsyncContext failed(nullptr);
    failed.complete(false);
    EXPECT_TRUE(failed.done());
    EXPECT_FALSE(failed.success());
}

TEST(MemoryAsyncContextTest, CallbackRunsExactlyOnce) {
    std::atomic<int> callback_count{0};
    MemoryAsyncContext context([&](bool success) {
        EXPECT_TRUE(success);
        callback_count.fetch_add(1);
    });

    std::vector<std::future<void>> completions;
    for (int i = 0; i < 8; ++i) {
        completions.emplace_back(std::async(std::launch::async, [&]() { context.complete(true); }));
    }
    for (auto& completion : completions) {
        completion.get();
    }

    EXPECT_EQ(callback_count.load(), 1);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
}

TEST(MemoryAsyncContextTest, CallbackExceptionMakesCompletionFail) {
    MemoryAsyncContext context([](bool) { throw std::runtime_error("callback failed"); });

    EXPECT_NO_THROW(context.complete(true));
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
}

TEST(MemoryAsyncContextTest, ConcurrentWaitersObserveOneCompletion) {
    MemoryAsyncContext context(nullptr);
    std::atomic<int>    waiter_count{0};
    std::vector<std::future<void>> waiters;
    for (int i = 0; i < 8; ++i) {
        waiters.emplace_back(std::async(std::launch::async, [&]() {
            context.waitDone();
            waiter_count.fetch_add(1);
        }));
    }

    context.complete(true);
    for (auto& waiter : waiters) {
        EXPECT_EQ(waiter.wait_for(1s), std::future_status::ready);
    }
    EXPECT_EQ(waiter_count.load(), 8);
    EXPECT_TRUE(context.success());
}

TEST(MemoryAsyncContextTest, TaskExceptionCanPublishTerminalFailure) {
    MemoryAsyncContext context(nullptr);
    auto worker = std::async(std::launch::async, [&]() {
        try {
            throw std::runtime_error("task setup failed");
        } catch (...) {
            context.complete(false);
        }
    });

    context.waitDone();
    worker.get();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
}

}  // namespace
}  // namespace rtp_llm
