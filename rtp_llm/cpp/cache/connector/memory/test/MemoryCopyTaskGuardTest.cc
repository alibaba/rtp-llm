#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyTaskGuard.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

RemoteLoadLeaseRetainer::Config testRetainerConfig() {
    return RemoteLoadLeaseRetainer::Config{
        /*max_jobs=*/4,
        /*initial_backoff=*/1ms,
        /*max_backoff=*/5ms,
        /*stop_grace=*/1s,
        /*worker_count=*/1,
    };
}

bool waitUntil(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = 1s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}

TEST(MemoryCopyTaskGuardTest, DroppedQueuedTaskFailsContextAndReleasesUnstartedLease) {
    RemoteLoadLeaseRetainer retainer(testRetainerConfig());
    std::atomic<int>        callback_count{0};
    std::atomic<int>        quiesce_count{0};
    auto context = std::make_shared<MemoryAsyncContext>([&](bool success) {
        EXPECT_FALSE(success);
        ++callback_count;
    });
    auto lease      = std::make_shared<int>(7);
    auto weak_lease = std::weak_ptr<int>(lease);
    auto ticket_status = retainer.reserve("queued", lease, [&]() {
        ++quiesce_count;
        return true;
    });
    ASSERT_TRUE(ticket_status.ok()) << ticket_status.status();
    lease.reset();

    auto guard = std::make_shared<MemoryCopyTaskGuard>(context, std::move(*ticket_status));
    auto pool  = std::make_unique<autil::LockFreeThreadPool>(
        /*thread_num=*/1, /*queue_size=*/4, /*thread_init_func=*/nullptr, "MemoryCopyDropTest");
    ASSERT_TRUE(pool->start());

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    auto               release = release_blocker.get_future().share();
    ASSERT_EQ(pool->pushTask([&]() {
                  blocker_started.set_value();
                  release.wait();
              }),
              autil::ThreadPoolBase::ERROR_NONE);
    ASSERT_EQ(blocker_started.get_future().wait_for(1s), std::future_status::ready);

    std::atomic<int> dispatch_count{0};
    ASSERT_EQ(pool->pushTask([guard, &dispatch_count]() { ++dispatch_count; }),
              autil::ThreadPoolBase::ERROR_NONE);
    guard.reset();

    auto stop = std::async(std::launch::async, [&]() { pool->stop(); });
    ASSERT_TRUE(waitUntil([&]() { return !pool->IsRunning(); }));
    release_blocker.set_value();
    ASSERT_EQ(stop.wait_for(1s), std::future_status::ready);
    stop.get();
    EXPECT_EQ(dispatch_count.load(), 0);
    EXPECT_FALSE(context->done());

    pool.reset();
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(callback_count.load(), 1);
    EXPECT_EQ(quiesce_count.load(), 0);
    EXPECT_EQ(retainer.activeJobsForTest(), 0u);
    EXPECT_TRUE(weak_lease.expired());
}

TEST(MemoryCopyTaskGuardTest, StartedAbandonRetainsLeaseUntilBackgroundQuiesceSucceeds) {
    RemoteLoadLeaseRetainer retainer(testRetainerConfig());
    std::atomic<int>        callback_count{0};
    std::atomic<int>        quiesce_count{0};
    auto context = std::make_shared<MemoryAsyncContext>([&](bool success) {
        EXPECT_FALSE(success);
        ++callback_count;
    });
    auto lease      = std::make_shared<int>(9);
    auto weak_lease = std::weak_ptr<int>(lease);
    std::atomic<bool> allow_quiesce{false};
    auto ticket_status = retainer.reserve("started", lease, [&]() {
        ++quiesce_count;
        return allow_quiesce.load();
    });
    ASSERT_TRUE(ticket_status.ok()) << ticket_status.status();
    lease.reset();

    auto guard = std::make_shared<MemoryCopyTaskGuard>(context, std::move(*ticket_status));
    ASSERT_TRUE(guard->enterBeforeDeadline(/*operation_deadline_unix_ms=*/1100,
                                           /*retention_timeout_ms=*/100,
                                           /*safety_window_ms=*/100,
                                           /*now_unix_ms=*/1000));
    ASSERT_TRUE(guard->markStarted());
    guard->abandon();
    guard.reset();

    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(callback_count.load(), 1);
    ASSERT_TRUE(waitUntil([&]() { return quiesce_count.load() > 0; }));
    EXPECT_FALSE(weak_lease.expired());
    allow_quiesce.store(true);
    ASSERT_TRUE(waitUntil([&]() { return retainer.activeJobsForTest() == 0; }));
    EXPECT_GE(quiesce_count.load(), 1);
    EXPECT_TRUE(weak_lease.expired());
}

TEST(MemoryCopyTaskGuardTest, ExpiredAdmissionFailsExactlyOnceWithoutStartingOrQuiescing) {
    RemoteLoadLeaseRetainer retainer(testRetainerConfig());
    std::atomic<int>        callback_count{0};
    std::atomic<int>        quiesce_count{0};
    auto context = std::make_shared<MemoryAsyncContext>([&](bool success) {
        EXPECT_FALSE(success);
        ++callback_count;
    });
    auto ticket_status = retainer.reserve("expired", std::make_shared<int>(11), [&]() {
        ++quiesce_count;
        return true;
    });
    ASSERT_TRUE(ticket_status.ok()) << ticket_status.status();

    {
        auto guard = std::make_shared<MemoryCopyTaskGuard>(context, std::move(*ticket_status));
        EXPECT_FALSE(guard->enterBeforeDeadline(/*operation_deadline_unix_ms=*/1000,
                                                /*retention_timeout_ms=*/100,
                                                /*safety_window_ms=*/100,
                                                /*now_unix_ms=*/1000));
        guard->cancelBeforeDispatch();
    }

    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(callback_count.load(), 1);
    EXPECT_EQ(quiesce_count.load(), 0);
    EXPECT_EQ(retainer.activeJobsForTest(), 0u);
}

}  // namespace
}  // namespace rtp_llm
