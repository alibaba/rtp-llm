#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

namespace rtp_llm {
namespace {

TEST(BlockTreeTaskPoolTest, StartOnlySucceedsOnce) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    EXPECT_TRUE(pool.start());
    EXPECT_FALSE(pool.start());
}

TEST(BlockTreeTaskPoolTest, SubmitAndWaitForIdleTrackAcceptedTasks) {
    BlockTreeTaskPool pool(2, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::atomic<int> completed{0};
    ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1); }));
    ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1); }));
    pool.waitForIdle();

    EXPECT_EQ(completed.load(), 2);
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, ThrowingTaskStillSettlesPendingCount) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());
    ASSERT_TRUE(pool.submit([] { throw std::runtime_error("expected"); }));

    pool.waitForIdle();
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, ShutdownRejectsNewTasksAndIsIdempotent) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());
    pool.shutdown();
    pool.shutdown();

    EXPECT_FALSE(pool.submit([] {}));
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, CompletionTasksPreemptQueuedNormalTasksAndRemainFifo) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    std::mutex          events_mutex;
    std::vector<int>    events;
    ASSERT_TRUE(pool.submit([&] {
        worker_ready.set_value();
        release_future.wait();
    }));
    ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    ASSERT_TRUE(pool.submit([&] {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(3);
    }));
    ASSERT_TRUE(pool.submit([&] {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(4);
    }));
    ASSERT_TRUE(pool.submitCompletion([&] {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(1);
    }));
    ASSERT_TRUE(pool.submitCompletion([&] {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(2);
    }));

    release_worker.set_value();
    pool.waitForIdle();
    EXPECT_EQ(events, (std::vector<int>{1, 2, 3, 4}));
}

TEST(BlockTreeTaskPoolTest, StopAdmissionKeepsUnboundedCompletionQueueOpen) {
    BlockTreeTaskPool pool(1, 1, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());
    pool.stopAdmission();

    EXPECT_FALSE(pool.submit([] {}));
    std::atomic<int> completions{0};
    ASSERT_TRUE(pool.submitCompletion([&] { ++completions; }));
    ASSERT_TRUE(pool.submitCompletion([&] { ++completions; }));
    pool.waitForIdle();
    EXPECT_EQ(completions.load(), 2);
}

TEST(BlockTreeTaskPoolTest, FullQueueRejectsSubmissionWithoutBlockingAndRestoresPendingCount) {
    // The public queue bound applies only to normal tasks waiting in the local
    // FIFO. The task already running on the worker does not consume a queue slot.
    BlockTreeTaskPool pool(1, 1, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();

    std::atomic<int>  executed{0};
    std::atomic<bool> rejected_task_ran{false};
    const auto        worker_task = [&] {
        worker_ready.set_value();
        release_future.wait();
        executed.fetch_add(1);
    };

    // Occupy the only worker.
    ASSERT_TRUE(pool.submit(worker_task));
    ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Fill the only normal queue slot; the busy worker cannot consume it yet.
    ASSERT_TRUE(pool.submit([&executed] { executed.fetch_add(1); }));

    // A further submit must fail fast instead of blocking forever.
    auto rejected = std::async(std::launch::async, [&pool, &rejected_task_ran] {
        return pool.submit([&rejected_task_ran] { rejected_task_ran.store(true); });
    });
    ASSERT_EQ(rejected.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(rejected.get());
    EXPECT_FALSE(rejected_task_ran.load());

    // Unblock the worker; all accepted tasks must still run.
    release_worker.set_value();
    pool.waitForIdle();

    EXPECT_EQ(executed.load(), 2);
    EXPECT_FALSE(rejected_task_ran.load());
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, BusinessCreditsBoundInFlightWorkAndKeepWaitForIdleBlocked) {
    BlockTreeTaskPool pool(1, 2, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    ASSERT_TRUE(pool.acquireBusinessCredit());
    ASSERT_TRUE(pool.acquireBusinessCredit());
    EXPECT_FALSE(pool.acquireBusinessCredit());

    auto idle = std::async(std::launch::async, [&pool] { pool.waitForIdle(); });
    EXPECT_EQ(idle.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

    pool.releaseBusinessCredit();
    EXPECT_EQ(idle.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    pool.releaseBusinessCredit();
    EXPECT_EQ(idle.wait_for(std::chrono::seconds(5)), std::future_status::ready);
}

TEST(BlockTreeTaskPoolTest, StopAdmissionRejectsNewBusinessCreditsButAllowsRelease) {
    BlockTreeTaskPool pool(1, 1, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());
    ASSERT_TRUE(pool.acquireBusinessCredit());
    pool.stopAdmission();

    EXPECT_FALSE(pool.acquireBusinessCredit());
    pool.releaseBusinessCredit();
    pool.waitForIdle();
}

}  // namespace
}  // namespace rtp_llm
