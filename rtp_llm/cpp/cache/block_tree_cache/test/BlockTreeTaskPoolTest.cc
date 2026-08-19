#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>

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

TEST(BlockTreeTaskPoolTest, FullQueueRejectsSubmissionWithoutBlockingAndRestoresPendingCount) {
    // LockFreeThreadPool's in-flight counter includes the task currently being
    // executed, so with thread_count=1/queue_size=1 one task must block the
    // worker and two more must fill the queue before the next submit is
    // rejected (> queue_size).
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

    // Fill both queue slots; the busy worker cannot consume them yet.
    ASSERT_TRUE(pool.submit([&executed] { executed.fetch_add(1); }));
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

    EXPECT_EQ(executed.load(), 3);
    EXPECT_FALSE(rejected_task_ran.load());
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

}  // namespace
}  // namespace rtp_llm
