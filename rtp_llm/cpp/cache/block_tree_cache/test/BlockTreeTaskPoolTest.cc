#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
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
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&completed] { completed.fetch_add(1); }));
    ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1); }));
    pool.waitForIdle();

    EXPECT_EQ(completed.load(), 2);
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, WaitForIdleOnlyTracksTaskBodyNotExternalAsyncCompletion) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::atomic<bool>     transfer_done{false};
    std::function<void()> finish_transfer;
    ASSERT_TRUE(pool.submit([&] {
        // Model a task that starts an external asynchronous transfer and
        // returns after registering its eventual completion callback.
        finish_transfer = [&transfer_done] { transfer_done.store(true); };
    }));

    pool.waitForIdle();
    ASSERT_TRUE(finish_transfer);
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
    EXPECT_FALSE(transfer_done.load());

    finish_transfer();
    EXPECT_TRUE(transfer_done.load());
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

TEST(BlockTreeTaskPoolTest, ShutdownClearsPopulatedQueuesAndReclaimsPending) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    std::atomic<bool>  worker_released{false};
    auto               release = [&] {
        if (!worker_released.exchange(true)) {
            release_worker.set_value();
        }
    };
    [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [&](void*) { release(); });
    ASSERT_TRUE(pool.submit([&] {
        worker_ready.set_value();
        release_future.wait();
    }));
    if (ready_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release();
        pool.shutdown();
        FAIL() << "blocking task did not occupy the worker";
    }

    // Populate all three queues while the only worker stays busy.
    std::atomic<bool> load_ran{false};
    std::atomic<bool> background_ran{false};
    std::atomic<bool> completion_ran{false};
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&load_ran] { load_ran.store(true); }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::BACKGROUND, [&background_ran] { background_ran.store(true); }));
    ASSERT_TRUE(pool.submitCompletion([&completion_ran] { completion_ran.store(true); }));
    EXPECT_EQ(pool.pending_tasks_.load(), 4);

    // shutdown() clears the queues under the lock, then joins the worker, so run
    // it on another thread and release the worker only after the clear is
    // observable via shutdown_ (set inside the same critical section).
    std::thread shutdown_thread([&pool] { pool.shutdown(); });
    auto        shutdown_started = [&pool] {
        std::lock_guard<std::mutex> lock(pool.lifecycle_mutex_);
        return pool.shutdown_;
    };
    while (!shutdown_started()) {
        std::this_thread::yield();
    }
    release();
    shutdown_thread.join();

    EXPECT_FALSE(load_ran.load());
    EXPECT_FALSE(background_ran.load());
    EXPECT_FALSE(completion_ran.load());
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, CompletionTasksPreemptQueuedNormalTasksAndRemainFifo) {
    BlockTreeTaskPool pool(1, 8, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    std::mutex         events_mutex;
    std::vector<int>   events;
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

TEST(BlockTreeTaskPoolTest, LoadPriorityIsBoundedAndPreservesFifoAcrossClasses) {
    BlockTreeTaskPool pool(1, 16, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    std::atomic<bool>  worker_released{false};
    auto               release = [&] {
        if (!worker_released.exchange(true)) {
            release_worker.set_value();
        }
    };
    [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [&](void*) { release(); });
    ASSERT_TRUE(pool.submit([&] {
        worker_ready.set_value();
        release_future.wait();
    }));
    if (ready_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release();
        pool.shutdown();
        FAIL() << "blocking task did not occupy the worker";
    }

    std::vector<int>  events;
    std::atomic<bool> completion_accepted{false};
    ASSERT_TRUE(pool.submit([&events] { events.push_back(1); }));
    ASSERT_TRUE(pool.submit([&events] { events.push_back(2); }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&events] { events.push_back(10); }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&] {
        events.push_back(11);
        completion_accepted.store(pool.submitCompletion([&events] { events.push_back(100); }));
    }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&events] { events.push_back(12); }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&events] { events.push_back(13); }));
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [&events] { events.push_back(14); }));

    release();
    pool.waitForIdle();

    EXPECT_TRUE(completion_accepted.load());
    EXPECT_EQ(events, (std::vector<int>{10, 11, 100, 12, 13, 1, 14, 2}));
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

TEST(BlockTreeTaskPoolTest, ReservedSlotsRejectBackgroundButRemainAvailableToLoad) {
    const size_t      queue_size       = BlockTreeTaskPool::kLoadReservedSlots + 2;
    const size_t      background_limit = queue_size - BlockTreeTaskPool::kLoadReservedSlots;
    BlockTreeTaskPool pool(1, queue_size, "BlockTreeTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    std::atomic<bool>  worker_released{false};
    auto               release = [&] {
        if (!worker_released.exchange(true)) {
            release_worker.set_value();
        }
    };
    [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [&](void*) { release(); });
    ASSERT_TRUE(pool.submit([&] {
        worker_ready.set_value();
        release_future.wait();
    }));
    if (ready_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release();
        pool.shutdown();
        FAIL() << "blocking task did not occupy the worker";
    }

    std::atomic<bool> rejected_task_ran{false};
    // Background may fill only the non-reserved slots.
    for (size_t index = 0; index < background_limit; ++index) {
        ASSERT_TRUE(pool.submit([] {}));
    }
    EXPECT_FALSE(pool.submit([&rejected_task_ran] { rejected_task_ran.store(true); }));

    // Load may also use the reserved slots, up to the shared total capacity.
    for (size_t index = 0; index < BlockTreeTaskPool::kLoadReservedSlots; ++index) {
        ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [] {}));
    }
    EXPECT_FALSE(pool.submit(BlockTreeTaskClass::LOAD, [&rejected_task_ran] { rejected_task_ran.store(true); }));
    EXPECT_EQ(pool.pending_tasks_.load(), static_cast<int>(1 + queue_size));

    release();
    pool.waitForIdle();

    EXPECT_FALSE(rejected_task_ran.load());
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockTreeTaskPoolTest, SmallPoolsSkipReserveAndUnboundedQueuesStayUnlimited) {
    // A pool no larger than the reserve must not starve Background.
    {
        BlockTreeTaskPool pool(1, 1, "BlockTreeTaskPoolTest");
        ASSERT_TRUE(pool.start());

        std::promise<void> worker_ready;
        std::promise<void> release_worker;
        auto               ready_future   = worker_ready.get_future();
        auto               release_future = release_worker.get_future();
        std::atomic<bool>  worker_released{false};
        auto               release = [&] {
            if (!worker_released.exchange(true)) {
                release_worker.set_value();
            }
        };
        [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [&](void*) { release(); });
        ASSERT_TRUE(pool.submit([&] {
            worker_ready.set_value();
            release_future.wait();
        }));
        ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

        // The single slot is available to Background despite the reserve.
        EXPECT_TRUE(pool.submit([] {}));
        EXPECT_FALSE(pool.submit([] {}));
        EXPECT_FALSE(pool.submit(BlockTreeTaskClass::LOAD, [] {}));

        release();
        pool.waitForIdle();
    }

    // An unbounded pool applies no capacity limit to either class.
    {
        BlockTreeTaskPool pool(1, 0, "BlockTreeTaskPoolTest");
        ASSERT_TRUE(pool.start());

        std::promise<void> worker_ready;
        std::promise<void> release_worker;
        auto               ready_future   = worker_ready.get_future();
        auto               release_future = release_worker.get_future();
        std::atomic<bool>  worker_released{false};
        auto               release = [&] {
            if (!worker_released.exchange(true)) {
                release_worker.set_value();
            }
        };
        [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [&](void*) { release(); });
        ASSERT_TRUE(pool.submit([&] {
            worker_ready.set_value();
            release_future.wait();
        }));
        ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

        EXPECT_TRUE(pool.submit([] {}));
        EXPECT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [] {}));

        release();
        pool.waitForIdle();
    }
}

}  // namespace
}  // namespace rtp_llm
