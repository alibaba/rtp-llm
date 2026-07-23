#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockCacheTaskPool.h"

namespace rtp_llm {
namespace {

TEST(BlockCacheTaskPoolTest, StartOnlySucceedsOnce) {
    BlockCacheTaskPool pool(1, 8, "BlockCacheTaskPoolTest");
    EXPECT_TRUE(pool.start());
    EXPECT_FALSE(pool.start());
}

TEST(BlockCacheTaskPoolTest, SubmitAndWaitForIdleTrackAcceptedTasks) {
    BlockCacheTaskPool pool(2, 8, "BlockCacheTaskPoolTest");
    ASSERT_TRUE(pool.start());

    std::atomic<int> completed{0};
    ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1); }));
    ASSERT_TRUE(pool.submit([&completed] { completed.fetch_add(1); }));
    pool.waitForIdle();

    EXPECT_EQ(completed.load(), 2);
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockCacheTaskPoolTest, ThrowingTaskStillSettlesPendingCount) {
    BlockCacheTaskPool pool(1, 8, "BlockCacheTaskPoolTest");
    ASSERT_TRUE(pool.start());
    ASSERT_TRUE(pool.submit([] { throw std::runtime_error("expected"); }));

    pool.waitForIdle();
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

TEST(BlockCacheTaskPoolTest, ShutdownRejectsNewTasksAndIsIdempotent) {
    BlockCacheTaskPool pool(1, 8, "BlockCacheTaskPoolTest");
    ASSERT_TRUE(pool.start());
    pool.shutdown();
    pool.shutdown();

    EXPECT_FALSE(pool.submit([] {}));
    EXPECT_EQ(pool.pending_tasks_.load(), 0);
}

}  // namespace
}  // namespace rtp_llm
