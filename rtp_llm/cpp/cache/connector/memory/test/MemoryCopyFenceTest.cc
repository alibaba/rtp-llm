#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyFence.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

TEST(MemoryCopyFenceTest, QuiesceWaitsUntilCopyHandlerExits) {
    MemoryCopyFence fence;
    auto            begin = fence.begin("copy-a", 1s);
    ASSERT_TRUE(begin) << begin.error;

    auto quiesce = std::async(std::launch::async, [&]() { return fence.sealAndWait("copy-a", 1s, 1s); });
    EXPECT_EQ(quiesce.wait_for(20ms), std::future_status::timeout);

    begin.operation.reset();
    EXPECT_TRUE(quiesce.get());
}

TEST(MemoryCopyFenceTest, QuiesceBeforeCopyRejectsLateHandler) {
    MemoryCopyFence fence;
    ASSERT_TRUE(fence.sealAndWait("copy-a", 100ms, 1s));

    const auto begin = fence.begin("copy-a", 1s);
    EXPECT_FALSE(begin);
    EXPECT_EQ(begin.error, "memory copy operation is sealed");
}

TEST(MemoryCopyFenceTest, TimedOutQuiesceStaysSealedForRetry) {
    MemoryCopyFence fence;
    auto            begin = fence.begin("copy-a", 1s);
    ASSERT_TRUE(begin) << begin.error;

    EXPECT_FALSE(fence.sealAndWait("copy-a", 1ms, 1s));
    EXPECT_FALSE(fence.begin("copy-a", 1s));

    begin.operation.reset();
    EXPECT_TRUE(fence.sealAndWait("copy-a", 100ms, 1s));
}

TEST(MemoryCopyFenceTest, StopRejectsNewCopiesAndWaitsForEveryActiveHandler) {
    MemoryCopyFence fence;
    auto            first  = fence.begin("copy-a", 1s);
    auto            second = fence.begin("copy-b", 1s);
    ASSERT_TRUE(first) << first.error;
    ASSERT_TRUE(second) << second.error;

    auto stop = std::async(std::launch::async, [&]() { return fence.stopAndWait(1s); });
    EXPECT_EQ(stop.wait_for(20ms), std::future_status::timeout);

    MemoryCopyFence::BeginResult after_stop;
    for (int attempt = 0; attempt < 100; ++attempt) {
        after_stop = fence.begin("copy-after-stop-" + std::to_string(attempt), 1s);
        if (!after_stop) {
            break;
        }
        after_stop.operation.reset();
    }
    ASSERT_FALSE(after_stop);
    EXPECT_EQ(after_stop.error, "memory copy fence is stopped");

    first.operation.reset();
    EXPECT_EQ(stop.wait_for(20ms), std::future_status::timeout);
    second.operation.reset();
    EXPECT_TRUE(stop.get());
}

TEST(MemoryCopyFenceTest, CopyGuardCanOutliveFenceObject) {
    MemoryCopyFence::Operation operation;
    {
        auto fence = std::make_unique<MemoryCopyFence>();
        auto begin = fence->begin("copy-a", 1s);
        ASSERT_TRUE(begin) << begin.error;
        operation = std::move(begin.operation);
    }
    EXPECT_NO_THROW(operation.reset());
}

TEST(MemoryCopyFenceTest, DuplicateCopyIdIsRejected) {
    MemoryCopyFence fence;
    auto            first = fence.begin("copy-a", 1s);
    ASSERT_TRUE(first) << first.error;

    const auto duplicate = fence.begin("copy-a", 1s);
    EXPECT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error, "memory copy operation has already begun");
}

TEST(MemoryCopyFenceTest, NewAdmissionDoesNotScanEveryUnexpiredEntry) {
    MemoryCopyFence fence;
    for (int i = 0; i < 10000; ++i) {
        auto begin = fence.begin("bulk-" + std::to_string(i), 1h);
        ASSERT_TRUE(begin) << begin.error;
        begin.operation.reset();
    }
    ASSERT_EQ(fence.entryCountForTest(), 10000u);

    const size_t checks_before = fence.pruneCandidateChecksForTest();
    auto         next          = fence.begin("next", 1h);
    ASSERT_TRUE(next) << next.error;
    const size_t checks_after = fence.pruneCandidateChecksForTest();

    EXPECT_LE(checks_after - checks_before, 1u);
}

TEST(MemoryCopyFenceTest, StaleHeapExpiryCannotDeleteExtendedTombstone) {
    MemoryCopyFence fence;
    const auto      before_begin = std::chrono::steady_clock::now();
    auto            begin        = fence.begin("copy-a", 1ms);
    ASSERT_TRUE(begin) << begin.error;
    begin.operation.reset();

    ASSERT_TRUE(fence.sealAndWait("copy-a", 100ms, 1h));
    fence.pruneExpiredAtForTest(before_begin + 1s);

    EXPECT_EQ(fence.entryCountForTest(), 1u);
    const auto late = fence.begin("copy-a", 1h);
    EXPECT_FALSE(late);
    EXPECT_EQ(late.error, "memory copy operation is sealed");
}

TEST(MemoryCopyFenceTest, ExpiredDeadlineCannotReopenPrunedTombstone) {
    MemoryCopyFence fence;
    ASSERT_TRUE(fence.sealAndWait("copy-a", 100ms, 1ms));
    fence.pruneExpiredAtForTest(std::chrono::steady_clock::now() + 1h);
    ASSERT_EQ(fence.entryCountForTest(), 0u);

    const auto expired_deadline = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count()
                                  - 1;
    const auto late = fence.beginBeforeDeadline(
        "copy-a", 1s, expired_deadline);
    EXPECT_FALSE(late);
    EXPECT_EQ(late.error, "memory copy operation deadline has expired");
    EXPECT_EQ(fence.entryCountForTest(), 0u);
}

TEST(MemoryCopyFenceTest, SealRetentionStartsAfterStateLockIsAcquired) {
    MemoryCopyFence        fence;
    std::mutex             mutex;
    std::condition_variable changed;
    bool                   lock_held{false};
    bool                   release_lock{false};

    auto lock_holder = std::async(std::launch::async, [&]() {
        fence.withStateLockForTest([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            lock_held = true;
            changed.notify_all();
            changed.wait(lock, [&]() { return release_lock; });
        });
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(changed.wait_for(lock, 1s, [&]() { return lock_held; }));
    }

    auto quiesce = std::async(std::launch::async, [&]() { return fence.sealAndWait("copy-a", 1s, 10ms); });
    std::this_thread::sleep_for(30ms);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_lock = true;
    }
    changed.notify_all();
    lock_holder.get();
    ASSERT_TRUE(quiesce.get());

    const auto late = fence.begin("copy-a", 1s);
    EXPECT_FALSE(late);
    EXPECT_EQ(late.error, "memory copy operation is sealed");
}

TEST(MemoryCopyFenceTest, ExistingFenceStateWinsOverExpiredDeadline) {
    MemoryCopyFence fence;
    const auto      expired_deadline = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count()
                                  - 1;

    ASSERT_TRUE(fence.sealAndWait("sealed", 100ms, 1s));
    const auto sealed = fence.beginBeforeDeadline("sealed", 1s, expired_deadline);
    EXPECT_FALSE(sealed);
    EXPECT_EQ(sealed.error, "memory copy operation is sealed");

    auto active = fence.begin("active", 1s);
    ASSERT_TRUE(active) << active.error;
    const auto duplicate = fence.beginBeforeDeadline("active", 1s, expired_deadline);
    EXPECT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error, "memory copy operation has already begun");
    active.operation.reset();

    ASSERT_TRUE(fence.stopAndWait(100ms));
    const auto stopped = fence.beginBeforeDeadline("new", 1s, expired_deadline);
    EXPECT_FALSE(stopped);
    EXPECT_EQ(stopped.error, "memory copy fence is stopped");
}

TEST(MemoryCopyFenceTest, ActiveOperationCrossingExpiryIsRemovedWhenGuardExits) {
    MemoryCopyFence fence;
    const auto      before_begin = std::chrono::steady_clock::now();
    auto            begin        = fence.begin("copy-a", 1ms);
    ASSERT_TRUE(begin) << begin.error;

    fence.pruneExpiredAtForTest(before_begin + 1s);
    ASSERT_EQ(fence.entryCountForTest(), 1u);

    begin.operation.reset();
    EXPECT_EQ(fence.entryCountForTest(), 0u);
}

}  // namespace
}  // namespace rtp_llm
