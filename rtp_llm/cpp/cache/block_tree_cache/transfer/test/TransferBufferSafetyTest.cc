#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBufferSafety.h"

namespace rtp_llm {
namespace {

TEST(TransferBufferSafetyTest, OnlyExplicitTrueReportsAndDestructionDoesNotReport) {
    size_t calls = 0;
    {
        TransferBufferSafety unproven([&] { ++calls; });
        unproven.report(false);
    }
    EXPECT_EQ(calls, 0u);

    TransferBufferSafety proof([&] { ++calls; });
    proof.report(false);
    EXPECT_EQ(calls, 0u);
    proof.report(true);
    proof.report(false);
    proof.report(true);
    EXPECT_EQ(calls, 1u);
}

TEST(TransferBufferSafetyTest, ConcurrentDuplicateProofsRunCallbackOnce) {
    std::atomic<size_t> calls{0};
    TransferBufferSafety proof([&] { ++calls; });
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            for (size_t j = 0; j < 64; ++j) {
                proof.report(false);
                proof.report(true);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(calls.load(), 1u);
}

TEST(TransferBufferSafetyTrackerTest, SynchronousProofsCannotFinishOpenSubmission) {
    size_t calls = 0;
    auto tracker = std::make_shared<TransferBufferSafetyTracker>([&] { ++calls; });
    auto first = tracker->addBatch();
    first->report(true);
    EXPECT_EQ(calls, 0u);
    auto second = tracker->addBatch();
    tracker->finishSubmitting();
    tracker->finishSubmitting();
    EXPECT_EQ(calls, 0u);
    first->report(true);
    second->report(false);
    EXPECT_EQ(calls, 0u);
    second->report(true);
    tracker->finishSubmitting();
    EXPECT_EQ(calls, 1u);
    EXPECT_THROW(tracker->addBatch(), std::logic_error);
}

TEST(TransferBufferSafetyTrackerTest, LostOrUnprovenBatchNeverCountsAsSafe) {
    size_t calls = 0;
    auto tracker = std::make_shared<TransferBufferSafetyTracker>([&] { ++calls; });
    auto first = tracker->addBatch();
    {
        auto abandoned = tracker->addBatch();
        abandoned->report(false);
    }
    tracker->finishSubmitting();
    first->report(true);
    tracker.reset();
    EXPECT_EQ(calls, 0u);
}

TEST(TransferBufferSafetyTrackerTest, EmptyClosedExecutionReportsOnceOutsideTrackerLock) {
    size_t calls = 0;
    std::shared_ptr<TransferBufferSafetyTracker> tracker;
    tracker = std::make_shared<TransferBufferSafetyTracker>([&] {
        ++calls;
        tracker->finishSubmitting();
        EXPECT_THROW(tracker->addBatch(), std::logic_error);
    });
    tracker->finishSubmitting();
    tracker->finishSubmitting();
    EXPECT_EQ(calls, 1u);
}

TEST(TransferBufferSafetyTrackerTest, LastProofReportsOutsideTrackerLockAndKeepsTrackerAlive) {
    size_t calls = 0;
    std::weak_ptr<TransferBufferSafetyTracker> weak;
    auto tracker = std::make_shared<TransferBufferSafetyTracker>([&] {
        ++calls;
        const auto active = weak.lock();
        ASSERT_NE(active, nullptr);
        active->finishSubmitting();
    });
    weak = tracker;
    auto proof = tracker->addBatch();
    tracker->finishSubmitting();
    tracker.reset();
    ASSERT_FALSE(weak.expired());
    proof->report(true);
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(weak.expired());
}

TEST(TransferBufferSafetyTrackerTest, SubmissionCloseRacingConcurrentProofsReportsOnce) {
    std::atomic<size_t> calls{0};
    auto tracker = std::make_shared<TransferBufferSafetyTracker>([&] { ++calls; });
    std::vector<std::shared_ptr<TransferBufferSafety>> proofs;
    for (size_t i = 0; i < 64; ++i) {
        proofs.push_back(tracker->addBatch());
    }
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (size_t j = i; j < proofs.size(); j += 8) {
                proofs[j]->report(false);
                proofs[j]->report(true);
                proofs[j]->report(true);
            }
        });
    }
    start.store(true, std::memory_order_release);
    tracker->finishSubmitting();
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(calls.load(), 1u);
}

}  // namespace
}  // namespace rtp_llm
