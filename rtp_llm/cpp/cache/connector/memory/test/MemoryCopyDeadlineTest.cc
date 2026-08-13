#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyDeadline.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyFence.h"

#include <chrono>
#include <limits>
#include <thread>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

TEST(MemoryCopyDeadlineTest, CopyRetentionCoversDeadlineAndSafetyWindow) {
    const auto decision = MemoryCopyDeadline::evaluateCopy(
        /*operation_deadline_unix_ms=*/1100,
        /*requested_retention_ms=*/10,
        /*safety_window_ms=*/50,
        /*now_unix_ms=*/1000);

    ASSERT_TRUE(decision) << decision.error;
    EXPECT_EQ(decision.retention, 150ms);
}

TEST(MemoryCopyDeadlineTest, RequestDeadlineCapsLocalCopyDeadline) {
    EXPECT_EQ(MemoryCopyDeadline::resolve(/*now_unix_ms=*/1000,
                                          /*local_timeout_ms=*/100,
                                          /*request_deadline_unix_ms=*/1050),
              1050);
    EXPECT_EQ(MemoryCopyDeadline::resolve(/*now_unix_ms=*/1000,
                                          /*local_timeout_ms=*/100,
                                          /*request_deadline_unix_ms=*/1200),
              1100);
    EXPECT_EQ(MemoryCopyDeadline::resolve(/*now_unix_ms=*/1000,
                                          /*local_timeout_ms=*/100,
                                          /*request_deadline_unix_ms=*/0),
              1100);
}

TEST(MemoryCopyDeadlineTest, ExpiredRequestDeadlineFailsBeforeCopyAdmission) {
    EXPECT_EQ(MemoryCopyDeadline::resolve(/*now_unix_ms=*/1000,
                                          /*local_timeout_ms=*/100,
                                          /*request_deadline_unix_ms=*/1000),
              0);
    EXPECT_EQ(MemoryCopyDeadline::resolve(/*now_unix_ms=*/1000,
                                          /*local_timeout_ms=*/100,
                                          /*request_deadline_unix_ms=*/999),
              0);
}

TEST(MemoryCopyDeadlineTest, CopyRpcTimeoutUsesRemainingRequestBudget) {
    EXPECT_EQ(MemoryCopyDeadline::rpcTimeout(/*operation_deadline_unix_ms=*/1050,
                                             /*local_timeout_ms=*/100,
                                             /*now_unix_ms=*/1000),
              50);
    EXPECT_EQ(MemoryCopyDeadline::rpcTimeout(/*operation_deadline_unix_ms=*/1200,
                                             /*local_timeout_ms=*/100,
                                             /*now_unix_ms=*/1000),
              100);
    EXPECT_EQ(MemoryCopyDeadline::rpcTimeout(/*operation_deadline_unix_ms=*/1000,
                                             /*local_timeout_ms=*/100,
                                             /*now_unix_ms=*/1000),
              0);
}

TEST(MemoryCopyDeadlineTest, ExpiredCopyIsRejectedAfterItsTombstoneWasPruned) {
    const std::string operation_id = "copy-a";
    MemoryCopyFence fence;
    ASSERT_TRUE(fence.sealAndWait(operation_id, 100ms, 1ms));
    std::this_thread::sleep_for(5ms);

    auto prune_trigger = fence.begin("prune-trigger", 1s);
    ASSERT_TRUE(prune_trigger) << prune_trigger.error;
    EXPECT_EQ(fence.entryCountForTest(), 1u);

    const auto expired = MemoryCopyDeadline::evaluateCopy(
        /*operation_deadline_unix_ms=*/100,
        /*requested_retention_ms=*/1,
        /*safety_window_ms=*/10,
        /*now_unix_ms=*/101);
    EXPECT_FALSE(expired);
    EXPECT_EQ(expired.error, "memory copy operation deadline has expired");

    // Admission checks the absolute deadline before calling begin(), so the old
    // operation cannot reappear after its relative tombstone has been pruned.
    if (expired) {
        (void)fence.begin(operation_id, expired.retention);
    }
    EXPECT_EQ(fence.entryCountForTest(), 1u);
}

TEST(MemoryCopyDeadlineTest, ExpiredOperationCanStillBeQuiesced) {
    const auto decision = MemoryCopyDeadline::evaluateQuiesce(
        /*operation_deadline_unix_ms=*/100,
        /*requested_retention_ms=*/20,
        /*safety_window_ms=*/10,
        /*now_unix_ms=*/200);

    ASSERT_TRUE(decision) << decision.error;
    EXPECT_EQ(decision.retention, 20ms);
}

TEST(MemoryCopyDeadlineTest, RejectsInvalidAndOverflowingInputs) {
    EXPECT_EQ(MemoryCopyDeadline::make(100, 0), 0);
    EXPECT_EQ(MemoryCopyDeadline::make(std::numeric_limits<int64_t>::max(), 1), 0);
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(0, 1, 1, 0));
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(1, 0, 1, 0));
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(1, 1, 0, 0));
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(std::numeric_limits<int64_t>::max(), 1, 1, 0));
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(
        MemoryCopyDeadline::kMaxWireDurationMs + 1, 1, 1, 0));
    EXPECT_FALSE(MemoryCopyDeadline::evaluateCopy(
        1, MemoryCopyDeadline::kMaxWireDurationMs + 1, 1, 0));
}

}  // namespace
}  // namespace rtp_llm
