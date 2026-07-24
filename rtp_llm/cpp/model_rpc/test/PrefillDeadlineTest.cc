#include <gtest/gtest.h>

#include <cstdint>

#include "rtp_llm/cpp/model_rpc/PrefillDeadlineUtils.h"

namespace rtp_llm {

// ===========================================================================
// computeRemainingTimeoutMs
// ---------------------------------------------------------------------------
// Pure helper extracted from remoteAllocateResource.  It deducts the time
// already spent in prefill (request_begin_time_us -> current_time_us) from the
// configured RPC timeout so the decode-side gRPC deadline reflects only the
// remaining budget.
//
// Because current_time_us is an explicit parameter (not read from the wall
// clock), every test case below is fully deterministic.
// ===========================================================================

// 1000 ms budget, 100 ms elapsed -> 900 ms remaining.
TEST(PrefillDeadlineTest, DeductsElapsedWhenTimeoutPositive) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/1000,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/1'100'000),
              900);
}

// 100 ms budget, 200 ms elapsed -> clamped to 0.
TEST(PrefillDeadlineTest, ClampsToZeroWhenElapsedExceedsTimeout) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/100,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/1'200'000),
              0);
}

// Boundary: elapsed exactly equals timeout -> 0.
TEST(PrefillDeadlineTest, ReturnsZeroWhenElapsedEqualsTimeout) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/100,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/1'100'000),
              0);
}

// timeout_ms == 0 is the "no deadline" sentinel -> returned unchanged.
TEST(PrefillDeadlineTest, NoDeductionWhenTimeoutZero) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/0,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/2'000'000),
              0);
}

// Negative timeout is a "disabled" sentinel -> returned unchanged.
TEST(PrefillDeadlineTest, NoDeductionWhenTimeoutNegative) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/-1,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/2'000'000),
              -1);
}

// current == begin -> elapsed 0 -> full timeout.
TEST(PrefillDeadlineTest, NoElapsedTimeReturnsFullTimeout) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/500,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/1'000'000),
              500);
}

// 999 us elapsed -> 0 ms after integer division -> full timeout.
TEST(PrefillDeadlineTest, SubMillisecondElapsedTruncatesToZero) {
    EXPECT_EQ(computeRemainingTimeoutMs(
                  /*timeout_ms=*/500,
                  /*request_begin_time_us=*/1'000'000,
                  /*current_time_us=*/1'000'999),
              500);
}

}  // namespace rtp_llm
