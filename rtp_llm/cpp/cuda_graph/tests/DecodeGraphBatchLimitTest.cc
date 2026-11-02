#include "rtp_llm/cpp/cuda_graph/DecodeGraphBatchLimit.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(DecodeGraphBatchLimitTest, ExplicitCaptureBoundsTheWhaleScheduler) {
    // Whale admitted batches 65 and 66 with max_generate_batch_size=96,
    // although its only captured graph had capacity 64.
    const int admission_limit = std::min(96, decodeGraphBatchLimit(96, {64}));
    EXPECT_EQ(admission_limit, 64);
    EXPECT_LE(64, admission_limit);
    EXPECT_GT(65, admission_limit);
    EXPECT_GT(66, admission_limit);
}

TEST(DecodeGraphBatchLimitTest, UsesLargestCaptureRegardlessOfOrder) {
    EXPECT_EQ(decodeGraphBatchLimit(96, {32, 1, 64, 8, 64}), 64);
    EXPECT_EQ(std::min(32, decodeGraphBatchLimit(96, {64})), 32);
    EXPECT_EQ(decodeGraphBatchLimit(96, {}), 96);
}

TEST(DecodeGraphBatchLimitTest, RejectsCaptureOutsideAllocatedBuffers) {
    EXPECT_THROW(decodeGraphBatchLimit(64, {96}), std::invalid_argument);
    EXPECT_THROW(decodeGraphBatchLimit(96, {0, 64}), std::invalid_argument);
    EXPECT_THROW(decodeGraphBatchLimit(96, {-1, 64}), std::invalid_argument);
    EXPECT_THROW(decodeGraphBatchLimit(0, {}), std::invalid_argument);
}

}  // namespace rtp_llm
