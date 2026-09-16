#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_prefill_utils.h"
#include "gtest/gtest.h"

namespace rtp_llm {

TEST(CudaGraphPrefillCaptureRangeTest, UsesDecodeBatchBucketsForMtpDraftPrefill) {
    const auto capture_seq_lens =
        buildMtpDraftPrefillCaptureSequenceLengths({1, 2, 4, 8, 16, 32}, 32, 5);

    EXPECT_EQ(capture_seq_lens, (std::vector<int>{5, 10, 20, 40, 80, 160}));

    for (int requested_seq_len = 1; requested_seq_len <= 160; ++requested_seq_len) {
        const auto capture = std::lower_bound(capture_seq_lens.begin(), capture_seq_lens.end(), requested_seq_len);
        ASSERT_NE(capture, capture_seq_lens.end()) << "requested_seq_len=" << requested_seq_len;
        EXPECT_GE(*capture, requested_seq_len);
        if (capture != capture_seq_lens.begin()) {
            EXPECT_LT(*(capture - 1), requested_seq_len);
        }
    }
    EXPECT_EQ(std::lower_bound(capture_seq_lens.begin(), capture_seq_lens.end(), 161), capture_seq_lens.end());
}

TEST(CudaGraphPrefillCaptureRangeTest, IgnoresBucketsAboveDraftPrefillCapacity) {
    EXPECT_EQ(buildMtpDraftPrefillCaptureSequenceLengths({32, 1, 8, 8, 64}, 32, 5),
              (std::vector<int>{5, 40, 160}));
}

TEST(CudaGraphPrefillCaptureRangeTest, RejectsInvalidOrOverflowingCaptureCapacity) {
    EXPECT_THROW(buildMtpDraftPrefillCaptureSequenceLengths({1}, 0, 5), std::invalid_argument);
    EXPECT_THROW(buildMtpDraftPrefillCaptureSequenceLengths({1}, 32, 0), std::invalid_argument);
    EXPECT_THROW(buildMtpDraftPrefillCaptureSequenceLengths(
                     {1}, static_cast<size_t>(std::numeric_limits<int>::max()) + 1, 1),
                 std::overflow_error);
    EXPECT_THROW(buildMtpDraftPrefillCaptureSequenceLengths(
                     {1}, static_cast<size_t>(std::numeric_limits<int>::max() / 2) + 1, 2),
                 std::overflow_error);
}

}  // namespace rtp_llm
