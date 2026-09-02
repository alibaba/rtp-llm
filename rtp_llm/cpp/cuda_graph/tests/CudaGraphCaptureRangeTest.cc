#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_capture_range.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphCaptureRangeTest, DraftPrefillUsesDecodeBatchBuckets) {
    const std::vector<int> decode_batch_sizes{1, 4, 8, 16, 24, 32, 64};
    const std::vector<int> expected_seq_lens{4, 16, 32, 64, 96, 128, 256};

    EXPECT_EQ(draftPrefillCaptureSeqLens(decode_batch_sizes, 4), expected_seq_lens);
}

TEST(CudaGraphCaptureRangeTest, EmptyDecodeRangeStaysEmpty) {
    EXPECT_TRUE(draftPrefillCaptureSeqLens({}, 4).empty());
}

}  // namespace
}  // namespace rtp_llm
