#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/mtp_draft_prefill_buckets.h"

namespace rtp_llm {

// The MTP draft prefill capture set is only reachable in production (isMtpDraftPrefillCudaGraph()
// requires num_tokens_per_bs != max_seq_len, which the python test harness cannot express), so the
// bucketing itself is covered here.
TEST(MtpDraftPrefillBucketsTest, BucketsArePowersOfTwoPlusMaxBatch) {
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/32, /*num_tokens_per_bs=*/1),
              std::vector<int>({1, 2, 4, 8, 16, 32}));
    // num_tokens_per_bs scales every bucket: the capture keys are sequence lengths.
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/8, /*num_tokens_per_bs=*/3),
              std::vector<int>({3, 6, 12, 24}));
}

TEST(MtpDraftPrefillBucketsTest, CaptureCountGrowsLogarithmicallyWithConcurrencyLimit) {
    // The point of bucketing: one graph per batch size made the capture count -- and the
    // device memory it holds -- grow linearly with CONCURRENCY_LIMIT.
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/128, /*num_tokens_per_bs=*/2).size(), 8u);
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/256, /*num_tokens_per_bs=*/2).size(), 9u);
}

TEST(MtpDraftPrefillBucketsTest, BucketsAreStrictlyAscendingSoLowerBoundCanReplay) {
    // tryGetRealGraphPrefillSeqLen() picks a graph with lower_bound, which requires a sorted
    // range; a non-monotonic set would silently replay the wrong graph.
    for (size_t max_bs : {1u, 2u, 3u, 5u, 17u, 31u, 32u, 33u, 64u}) {
        const auto seq_lens = mtpDraftPrefillCaptureSeqLens(max_bs, /*num_tokens_per_bs=*/4);
        ASSERT_FALSE(seq_lens.empty()) << "max_bs " << max_bs;
        EXPECT_TRUE(std::is_sorted(seq_lens.begin(), seq_lens.end(), std::less_equal<int>()))
            << "buckets not strictly ascending for max_bs " << max_bs;
        // The largest batch keeps an exact graph, so it never pays padded compute.
        EXPECT_EQ(seq_lens.back(), static_cast<int>(max_bs) * 4) << "max_bs " << max_bs;
    }
}

TEST(MtpDraftPrefillBucketsTest, SingleBatchProducesOneBucket) {
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/1, /*num_tokens_per_bs=*/7), std::vector<int>({7}));
}

TEST(MtpDraftPrefillBucketsTest, DegenerateConfigCapturesNothing) {
    // max_bs == 0 must not register a seq_len=0 graph: capturePrefill() would capture a
    // zero-length slice and tryGetRealGraphPrefillSeqLen() would match key 0.
    EXPECT_TRUE(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/0, /*num_tokens_per_bs=*/8).empty());
    EXPECT_TRUE(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/8, /*num_tokens_per_bs=*/0).empty());
    EXPECT_TRUE(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/8, /*num_tokens_per_bs=*/-1).empty());
}

}  // namespace rtp_llm
