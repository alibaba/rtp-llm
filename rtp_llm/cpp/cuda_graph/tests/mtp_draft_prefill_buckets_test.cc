#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/mtp_draft_prefill_buckets.h"

namespace rtp_llm {

// The MTP draft prefill capture set is only reachable in production (isMtpDraftPrefillCudaGraph()
// requires num_tokens_per_bs != max_seq_len, which the python test harness cannot express), so the
// capture-set construction is covered here.
TEST(MtpDraftPrefillBucketsTest, CoversEveryBatchSize) {
    // One capture key per batch size 1..max_bs. A sparse set (e.g. powers of two) is unsafe:
    // the prefill graph bakes in the captured batch's attention layout, so a draft batch with no
    // exact key replays into a larger key's graph and reads out of bounds.
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/8, /*num_tokens_per_bs=*/1),
              std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8}));
    // num_tokens_per_bs scales every key: the capture keys are sequence lengths.
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/4, /*num_tokens_per_bs=*/3),
              std::vector<int>({3, 6, 9, 12}));
}

TEST(MtpDraftPrefillBucketsTest, CaptureCountEqualsMaxBatch) {
    // One graph per batch size, so the capture count grows linearly with CONCURRENCY_LIMIT --
    // the intended, correctness-required cost (see header).
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/128, /*num_tokens_per_bs=*/2).size(), 128u);
    EXPECT_EQ(mtpDraftPrefillCaptureSeqLens(/*max_bs=*/256, /*num_tokens_per_bs=*/2).size(), 256u);
}

TEST(MtpDraftPrefillBucketsTest, KeysAreStrictlyAscendingAndComplete) {
    // tryGetRealGraphPrefillSeqLen() picks a graph with lower_bound, which requires a sorted
    // range; and every k*num_tokens_per_bs in [1..max_bs] must be present so any reachable draft
    // batch finds an exact-layout graph rather than falling back to a mismatched one.
    for (size_t max_bs : {1u, 2u, 3u, 5u, 17u, 31u, 32u, 33u, 64u}) {
        const int  ntpb     = 4;
        const auto seq_lens = mtpDraftPrefillCaptureSeqLens(max_bs, ntpb);
        ASSERT_EQ(seq_lens.size(), max_bs) << "max_bs " << max_bs;
        EXPECT_TRUE(std::is_sorted(seq_lens.begin(), seq_lens.end())) << "max_bs " << max_bs;
        for (size_t k = 0; k < seq_lens.size(); ++k) {
            EXPECT_EQ(seq_lens[k], static_cast<int>(k + 1) * ntpb) << "max_bs " << max_bs << " k " << k;
        }
    }
}

TEST(MtpDraftPrefillBucketsTest, SingleBatchProducesOneKey) {
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
