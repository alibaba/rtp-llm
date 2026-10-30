#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cuda_graph/prefill_padding_offset_tail.h"

// Contract under test for rebuildPrefillPaddingOffsetTail:
//   - Invalid arguments (null buffer, zero capacity, live_tokens <= 0 or
//     live_tokens > capacity) return false without writing any memory.
//   - live_tokens == capacity is an identity fast path: return true without
//     writing any memory.
//   - Otherwise only the tail rows [live_tokens, capacity) are rewritten as
//     padding_offset[row] = last_live_tgt - row, where
//     last_live_tgt = (live_tokens - 1) + padding_offset[live_tokens - 1], so
//     every tail row decodes to the last live row. The live prefix
//     [0, live_tokens) is never modified.
namespace rtp_llm {
namespace {

TEST(PrefillPaddingOffsetTailTest, RejectsInvalidArgumentsWithoutWriting) {
    const std::vector<int32_t> sentinel{7, 7, 7, 7};

    // Null buffer.
    EXPECT_FALSE(rebuildPrefillPaddingOffsetTail(nullptr, sentinel.size(), 2));

    // Zero capacity and zero live tokens.
    std::vector<int32_t> padding_offset = sentinel;
    EXPECT_FALSE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), 0, 0));
    EXPECT_EQ(padding_offset, sentinel);

    // Zero live tokens with capacity.
    padding_offset = sentinel;
    EXPECT_FALSE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), sentinel.size(), 0));
    EXPECT_EQ(padding_offset, sentinel);

    // Live tokens beyond capacity.
    padding_offset = sentinel;
    EXPECT_FALSE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), sentinel.size(), sentinel.size() + 1));
    EXPECT_EQ(padding_offset, sentinel);
}

TEST(PrefillPaddingOffsetTailTest, LiveEqualsCapacityIsIdentity) {
    const std::vector<int32_t> input{0, 0, 2, 2, 2, 2};
    std::vector<int32_t>       padding_offset = input;

    EXPECT_TRUE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), padding_offset.size(), padding_offset.size()));
    EXPECT_EQ(padding_offset, input);
}

TEST(PrefillPaddingOffsetTailTest, RebuildsStaleTailAfterBatchShrink) {
    // Capture width 8. Previous round ran batch=2 with slot0 holding 1 token
    // and slot1 holding 4 tokens at stride 4: live rows 0..4 decode to
    // tgt 0,4,5,6,7. Rows 5..7 keep the residue of the previous larger batch
    // (decoding to tgt 8..10), which points consumers past the live page
    // table.
    std::vector<int32_t> padding_offset{0, 3, 3, 3, 3, 3, 3, 3};

    ASSERT_TRUE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), padding_offset.size(), 5));
    // last_live_tgt = (5 - 1) + padding_offset[4] = 4 + 3 = 7, so rows 5..7
    // decode to tgt 7: padding_offset[row] = 7 - row.
    EXPECT_EQ(padding_offset, (std::vector<int32_t>{0, 3, 3, 3, 3, 2, 1, 0}));

    for (size_t row = 5; row < padding_offset.size(); ++row) {
        EXPECT_EQ(static_cast<int32_t>(row) + padding_offset[row], 7);
    }

    // Idempotent: replaying the rebuild keeps the same layout.
    ASSERT_TRUE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), padding_offset.size(), 5));
    EXPECT_EQ(padding_offset, (std::vector<int32_t>{0, 3, 3, 3, 3, 2, 1, 0}));
}

TEST(PrefillPaddingOffsetTailTest, MapsWholeTailToLastLiveRowWhenBatchShrinksToOne) {
    // Capture width 4, live batch shrank to a single token decoding to tgt 0;
    // rows 1..3 still hold the previous round residue.
    std::vector<int32_t> padding_offset{0, 3, 3, 3};

    ASSERT_TRUE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), padding_offset.size(), 1));
    // last_live_tgt = 0 + padding_offset[0] = 0, so every tail row decodes to
    // tgt 0: padding_offset[row] = 0 - row.
    EXPECT_EQ(padding_offset, (std::vector<int32_t>{0, -1, -2, -3}));

    for (size_t row = 1; row < padding_offset.size(); ++row) {
        EXPECT_EQ(static_cast<int32_t>(row) + padding_offset[row], 0);
    }
}

TEST(PrefillPaddingOffsetTailTest, KeepsMixedStrideLivePrefixAndAnchorsTailAtLastLiveRow) {
    // Capture width 12. Live batch holds two slots at stride 6: slot0 has 3
    // tokens (tgt 0..2), slot1 has 5 tokens (tgt 6..10); rows 8..11 keep stale
    // residue decoding to tgt 17..20.
    std::vector<int32_t> padding_offset{0, 0, 0, 3, 3, 3, 3, 3, 9, 9, 9, 9};

    ASSERT_TRUE(rebuildPrefillPaddingOffsetTail(padding_offset.data(), padding_offset.size(), 8));
    // last_live_tgt = (8 - 1) + padding_offset[7] = 7 + 3 = 10, so rows 8..11
    // decode to tgt 10: padding_offset[row] = 10 - row.
    EXPECT_EQ(padding_offset, (std::vector<int32_t>{0, 0, 0, 3, 3, 3, 3, 3, 2, 1, 0, -1}));

    for (size_t row = 8; row < padding_offset.size(); ++row) {
        EXPECT_EQ(static_cast<int32_t>(row) + padding_offset[row], 10);
    }
}

}  // namespace
}  // namespace rtp_llm
