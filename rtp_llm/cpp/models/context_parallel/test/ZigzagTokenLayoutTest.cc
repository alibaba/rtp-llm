#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(ZigzagTokenLayoutTest, ComputesPaddingAndPerRankTokens) {
    EXPECT_EQ(makeZigzagTokenLayout(0, 4).padded_token_count, 0);

    const auto short_layout = makeZigzagTokenLayout(1, 4);
    EXPECT_EQ(short_layout.padded_token_count, 8);
    EXPECT_EQ(short_layout.padding_token_count, 7);
    EXPECT_EQ(short_layout.token_count_per_rank, 2);

    const auto aligned_layout = makeZigzagTokenLayout(8, 4);
    EXPECT_EQ(aligned_layout.padded_token_count, 8);
    EXPECT_EQ(aligned_layout.padding_token_count, 0);
    EXPECT_EQ(aligned_layout.token_count_per_rank, 2);

    const auto long_layout = makeZigzagTokenLayout(200002, 4);
    EXPECT_EQ(long_layout.padded_token_count, 200008);
    EXPECT_EQ(long_layout.padding_token_count, 6);
    EXPECT_EQ(long_layout.token_count_per_rank, 50002);
}

TEST(ZigzagTokenLayoutTest, RejectsZeroCpSize) {
    EXPECT_THROW(makeZigzagTokenLayout(1, 0), std::invalid_argument);
}

TEST(ZigzagTokenLayoutTest, AlignsEveryLocalSegment) {
    const auto short_layout = makeZigzagTokenLayout(1, 4, 64);
    EXPECT_EQ(short_layout.padded_token_count, 512);
    EXPECT_EQ(short_layout.padding_token_count, 511);
    EXPECT_EQ(short_layout.token_count_per_rank, 128);

    const auto partial_layout = makeZigzagTokenLayout(257, 2, 64);
    EXPECT_EQ(partial_layout.padded_token_count, 512);
    EXPECT_EQ(partial_layout.padding_token_count, 255);
    EXPECT_EQ(partial_layout.token_count_per_rank, 256);

    const auto aligned_layout = makeZigzagTokenLayout(512, 4, 64);
    EXPECT_EQ(aligned_layout.padded_token_count, 512);
    EXPECT_EQ(aligned_layout.padding_token_count, 0);
    EXPECT_EQ(aligned_layout.token_count_per_rank, 128);

    const auto arbitrary_layout = makeZigzagTokenLayout(75, 2, 37);
    EXPECT_EQ(arbitrary_layout.padded_token_count, 148);
    EXPECT_EQ(arbitrary_layout.padding_token_count, 73);
    EXPECT_EQ(arbitrary_layout.token_count_per_rank, 74);
}

TEST(ZigzagTokenLayoutTest, RejectsInvalidOrOverflowingAlignment) {
    EXPECT_THROW(makeZigzagTokenLayout(1, 2, 0), std::invalid_argument);
    EXPECT_THROW(makeZigzagTokenLayout(1, 4, std::numeric_limits<size_t>::max()), std::overflow_error);
    EXPECT_THROW(makeZigzagTokenLayout(std::numeric_limits<size_t>::max(), 1, 1), std::overflow_error);
}

}  // namespace rtp_llm
