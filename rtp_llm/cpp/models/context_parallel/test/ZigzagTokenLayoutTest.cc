#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

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

}  // namespace rtp_llm
