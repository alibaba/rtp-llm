#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"
#include "gtest/gtest.h"

namespace rtp_llm {

class LinearBlocksUtilsTest: public ::testing::Test {
protected:
};

TEST_F(LinearBlocksUtilsTest, testGetCachedTokenBlockSwapIdx) {
    int cached_src_block_idx, cached_des_block_idx;

    // normal case 1
    // cur_seq_length = 198, nxt_seq_length = 202, seq_size_per_block = 100
    // [-1, 199, 200, 201, 202] -> [-1, 200, 199, 201, 202]
    // return [2, 1]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(198, 202, 100);
    ASSERT_EQ(cached_src_block_idx, 2);
    ASSERT_EQ(cached_des_block_idx, 1);

    // normal case 2
    // cur_seq_length = 97, nxt_seq_length = 101, seq_size_per_block = 100
    // [98, 99, 100, 101] -> [100, 99, 98, 101]
    // return [2, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(97, 101, 100);
    ASSERT_EQ(cached_src_block_idx, 2);
    ASSERT_EQ(cached_des_block_idx, 0);

    // normal case 3
    // cur_seq_length = 397, nxt_seq_length = 403, seq_size_per_block = 100
    // [-1, -1, -1, 398, 399, 400, 401, 402, 403] -> [-1, -1, -1, 400, 399, 398, 401, 402, 403]
    // return [5, 3]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(397, 403, 100);
    ASSERT_EQ(cached_src_block_idx, 5);
    ASSERT_EQ(cached_des_block_idx, 3);

    // normal case 4
    // cur_seq_length = 50, nxt_seq_length = 54, seq_size_per_block = 100
    // [51, 52, 53, 54] -> [51, 52, 53, 54]
    // return [0, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(50, 54, 100);
    ASSERT_EQ(cached_src_block_idx, 0);
    ASSERT_EQ(cached_des_block_idx, 0);

    // around corner case 1
    // cur_seq_length = 95, nxt_seq_length = 99, seq_size_per_block = 100
    // [96, 97, 98, 99] -> [96, 97, 98, 99]
    // return [0, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(95, 99, 100);
    ASSERT_EQ(cached_src_block_idx, 0);
    ASSERT_EQ(cached_des_block_idx, 0);

    // around corner case 2
    // cur_seq_length = 100, nxt_seq_length = 104, seq_size_per_block = 100
    // [101, 102, 103, 104] -> [101, 102, 103, 104]
    // return [0, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(100, 104, 100);
    ASSERT_EQ(cached_src_block_idx, 0);
    ASSERT_EQ(cached_des_block_idx, 0);

    // corner case 1, cached token is the first token
    // cur_seq_length = 99, nxt_seq_length = 103, seq_size_per_block = 100
    // [100, 101, 102, 103] -> 100, 101, 102, 103]
    // return [0, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(99, 103, 100);
    ASSERT_EQ(cached_src_block_idx, 0);
    ASSERT_EQ(cached_des_block_idx, 0);

    // corner case 2, cached token is the last token
    // cur_seq_length = 96, nxt_seq_length = 100, seq_size_per_block = 100
    // [97, 98, 99, 100] -> [97, 98, 99, 100]
    // return [0, 0]
    std::tie(cached_src_block_idx, cached_des_block_idx) = getCachedTokenBlockSwapIdx(96, 100, 100);
    ASSERT_EQ(cached_src_block_idx, 0);
    ASSERT_EQ(cached_des_block_idx, 0);
}

TEST_F(LinearBlocksUtilsTest, testGetFinalTokenBlockSwapIdx) {
    int src_block_idx, des_block_idx;

    // normal case
    // cur_seq_length = 249, nxt_seq_length = 252, seq_size_per_block = 100
    // [-1, -1, 250, 251, 252] -> [-1, -1, 252, 251, 250]
    // return [4, 2]
    std::tie(src_block_idx, des_block_idx) = getFinalTokenBlockSwapIdx(249, 252, 100);
    ASSERT_EQ(src_block_idx, 4);
    ASSERT_EQ(des_block_idx, 2);

    // corner case 1
    // cur_seq_length = 200, nxt_seq_length = 205, seq_size_per_block = 100
    // [-1, 200, 201, 202, 203, 204, 205] -> [-1, 200, 205, 202, 203, 204, 201]
    // return [6, 2]
    std::tie(src_block_idx, des_block_idx) = getFinalTokenBlockSwapIdx(200, 205, 100);
    ASSERT_EQ(src_block_idx, 6);
    ASSERT_EQ(des_block_idx, 2);

    // corner case 2
    // cur_seq_length = 197, nxt_seq_length = 200, seq_size_per_block = 100
    // [-1, 198, 199, 200] -> [-1, 200, 199, 198]
    // return [3, 1]
    std::tie(src_block_idx, des_block_idx) = getFinalTokenBlockSwapIdx(197, 200, 100);
    ASSERT_EQ(src_block_idx, 3);
    ASSERT_EQ(des_block_idx, 1);
}

}  // namespace rtp_llm