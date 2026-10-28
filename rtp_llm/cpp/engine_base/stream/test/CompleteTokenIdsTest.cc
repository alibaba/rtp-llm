#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "gtest/gtest.h"

namespace rtp_llm {
std::shared_ptr<GenerateInput> tokenTestInput() {
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::tensor({1, 2}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    return input;
}
TEST(CompleteTokenIdsTest, ClampedSpeculativeRowsKeepOriginalStride) {
    CompleteTokenIds tokens(2, 2, 16, 1);
    tokens.init(tokenTestInput());
    int invalid = -1;
    ASSERT_TRUE(
        tokens.update(torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt32), 0, 3, 2, 4, 100, false, 0, invalid));
    EXPECT_EQ(tokens.completeTokenIdsVec(0), std::vector<int>({1, 2, 10, 11}));
    EXPECT_EQ(tokens.completeTokenIdsVec(1), std::vector<int>({1, 2, 20, 21}));
    const auto time = tokens.firstTokenTimeUs();
    ASSERT_TRUE(tokens.update(torch::tensor({{13}, {23}}, torch::kInt32), 0, 1, 2, 4, 100, false, 0, invalid));
    EXPECT_EQ(tokens.seqLength(), 4);
    EXPECT_EQ(tokens.firstTokenTimeUs(), time);
}
TEST(CompleteTokenIdsTest, ContentBudgetClosesAcrossChunkBoundaryAndTrimsOvershoot) {
    CompleteTokenIds tokens(1, 1, 16, 1);
    tokens.init(tokenTestInput());
    int invalid = -1;
    ASSERT_TRUE(tokens.update(torch::tensor({{7}}, torch::kInt32), 0, 1, 2, 16, 100, false, 0, invalid));
    EXPECT_FALSE(tokens.matchThinkEndToken(0, {7, 8}, 2, 1));
    ASSERT_TRUE(tokens.update(torch::tensor({{8, 9, 10}}, torch::kInt32), 0, 3, 2, 16, 100, false, 0, invalid));
    EXPECT_TRUE(tokens.matchThinkEndToken(0, {7, 8}, 2, 1));
    EXPECT_EQ(tokens.completeTokenIdsVec(0), std::vector<int>({1, 2, 7, 8, 9}));
    EXPECT_EQ(tokens.commonSeqLength(), 5);
}
}  // namespace rtp_llm
