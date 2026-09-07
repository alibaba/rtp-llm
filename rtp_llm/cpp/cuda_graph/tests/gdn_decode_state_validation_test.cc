#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/gdn_decode_state_validation.h"

namespace rtp_llm {
namespace {

TEST(GdnDecodeStateValidationTest, AcceptsRealRowsAndIgnoresGraphPadding) {
    const auto blocks  = torch::tensor({{1, 2}, {0, 0}}, torch::kInt32);
    const auto lengths = torch::tensor({1024, 0}, torch::kInt32);

    EXPECT_TRUE(validateGdnDecodeStateBlockTable(blocks, lengths, 1, 1024, 3).empty());
    EXPECT_NE(validateGdnDecodeStateBlockTable(blocks, lengths, 2, 1024, 3).find("non-positive"),
              std::string::npos);
}

TEST(GdnDecodeStateValidationTest, RejectsInvalidRealStateBlockIds) {
    const auto lengths = torch::tensor({1024}, torch::kInt32);

    EXPECT_NE(validateGdnDecodeStateBlockTable(torch::tensor({{1, 0}}, torch::kInt32), lengths, 1, 1024, 3)
                  .find("invalid state block ID"),
              std::string::npos);
    EXPECT_NE(validateGdnDecodeStateBlockTable(torch::tensor({{1, 3}}, torch::kInt32), lengths, 1, 1024, 3)
                  .find("invalid state block ID"),
              std::string::npos);
}

TEST(GdnDecodeStateValidationTest, RejectsRealRequestPastBlockTableWidth) {
    const auto blocks  = torch::tensor({{1, 2}}, torch::kInt32);
    const auto lengths = torch::tensor({2048}, torch::kInt32);

    EXPECT_NE(validateGdnDecodeStateBlockTable(blocks, lengths, 1, 1024, 4).find("block-table width"),
              std::string::npos);
}

TEST(GdnDecodeStateValidationTest, DisabledContractLeavesOtherModelsUnchanged) {
    EXPECT_TRUE(validateGdnDecodeStateBlockTable({}, {}, 1, 1024, 0).empty());
}

}  // namespace
}  // namespace rtp_llm
