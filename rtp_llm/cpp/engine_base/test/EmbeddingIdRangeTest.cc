#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/EmbeddingIdRange.h"

namespace rtp_llm {

class EmbeddingIdRangeTest: public ::testing::Test {};

TEST_F(EmbeddingIdRangeTest, testMaskedImageSlotsNormalizeTokenTypeSentinel) {
    std::vector<int>       types     = {-1};
    const std::vector<int> text_mask = {0};
    normalizeMaskedTokenTypeIds(types.data(), text_mask.data(), 1, 1);
    EXPECT_EQ(types[0], 0);
}

TEST_F(EmbeddingIdRangeTest, testTextTokenTypeIdsAreNotNormalized) {
    std::vector<int>       types     = {-1, 1};
    const std::vector<int> text_mask = {1, 0};
    normalizeMaskedTokenTypeIds(types.data(), text_mask.data(), 2, 2);
    EXPECT_EQ(types[0], -1);
    EXPECT_EQ(types[1], 0);
}

TEST_F(EmbeddingIdRangeTest, testRejectsShortMask) {
    const std::vector<int> text_mask = {0};
    const auto             status = validateTextTokensMaskLength(6, text_mask.data(), 1, 2);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("text_tokens_mask length 1 does not match input length 2"),
              std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testRejectsLongMaskForZeroLengthInput) {
    const std::vector<int> text_mask = {0};
    const auto             status = validateTextTokensMaskLength(7, text_mask.data(), 1, 0);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("text_tokens_mask length 1 does not match input length 0"),
              std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testAbsentMaskDoesNotRequireLength) {
    EXPECT_TRUE(validateTextTokensMaskLength(0, nullptr, 0, 2).ok());
}

}  // namespace rtp_llm
