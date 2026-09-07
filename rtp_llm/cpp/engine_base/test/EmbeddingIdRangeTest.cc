#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/EmbeddingIdRange.h"

namespace rtp_llm {

class EmbeddingIdRangeTest: public ::testing::Test {};

TEST_F(EmbeddingIdRangeTest, testAcceptsInRangeIds) {
    const std::vector<int> tokens    = {0, 1, 7};
    const std::vector<int> types     = {0, 1, 0};
    const std::vector<int> text_mask = {1, 1, 1};
    EXPECT_TRUE(validateEmbeddingIdRanges(0, tokens.data(), types.data(), text_mask.data(), 3, 3, 8, 2).ok());
}

TEST_F(EmbeddingIdRangeTest, testRejectsTokenIdAtOrAboveVocabSize) {
    const std::vector<int> tokens = {0, 8};
    const auto             status = validateEmbeddingIdRanges(3, tokens.data(), nullptr, nullptr, 0, 2, 8, 0);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("stream [3] token_id 8 exceed vocab_size 8"), std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testRejectsNegativeTokenId) {
    const std::vector<int> tokens = {-1};
    const auto             status = validateEmbeddingIdRanges(4, tokens.data(), nullptr, nullptr, 0, 1, 8, 0);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("token_id -1 exceed vocab_size 8"), std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testRejectsOutOfRangeTokenTypeId) {
    const std::vector<int> tokens = {1, 1};
    const std::vector<int> types  = {0, 2};
    const auto             status = validateEmbeddingIdRanges(5, tokens.data(), types.data(), nullptr, 0, 2, 8, 2);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("token_type_id 2 exceed type_vocab_size 2"), std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testUnconfiguredBoundsDisableValidation) {
    const std::vector<int> tokens = {-5, 999};
    const std::vector<int> types  = {-1, 42};
    EXPECT_TRUE(validateEmbeddingIdRanges(0, tokens.data(), types.data(), nullptr, 0, 2, 0, 0).ok());
}

TEST_F(EmbeddingIdRangeTest, testMaskedImageSlotsExemptOnlyWordIds) {
    // The mask prevents the word-table lookup. The kernel still indexes the
    // token-type table, and MultimodalProcessor initializes image rows to zero.
    const std::vector<int> tokens    = {1, 123456, -3};
    const std::vector<int> types     = {0, 0, 0};
    const std::vector<int> text_mask = {1, 0, 0};
    EXPECT_TRUE(validateEmbeddingIdRanges(0, tokens.data(), types.data(), text_mask.data(), 3, 3, 8, 2).ok());
}

TEST_F(EmbeddingIdRangeTest, testMaskedImageSlotsNormalizeTokenTypeSentinelBeforeValidation) {
    const std::vector<int> tokens    = {123456};
    std::vector<int>       types     = {-1};
    const std::vector<int> text_mask = {0};
    const auto status = normalizeAndValidateEmbeddingIds(0, tokens.data(), types.data(), text_mask.data(), 1, 1, 8, 2);
    EXPECT_EQ(types[0], 0);
    EXPECT_TRUE(status.ok());
}

TEST_F(EmbeddingIdRangeTest, testNormalizeAndValidateRejectsInvalidTextTokenId) {
    const std::vector<int> tokens    = {8, 123456};
    std::vector<int>       types     = {0, -1};
    const std::vector<int> text_mask = {1, 0};
    const auto status = normalizeAndValidateEmbeddingIds(8, tokens.data(), types.data(), text_mask.data(), 2, 2, 8, 2);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("stream [8] token_id 8 exceed vocab_size 8"), std::string::npos);
    EXPECT_EQ(types[1], 0);
}

TEST_F(EmbeddingIdRangeTest, testNormalizeAndValidateRejectsInvalidTextTokenTypeId) {
    const std::vector<int> tokens    = {1, 123456};
    std::vector<int>       types     = {2, -1};
    const std::vector<int> text_mask = {1, 0};
    const auto status = normalizeAndValidateEmbeddingIds(9, tokens.data(), types.data(), text_mask.data(), 2, 2, 8, 2);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("stream [9] token_type_id 2 exceed type_vocab_size 2"),
              std::string::npos);
    EXPECT_EQ(types[1], 0);
}

TEST_F(EmbeddingIdRangeTest, testMalformedMaskDoesNotNormalizeTokenTypeIds) {
    const std::vector<int> tokens    = {1, 123456};
    std::vector<int>       types     = {-1, -1};
    const std::vector<int> text_mask = {0};
    const auto status = normalizeAndValidateEmbeddingIds(10, tokens.data(), types.data(), text_mask.data(), 1, 2, 8, 2);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(types[0], -1);
    EXPECT_EQ(types[1], -1);
}

TEST_F(EmbeddingIdRangeTest, testTextTokenTypeIdsAreNotNormalized) {
    std::vector<int>       types     = {-1, 1};
    const std::vector<int> text_mask = {1, 0};
    normalizeMaskedTokenTypeIds(types.data(), text_mask.data(), 2, 2);
    EXPECT_EQ(types[0], -1);
    EXPECT_EQ(types[1], 0);
}

TEST_F(EmbeddingIdRangeTest, testRejectsShortMask) {
    const std::vector<int> tokens    = {123456, 8};
    const std::vector<int> text_mask = {0};
    const auto             status = validateEmbeddingIdRanges(6, tokens.data(), nullptr, text_mask.data(), 1, 2, 8, 0);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("text_tokens_mask length 1 does not match input length 2"),
              std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testRejectsLongMaskForZeroLengthInput) {
    const std::vector<int> tokens    = {8};
    const std::vector<int> text_mask = {0};
    const auto             status = validateEmbeddingIdRanges(7, tokens.data(), nullptr, text_mask.data(), 1, 0, 8, 0);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("text_tokens_mask length 1 does not match input length 0"),
              std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testAbsentMaskDoesNotRequireLength) {
    const std::vector<int> tokens = {1, 2};
    EXPECT_TRUE(validateEmbeddingIdRanges(0, tokens.data(), nullptr, nullptr, 0, 2, 8, 0).ok());
}

TEST_F(EmbeddingIdRangeTest, testValidatesPositionIdsAgainstEmbeddingTableLength) {
    EXPECT_TRUE(validatePositionIdRange(31, 6, 2, 8).ok());

    const auto status = validatePositionIdRange(31, 7, 2, 8);
    ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(status.message()).find("stream [31]"), std::string::npos);
    EXPECT_NE(std::string(status.message()).find("position_embedding_count 8"), std::string::npos);
}

TEST_F(EmbeddingIdRangeTest, testRejectsInvalidPositionIdConfigurationWithoutOverflow) {
    EXPECT_FALSE(validatePositionIdRange(1, 1, 0, 0).ok());
    EXPECT_FALSE(validatePositionIdRange(1, 1, -1, 8).ok());
    EXPECT_FALSE(validatePositionIdRange(1, 1, 9, 8).ok());
}

TEST_F(EmbeddingIdRangeTest, testPositionEmbeddingRowLimitUsesShorterConfiguredTable) {
    EXPECT_EQ(positionEmbeddingRowLimit(8, 6), 6);
}

TEST_F(EmbeddingIdRangeTest, testPositionEmbeddingRowLimitKeepsShorterMaxSequenceLimit) {
    EXPECT_EQ(positionEmbeddingRowLimit(8, 16), 8);
}

TEST_F(EmbeddingIdRangeTest, testPositionEmbeddingRowLimitIgnoresUnconfiguredTable) {
    EXPECT_EQ(positionEmbeddingRowLimit(8, 0), 8);
}

TEST_F(EmbeddingIdRangeTest, testPositionIdBiasUsesStyleAndPadToken) {
    EXPECT_EQ(positionIdBias(0, 7), 0);
    EXPECT_EQ(positionIdBias(1, 7), 8);
}

}  // namespace rtp_llm
