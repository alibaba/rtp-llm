#include "rtp_llm/cpp/models/BertUqiAttention.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace rtp_llm {
namespace {

template<typename T>
std::vector<T> tensorValues(const torch::Tensor& tensor) {
    const auto* data = tensor.data_ptr<T>();
    return std::vector<T>(data, data + tensor.numel());
}

TEST(BertUqiAttentionTest, MasksOnlyProfileKeysForNonProfileQueries) {
    auto tokens =
        torch::tensor({101, 11, 102, 2, 90, 91, 102, 201, 201, 101, 12, 102, 101, 2, 77, 102, 202}, torch::kInt32);
    auto lengths = torch::tensor({9, 3, 5}, torch::kInt32);

    auto metadata = buildBertUqiInputs(tokens, lengths, 2, 102);

    auto mask = metadata.attention_mask;
    EXPECT_TRUE(mask.is_pinned());
    ASSERT_EQ(mask.numel(), 9 * 9 + 3 * 3 + 5 * 5);
    const auto*                          visible = mask.data_ptr<bool>();
    const std::vector<std::vector<bool>> profiles{{false, false, false, true, true, true, true, false, false},
                                                  {false, false, false},
                                                  {false, true, true, true, false}};
    int64_t                              offset = 0;
    for (const auto& profile : profiles) {
        for (size_t q = 0; q < profile.size(); ++q) {
            for (size_t k = 0; k < profile.size(); ++k) {
                EXPECT_EQ(visible[offset++], profile[q] || !profile[k]);
            }
        }
    }
}

TEST(BertUqiAttentionTest, KeepsOrdinaryBertBatchOnSinglePass) {
    auto metadata = buildBertUqiInputs(
        torch::tensor({101, 10, 102, 101, 11, 12, 102}, torch::kInt32), torch::tensor({3, 4}, torch::kInt32), 2, 102);

    EXPECT_EQ(metadata.attention_mask.numel(), 0);
}

TEST(BertUqiAttentionTest, PoolsOriginalRowsIncludingVisionAndMissingProfile) {
    auto metadata = buildBertUqiInputs(
        torch::tensor({101, 11, 102, 2, 90, 102, 201, 201, 101, 12, 102, 101, 2, 77, 102}, torch::kInt32),
        torch::tensor({8, 3, 4}, torch::kInt32),
        2,
        102);
    auto positions = metadata.pooling_positions;
    EXPECT_TRUE(positions.is_pinned());
    EXPECT_EQ(positions.sizes(), (torch::IntArrayRef{3, 2}));
    EXPECT_EQ(tensorValues<int64_t>(positions), (std::vector<int64_t>{0, 3, 8, 8, 11, 12}));
}

TEST(BertUqiAttentionTest, RejectsMalformedSegments) {
    EXPECT_THROW(
        buildBertUqiInputs(torch::tensor({101, 2, 8}, torch::kInt32), torch::tensor({3}, torch::kInt32), 2, 102),
        std::invalid_argument);
    EXPECT_THROW(buildBertUqiInputs(
                     torch::tensor({101, 2, 8, 2, 102}, torch::kInt32), torch::tensor({5}, torch::kInt32), 2, 102),
                 std::invalid_argument);
    EXPECT_THROW(
        buildBertUqiInputs(torch::tensor({101, 10, 102}, torch::kInt32), torch::tensor({2}, torch::kInt32), 2, 102),
        std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
