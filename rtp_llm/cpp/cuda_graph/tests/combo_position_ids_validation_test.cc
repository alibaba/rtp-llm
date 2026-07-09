#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/combo_position_ids_validation.h"

namespace rtp_llm {
namespace {

TEST(ComboPositionIdsValidationTest, AcceptsExactSourceAndDestinationSizes) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    const auto src     = torch::zeros({6}, options);
    const auto dst     = torch::zeros({6}, options);
    size_t     copy_numel;

    EXPECT_TRUE(validateComboPositionIdsForReplay(3, 2, src, dst, copy_numel));
    EXPECT_EQ(copy_numel, 6);
}

TEST(ComboPositionIdsValidationTest, RejectsTooSmallSourceOrDestination) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    size_t     copy_numel;

    EXPECT_FALSE(
        validateComboPositionIdsForReplay(3, 2, torch::zeros({3}, options), torch::zeros({6}, options), copy_numel));
    EXPECT_EQ(copy_numel, 6);
    EXPECT_FALSE(
        validateComboPositionIdsForReplay(3, 2, torch::zeros({6}, options), torch::zeros({3}, options), copy_numel));
    EXPECT_EQ(copy_numel, 6);
}

TEST(ComboPositionIdsValidationTest, RejectsInvalidTensorContracts) {
    const auto int_options = torch::TensorOptions().dtype(torch::kInt32);
    const auto valid       = torch::zeros({6}, int_options);
    size_t     copy_numel;

    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::Tensor(), valid, copy_numel));
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::zeros({6}, torch::kInt64), valid, copy_numel));
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::zeros({2, 3}, int_options).t(), valid, copy_numel));
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::zeros({5}, int_options), valid, copy_numel));
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 0, valid, valid, copy_numel));
}

TEST(ComboPositionIdsValidationTest, DisabledFactorNeedsNoPositionIds) {
    size_t copy_numel = 1;
    EXPECT_TRUE(validateComboPositionIdsForReplay(0, 0, torch::Tensor(), torch::Tensor(), copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

}  // namespace
}  // namespace rtp_llm
