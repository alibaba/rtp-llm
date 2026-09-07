#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/combo_position_ids_validation.h"

namespace rtp_llm {
namespace {

TEST(ComboPositionIdsValidationCompatibilityTest, PublicHeaderForwardsReplayContract) {
    size_t copy_numel = 1;
    EXPECT_TRUE(validateComboPositionIdsForReplay(0, 0, torch::Tensor(), torch::Tensor(), copy_numel));
    EXPECT_EQ(copy_numel, 0);

    const auto cpu = torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt32));
    copy_numel     = 1;
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, cpu, cpu, copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

}  // namespace
}  // namespace rtp_llm
