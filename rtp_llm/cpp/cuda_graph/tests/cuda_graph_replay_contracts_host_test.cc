#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_replay_contracts.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphReplayContractsHostTest, EvaluatesDeviceRequirementWithoutAllocatingCudaTensors) {
    const c10::Device cuda_0(torch::kCUDA, 0);
    const c10::Device cuda_1(torch::kCUDA, 1);

    EXPECT_TRUE(satisfiesReplayIdDeviceRequirement(ReplayIdDeviceRequirement::kSameDevice, cuda_0, cuda_0));
    EXPECT_FALSE(satisfiesReplayIdDeviceRequirement(ReplayIdDeviceRequirement::kSameDevice, cuda_0, cuda_1));
    EXPECT_TRUE(satisfiesReplayIdDeviceRequirement(ReplayIdDeviceRequirement::kAny, cuda_0, cuda_1));
}

TEST(CudaGraphReplayContractsHostTest, AppliesReplayIdPolicySelectionWithoutAllocatingCudaTensors) {
    const c10::Device cuda_0(torch::kCUDA, 0);
    const c10::Device cuda_1(torch::kCUDA, 1);

    EXPECT_FALSE(satisfiesReplayIdDimRequirement(kBertReplayIdDimRequirement, 2, 2));
    EXPECT_TRUE(satisfiesReplayIdDimRequirement(kComboReplayIdDimRequirement, 2, 2));
    EXPECT_FALSE(satisfiesReplayIdDeviceRequirement(kBertReplayIdDeviceRequirement, cuda_0, cuda_1));
    EXPECT_TRUE(satisfiesReplayIdDeviceRequirement(kComboReplayIdDeviceRequirement, cuda_0, cuda_1));
}

TEST(CudaGraphReplayContractsHostTest, DisabledComboFactorNeedsNoPositionIds) {
    size_t copy_numel = 1;
    EXPECT_TRUE(validateComboPositionIdsForReplay(0, 0, torch::Tensor(), torch::Tensor(), copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsHostTest, RejectsCpuComboTensorsForD2DCopy) {
    const auto cpu = torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt32));
    size_t     copy_numel;

    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, cpu, cpu, copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsHostTest, DetectsCompleteBertTablePair) {
    const auto table = torch::zeros({2, 4});

    EXPECT_TRUE(hasBothBertEmbeddingTables(table, table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(table, torch::Tensor()));
    EXPECT_FALSE(hasBothBertEmbeddingTables(torch::Tensor(), table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(torch::empty({0}), table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(table, torch::empty({0})));
}

TEST(CudaGraphReplayContractsHostTest, CapturesBertEmbeddingInputsOnlyForPrefill) {
    const auto table = torch::zeros({2, 4});

    EXPECT_TRUE(shouldCaptureBertEmbeddingInputs(true, table, table));
    EXPECT_FALSE(shouldCaptureBertEmbeddingInputs(false, table, table));
    EXPECT_FALSE(shouldCaptureBertEmbeddingInputs(true, table, torch::Tensor()));
    EXPECT_FALSE(shouldCaptureBertEmbeddingInputs(false, table, torch::Tensor()));
}

TEST(CudaGraphReplayContractsHostTest, DetectsEveryRequestOwnedMultimodalSignal) {
    EXPECT_FALSE(hasRequestOwnedMultimodalSignals({}));
    EXPECT_TRUE(hasRequestOwnedMultimodalSignals({.multimodal_features = true}));
    EXPECT_TRUE(hasRequestOwnedMultimodalSignals({.multimodal_locs = true}));
    EXPECT_TRUE(hasRequestOwnedMultimodalSignals({.multimodal_extra = true}));
    EXPECT_TRUE(hasRequestOwnedMultimodalSignals({.text_tokens_mask = true}));
    EXPECT_TRUE(hasRequestOwnedMultimodalSignals(
        {.multimodal_features = true, .multimodal_locs = true, .multimodal_extra = true, .text_tokens_mask = true}));
}

}  // namespace
}  // namespace rtp_llm
