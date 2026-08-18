#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_replay_contracts.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphReplayContractsTest, BuildsDynamicFullPrefillSentinelLayout) {
    std::vector<int32_t> input_lengths{24, 32, -1, -1, -1};
    std::vector<int32_t> cu_seqlens(6, -1);
    std::vector<int32_t> padding_offset(64, -1);

    ASSERT_TRUE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 2, 4, 56, 64));
    EXPECT_EQ(input_lengths, (std::vector<int32_t>{24, 32, 0, 0, 8}));
    EXPECT_EQ(cu_seqlens, (std::vector<int32_t>{0, 24, 56, 56, 56, 64}));
    EXPECT_TRUE(
        std::all_of(padding_offset.begin(), padding_offset.begin() + 24, [](int32_t value) { return value == 0; }));
    EXPECT_TRUE(std::all_of(
        padding_offset.begin() + 24, padding_offset.begin() + 56, [](int32_t value) { return value == 40; }));
    EXPECT_TRUE(
        std::all_of(padding_offset.begin() + 56, padding_offset.end(), [](int32_t value) { return value == 200; }));
}

TEST(CudaGraphReplayContractsTest, AllowsZeroLengthSentinelAtExactTokenCapacity) {
    std::vector<int32_t> input_lengths{16, 48, -1};
    std::vector<int32_t> cu_seqlens(4, -1);
    std::vector<int32_t> padding_offset(64, -1);

    ASSERT_TRUE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 2, 2, 64, 64));
    EXPECT_EQ(input_lengths, (std::vector<int32_t>{16, 48, 0}));
    EXPECT_EQ(cu_seqlens, (std::vector<int32_t>{0, 16, 64, 64}));
}

TEST(CudaGraphReplayContractsTest, RejectsInvalidFullPrefillMetadata) {
    std::vector<int32_t> input_lengths{24, 32, -1};
    std::vector<int32_t> cu_seqlens(4, -1);
    std::vector<int32_t> padding_offset(64, -1);

    EXPECT_FALSE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 0, 2, 56, 64));
    EXPECT_FALSE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 3, 2, 56, 64));
    EXPECT_FALSE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 2, 2, 55, 64));

    input_lengths = {24, 0, -1};
    EXPECT_FALSE(
        prepareFullPrefillReplayMetadata(input_lengths.data(), cu_seqlens.data(), padding_offset.data(), 2, 2, 24, 64));
}

TEST(CudaGraphReplayContractsTest, AcceptsExactComboSourceAndDestinationSizes) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto src     = torch::zeros({6}, options);
    const auto dst     = torch::zeros({6}, options);
    size_t     copy_numel;

    EXPECT_TRUE(validateComboPositionIdsForReplay(3, 2, src, dst, copy_numel));
    EXPECT_EQ(copy_numel, 6);
}

TEST(CudaGraphReplayContractsTest, AcceptsContiguousMultidimensionalComboBuffers) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto src     = torch::zeros({2, 3}, options);
    const auto dst     = torch::zeros({2, 3}, options);
    size_t     copy_numel;

    EXPECT_TRUE(validateComboPositionIdsForReplay(3, 2, src, dst, copy_numel));
    EXPECT_EQ(copy_numel, 6);
}

TEST(CudaGraphReplayContractsTest, AllowsCrossDeviceComboIdsAsLegacyBehavior) {
    if (torch::cuda::device_count() < 2) {
        GTEST_SKIP() << "Multiple CUDA devices are required to verify the combo replay contract";
    }
    const auto src =
        torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, 0)));
    const auto dst =
        torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, 1)));
    size_t copy_numel;

    EXPECT_TRUE(validateComboPositionIdsForReplay(3, 2, src, dst, copy_numel));
    EXPECT_EQ(copy_numel, 6);
}

TEST(CudaGraphReplayContractsTest, RejectsTooSmallComboSourceOrDestination) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    size_t     copy_numel;

    EXPECT_FALSE(
        validateComboPositionIdsForReplay(3, 2, torch::zeros({3}, options), torch::zeros({6}, options), copy_numel));
    EXPECT_EQ(copy_numel, 0);
    EXPECT_FALSE(
        validateComboPositionIdsForReplay(3, 2, torch::zeros({6}, options), torch::zeros({3}, options), copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsTest, RejectsInvalidComboTensorContracts) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto valid       = torch::zeros({6}, int_options);
    size_t     copy_numel;

    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::Tensor(), valid, copy_numel));
    EXPECT_EQ(copy_numel, 0);
    const auto int64_cuda = torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, int64_cuda, valid, copy_numel));
    EXPECT_EQ(copy_numel, 0);
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::zeros({2, 3}, int_options).t(), valid, copy_numel));
    EXPECT_EQ(copy_numel, 0);
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, torch::zeros({5}, int_options), valid, copy_numel));
    EXPECT_EQ(copy_numel, 0);
    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 0, valid, valid, copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsTest, RejectsCpuComboTensorsForD2DCopy) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    const auto cpu     = torch::zeros({6}, options);
    size_t     copy_numel;

    EXPECT_FALSE(validateComboPositionIdsForReplay(3, 2, cpu, cpu, copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsTest, DisabledComboFactorNeedsNoPositionIds) {
    size_t copy_numel = 1;
    EXPECT_TRUE(validateComboPositionIdsForReplay(0, 0, torch::Tensor(), torch::Tensor(), copy_numel));
    EXPECT_EQ(copy_numel, 0);
}

TEST(CudaGraphReplayContractsTest, ValidatesReplayIdBufferRequirements) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto     options         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto     valid           = torch::zeros({4}, options);
    constexpr auto one_dimensional = ReplayIdDimRequirement::kOneDimensional;
    constexpr auto any_dim         = ReplayIdDimRequirement::kAny;
    constexpr auto same_device     = ReplayIdDeviceRequirement::kSameDevice;

    EXPECT_TRUE(validateReplayIdBufferForCopy(valid, valid, 4, one_dimensional, same_device));
    EXPECT_FALSE(validateReplayIdBufferForCopy(valid, valid, 0, one_dimensional, same_device));
    EXPECT_FALSE(validateReplayIdBufferForCopy(torch::Tensor(), valid, 4, one_dimensional, same_device));
    EXPECT_FALSE(validateReplayIdBufferForCopy(torch::zeros({2, 2}, options), valid, 4, one_dimensional, same_device));
    EXPECT_FALSE(validateReplayIdBufferForCopy(torch::zeros({3}, options), valid, 4, one_dimensional, same_device));
    EXPECT_TRUE(validateReplayIdBufferForCopy(torch::zeros({2, 2}, options), valid, 4, any_dim, same_device));
}

TEST(CudaGraphReplayContractsTest, RejectsCrossDeviceReplayIdsWhenSameDeviceIsRequired) {
    if (torch::cuda::device_count() < 2) {
        GTEST_SKIP() << "At least two CUDA devices are required for the cross-device replay contract";
    }
    const auto     options         = torch::TensorOptions().dtype(torch::kInt32);
    const auto     source          = torch::zeros({4}, options.device(torch::Device(torch::kCUDA, 0)));
    const auto     destination     = torch::zeros({4}, options.device(torch::Device(torch::kCUDA, 1)));
    constexpr auto one_dimensional = ReplayIdDimRequirement::kOneDimensional;
    constexpr auto same_device     = ReplayIdDeviceRequirement::kSameDevice;
    constexpr auto any_device      = ReplayIdDeviceRequirement::kAny;

    EXPECT_FALSE(validateReplayIdBufferForCopy(source, destination, 4, one_dimensional, same_device));
    EXPECT_TRUE(validateReplayIdBufferForCopy(source, destination, 4, one_dimensional, any_device));
}

TEST(CudaGraphReplayContractsTest, ValidatesReplaySourceAndDestinationSymmetrically) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto     options         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto     valid           = torch::zeros({4}, options);
    const auto     empty           = torch::empty({0}, options);
    const auto     cpu             = torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt32));
    const auto     int64_cuda      = torch::zeros({4}, options.dtype(torch::kInt64));
    const auto     non_contiguous  = torch::zeros({2, 4}, options).t();
    const auto     two_dimensional = torch::zeros({2, 2}, options);
    constexpr auto one_dimensional = ReplayIdDimRequirement::kOneDimensional;
    constexpr auto same_device     = ReplayIdDeviceRequirement::kSameDevice;

    EXPECT_TRUE(hasReplayIdBufferContract(valid, valid, one_dimensional, same_device));
    const std::vector<std::pair<const char*, torch::Tensor>> invalid_cases = {{"undefined", torch::Tensor()},
                                                                              {"empty", empty},
                                                                              {"cpu", cpu},
                                                                              {"int64", int64_cuda},
                                                                              {"non_contiguous", non_contiguous},
                                                                              {"two_dimensional", two_dimensional}};
    for (const auto& [name, invalid] : invalid_cases) {
        SCOPED_TRACE(name);
        EXPECT_FALSE(hasReplayIdBufferContract(invalid, valid, one_dimensional, same_device));
        EXPECT_FALSE(hasReplayIdBufferContract(valid, invalid, one_dimensional, same_device));
    }
}

TEST(CudaGraphReplayContractsTest, DetectsCompleteBertTablePair) {
    const auto table = torch::zeros({2, 4});

    EXPECT_TRUE(hasBothBertEmbeddingTables(table, table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(table, torch::Tensor()));
    EXPECT_FALSE(hasBothBertEmbeddingTables(torch::Tensor(), table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(torch::empty({0}), table));
    EXPECT_FALSE(hasBothBertEmbeddingTables(table, torch::empty({0})));
}

TEST(CudaGraphReplayContractsTest, ValidatesBertIdBuffersAsAPair) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for the D2D replay contract";
    }
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto valid   = torch::zeros({4}, options);

    EXPECT_TRUE(validateBertReplayIdBuffersForCopy(valid, valid, valid, valid, 4));
    EXPECT_FALSE(validateBertReplayIdBuffersForCopy(valid, valid, torch::Tensor(), valid, 4));
    EXPECT_FALSE(validateBertReplayIdBuffersForCopy(valid, valid, valid, torch::zeros({3}, options), 4));
}

}  // namespace
}  // namespace rtp_llm
