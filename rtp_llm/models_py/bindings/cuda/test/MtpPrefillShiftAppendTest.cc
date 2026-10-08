#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

namespace {

TEST(MtpPrefillShiftAppendTest, PackedRequestsUseEachRequestsLastSampleColumn) {
    const auto i32 = torch::TensorOptions(torch::kInt32);
    auto input = torch::tensor({100, 200, 201, 202}, i32).to(torch::kCUDA);
    auto lengths = torch::tensor({1, 3}, i32).to(torch::kCUDA);
    auto offsets = torch::tensor({1, 4}, i32).to(torch::kCUDA);
    auto sampled = torch::tensor({{5, 6, 7}, {8, 9, 10}}, i32).to(torch::kCUDA);
    auto output = torch::full_like(input, -1);

    rtp_llm::invokeMtpPrefillShiftAppend(input, lengths, offsets, sampled, output, 3, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_TRUE(torch::equal(output.cpu(), torch::tensor({7, 201, 202, 10}, i32)));
    EXPECT_TRUE(torch::equal(input.cpu(), torch::tensor({100, 200, 201, 202}, i32)));
}

TEST(MtpPrefillShiftAppendTest, DefaultPositionIdsShiftWithTokensAndKeepLastPosition) {
    const auto i32 = torch::TensorOptions(torch::kInt32);
    auto input = torch::tensor({100, 200, 201, 202}, i32).to(torch::kCUDA);
    auto positions = torch::tensor({9, 20, 21, 22}, i32).to(torch::kCUDA);
    auto lengths = torch::tensor({1, 3}, i32).to(torch::kCUDA);
    auto offsets = torch::tensor({1, 4}, i32).to(torch::kCUDA);
    auto sampled = torch::tensor({{5, 6, 7}, {8, 9, 10}}, i32).to(torch::kCUDA);
    auto output = torch::full_like(input, -1);
    auto next_positions = torch::full_like(positions, -1);

    rtp_llm::invokeMtpPrefillShiftAppend(input, lengths, offsets, sampled, output,
                                         positions, next_positions, 3, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_TRUE(torch::equal(output.cpu(), torch::tensor({7, 201, 202, 10}, i32)));
    EXPECT_TRUE(torch::equal(next_positions.cpu(), torch::tensor({9, 21, 22, 22}, i32)));
    EXPECT_TRUE(torch::equal(positions.cpu(), torch::tensor({9, 20, 21, 22}, i32)));
}

TEST(MtpPrefillShiftAppendTest, LongSingleRequestKeepsEveryInteriorToken) {
    const auto i32 = torch::TensorOptions(torch::kInt32);
    constexpr int64_t tokens = 65536;
    auto input = torch::arange(tokens, i32.device(torch::kCUDA));
    auto lengths = torch::tensor({tokens}, i32).to(torch::kCUDA);
    auto offsets = lengths.clone();
    auto sampled = torch::tensor({{3, 4, 5}}, i32).to(torch::kCUDA);
    auto output = torch::empty_like(input);

    rtp_llm::invokeMtpPrefillShiftAppend(input, lengths, offsets, sampled, output, 3, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_EQ(output[0].item<int32_t>(), 1);
    EXPECT_EQ(output[tokens / 2].item<int32_t>(), tokens / 2 + 1);
    EXPECT_EQ(output[tokens - 2].item<int32_t>(), tokens - 1);
    EXPECT_EQ(output[tokens - 1].item<int32_t>(), 5);
}

}  // namespace
