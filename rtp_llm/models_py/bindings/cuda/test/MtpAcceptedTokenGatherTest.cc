#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

namespace rtp_llm {
namespace {

TEST(MtpAcceptedTokenGatherTest, MixedIntegerTypesWidthsAndOffsetViews) {
    c10::InferenceMode inference;
    for (int batch_size : {1, 23, 92, 128, 131, 257}) {
        std::vector<torch::Tensor> tokens;
        std::vector<torch::Tensor> lengths;
        std::vector<int32_t>       expected;
        for (int i = 0; i < batch_size; ++i) {
            const int  width       = i % 7 + 1;
            const int  length      = i % width + 1;
            const auto token_type  = i % 2 ? torch::kInt64 : torch::kInt32;
            const auto length_type = i % 3 ? torch::kInt32 : torch::kInt64;
            auto storage = torch::arange(width + 5, torch::TensorOptions().dtype(token_type).device(torch::kCUDA));
            tokens.push_back(storage.narrow(0, 3, width).reshape({1, width}));
            auto length_storage = torch::tensor({-1, length, -1}, torch::TensorOptions().dtype(length_type)).cuda();
            lengths.push_back(length_storage.narrow(0, 1, 1));
            expected.push_back(3 + length - 1);
        }
        torch::Tensor output;
        ASSERT_TRUE(tryInvokeMtpAcceptedTokenGather(tokens, lengths, output, at::cuda::getCurrentCUDAStream()));
        EXPECT_EQ(output.scalar_type(), torch::kInt32);
        EXPECT_EQ(output.dim(), 1);
        EXPECT_EQ(output.size(0), batch_size);
        EXPECT_TRUE(torch::equal(output.cpu(), torch::tensor(expected, torch::kInt32))) << batch_size;
    }
}

TEST(MtpAcceptedTokenGatherTest, CurrentStreamAndUpdatedDeviceLengths) {
    c10::InferenceMode               inference;
    const auto                       stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard       guard(stream);
    auto                             tokens = torch::tensor({{11, 22, 33, 44}, {55, 66, 77, 88}}, torch::kInt32).cuda();
    auto                             lengths = torch::tensor({1, 4}, torch::kInt32).cuda();
    const std::vector<torch::Tensor> token_rows{tokens.narrow(0, 0, 1), tokens.narrow(0, 1, 1)};
    const std::vector<torch::Tensor> length_rows{lengths.narrow(0, 0, 1), lengths.narrow(0, 1, 1)};
    torch::Tensor                    first;
    ASSERT_TRUE(tryInvokeMtpAcceptedTokenGather(token_rows, length_rows, first, stream));
    lengths.fill_(2);
    tokens.add_(100);
    torch::Tensor second;
    ASSERT_TRUE(tryInvokeMtpAcceptedTokenGather(token_rows, length_rows, second, stream));
    // Inputs remain mutable between launches. Each read observes the preceding
    // writes on the caller's stream, without device/host synchronization.
    EXPECT_TRUE(torch::equal(first.cpu(), torch::tensor({11, 88}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(second.cpu(), torch::tensor({122, 166}, torch::kInt32)));
}

TEST(MtpAcceptedTokenGatherTest, UnsupportedRowsLeaveOutputUnchangedForFallback) {
    c10::InferenceMode inference;
    const auto         stream = at::cuda::getCurrentCUDAStream();
    auto               tokens = torch::arange(16, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto               length = torch::ones({1}, tokens.options());
    auto               output = torch::full({1}, 17, tokens.options());
    const auto*        original_output = output.data_ptr();
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({tokens.reshape({4, 4}).transpose(0, 1)}, {length}, output, stream));
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({tokens.to(torch::kFloat32)}, {length}, output, stream));
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({tokens}, {length.cpu()}, output, stream));
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({tokens}, {tokens.narrow(0, 0, 2)}, output, stream));
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({tokens.narrow(0, 0, 0)}, {length}, output, stream));
    EXPECT_FALSE(tryInvokeMtpAcceptedTokenGather({}, {}, output, stream));
    EXPECT_EQ(output.data_ptr(), original_output);
    EXPECT_EQ(output.item<int32_t>(), 17);
}

}  // namespace
}  // namespace rtp_llm
