#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

#if USING_CUDA && !USE_PPU
TEST(NormalOutputDispatcherTest, BatchInvariantPromptScoresIgnoreOtherRows) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "NVIDIA GPU required";
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto logits  = torch::randn({33, 65539}, options);
    auto tokens  = torch::arange(34, torch::kInt32);
    auto batch   = makePromptLogitsOutput(logits, tokens, 8, 0, -1, true, true);
    auto row     = makePromptLogitsOutput(logits.narrow(0, 17, 1), tokens.narrow(0, 17, 2), 8, 0, -1, true, true);
    ASSERT_TRUE(batch.has_value());
    ASSERT_TRUE(row.has_value());
    EXPECT_TRUE(torch::equal(batch->target_logprobs.narrow(0, 17, 1), row->target_logprobs));
    EXPECT_TRUE(torch::equal(batch->topk_logprobs.narrow(0, 17, 1), row->topk_logprobs));
}

TEST(NormalOutputDispatcherTest, BatchInvariantPromptScoresHandleLargeNegativeLogits) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "NVIDIA GPU required";
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto logits  = torch::full({1, 2}, -1e30, options);
    auto tokens  = torch::zeros({2}, torch::kInt32);
    auto output  = makePromptLogitsOutput(logits, tokens, 1, 0, -1, true, true);
    ASSERT_TRUE(output.has_value());
    EXPECT_NEAR(output->target_logprobs.item<float>(), -0.69314718f, 1e-6f);
}
#endif

TEST(NormalOutputDispatcherTest, ProcessorErrorsUseSamplerInputCoordinatesAfterBeamExpansion) {
    SamplerOutput sampler_output;
    sampler_output.processor_errors = {
        std::nullopt,
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second stream processor failed"),
    };

    // The first stream may expand from one input row to multiple output beams.
    // Its input slice must not consume the second stream's processor error.
    auto first_stream_error = collectStreamSamplerError(
        sampler_output.processor_errors, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/0, /*cur_batch_size=*/1);
    EXPECT_FALSE(first_stream_error.has_value());

    auto second_stream_error = collectStreamSamplerError(
        sampler_output.processor_errors, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/1, /*cur_batch_size=*/1);
    ASSERT_TRUE(second_stream_error.has_value());
    EXPECT_EQ(second_stream_error->code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(second_stream_error->ToString().find("second stream processor failed"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
