#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

TEST(NormalOutputDispatcherTest, RawLogprobsStayFiniteWhenProbabilitiesUnderflow) {
    auto logits = torch::tensor({{-500.0f, 0.0f, 500.0f}, {1.0f, 2.0f, 3.0f}}, torch::kBFloat16);
    auto before = logits.clone();
    auto ids = torch::tensor({{0}, {1}}, torch::kLong);
    auto actual = computeSelectedTokenRawLogprobs(logits, ids);
    auto expected = torch::log_softmax(logits.to(torch::kFloat32), -1).gather(1, ids);
    EXPECT_EQ(actual.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
    EXPECT_FLOAT_EQ(actual[0][0].item<float>(), -1000.0f);
    EXPECT_TRUE(torch::equal(logits, before));
}

TEST(NormalOutputDispatcherTest, RawLogprobsUseOriginalScoresBeforeSamplingTransforms) {
    auto raw = torch::tensor({{1.25f, -2.0f, 3.5f}, {-3.0f, 0.25f, 0.0f}}, torch::kFloat32);
    auto processed = raw.clone();
    processed.div_(2.0f);
    processed.select(1, 2).fill_(-1000.0f);
    auto ids = torch::tensor({{0}, {1}}, torch::kLong);
    auto actual = computeSelectedTokenRawLogprobs(raw, ids);
    auto expected = torch::log_softmax(raw, -1).gather(1, ids);
    auto transformed = torch::log_softmax(processed, -1).gather(1, ids);
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
    EXPECT_FALSE(torch::allclose(actual, transformed, 1e-6, 1e-6));
}

TEST(NormalOutputDispatcherTest, ProcessorErrorsUseSamplerInputCoordinatesAfterBeamExpansion) {
    SamplerOutput sampler_output;
    sampler_output.processor_errors = {
        std::nullopt,
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second stream processor failed"),
    };

    // The first stream may expand from one input row to multiple output beams.
    // Its input slice must not consume the second stream's processor error.
    auto first_stream_error = collectStreamSamplerError(
        sampler_output, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/0, /*cur_batch_size=*/1);
    EXPECT_FALSE(first_stream_error.has_value());

    auto second_stream_error = collectStreamSamplerError(
        sampler_output, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/1, /*cur_batch_size=*/1);
    ASSERT_TRUE(second_stream_error.has_value());
    EXPECT_EQ(second_stream_error->code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(second_stream_error->ToString().find("second stream processor failed"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
