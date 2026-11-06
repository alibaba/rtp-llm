#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/PromptScoringRows.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

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

TEST(NormalOutputDispatcherTest, PromptScoringRowRangeMatchesFlagAndClampRules) {
    struct Case {
        int  token_size;
        bool loss;
        bool prompt_logits;
        int  start;
        int  end;
        int  expect_a;
        int  expect_b;
    };
    const Case cases[] = {
        // Neither consumer -> always empty.
        {10, false, false, -1, -1, 0, 0},
        {1, false, false, -1, -1, 0, 0},
        // Prompt window clamps into [0, ts]; defaults cover the whole prompt.
        {10, false, true, -1, -1, 0, 10},
        {10, false, true, 2, 5, 2, 5},
        {10, false, true, 5, -1, 5, 10},
        {10, false, true, -5, 99, 0, 10},
        {1, false, true, -1, -1, 0, 1},
        {0, false, true, -1, -1, 0, 0},
        // Degenerate windows produce empty ranges (start >= end after clamping).
        {10, false, true, 8, 2, 8, 8},
        {10, false, true, 20, 25, 10, 10},
        {10, false, true, 4, 4, 4, 4},
        // Loss alone covers [0, ts-1): empty for ts <= 1.
        {10, true, false, -1, -1, 0, 9},
        {1, true, false, -1, -1, 0, 0},
        // Loss + prompt unions stay contiguous from 0; the end is the max of both ends.
        {10, true, true, 3, 7, 0, 9},
        {10, true, true, -1, -1, 0, 10},
        {10, true, true, 0, 12, 0, 10},
        {1, true, true, -1, -1, 0, 1},
        // An empty window past the prompt end still contributes its clamped end to the union.
        {10, true, true, 20, 25, 0, 10},
    };
    for (const auto& c : cases) {
        const auto [a, b] = promptScoringRowRange(c.token_size, c.loss, c.prompt_logits, c.start, c.end);
        EXPECT_EQ(a, c.expect_a) << "ts=" << c.token_size << " loss=" << c.loss << " prompt=" << c.prompt_logits << " ["
                                 << c.start << "," << c.end << ")";
        EXPECT_EQ(b, c.expect_b) << "ts=" << c.token_size << " loss=" << c.loss << " prompt=" << c.prompt_logits << " ["
                                 << c.start << "," << c.end << ")";
    }
}

}  // namespace
}  // namespace rtp_llm
