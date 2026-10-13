#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/NanDetection.h"
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

TEST(NormalOutputDispatcherTest, NanMetadataSelectsEarliestSiteOnlyInsideStreamRows) {
    auto records = torch::tensor({{0, 1, 9, 1}, {2, 1, 7, 2}, {1, 2, 4, 3}, {0, 1, 2, 1}}, torch::kInt64);
    auto first   = firstNanInfo(records, 1, 2, 17);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->layer_id, 1);
    EXPECT_EQ(first->stage, 2);
    EXPECT_EQ(first->token_index, 18);
    EXPECT_EQ(first->element_index, 4);
    EXPECT_EQ(first->kind, 3);
}

TEST(NormalOutputDispatcherTest, RejectedVerifyRowsCannotWinFirstSiteSelection) {
    auto records = torch::tensor({{4, 2, 7, 1}, {0, 1, 0, 1}, {0, 0, 0, 0}}, torch::kInt64);
    auto first   = firstNanInfo(records, 0, 1, 20);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->layer_id, 4);
    EXPECT_EQ(first->token_index, 20);
    EXPECT_FALSE(firstNanInfo(records, 2, 1, 30).has_value());
    EXPECT_FALSE(firstNanInfo(records, 0, 0, 20).has_value());
}

TEST(NormalOutputDispatcherTest, EarlierForwardWinsEvenWhenLaterForwardHasLowerLayer) {
    auto records        = torch::tensor({{8, 4, 5, 2}, {0, 1, 3, 1}}, torch::kInt64);
    auto proposal_first = firstNanInfo(records, 0, 2, 10, 0, 4, true);
    ASSERT_TRUE(proposal_first.has_value());
    EXPECT_EQ(proposal_first->layer_id, 8);
    auto first = firstNanInfo(records, 0, 1, 10);
    keepFirstNanInfo(first, firstNanInfo(records, 1, 1, 11));
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->layer_id, 8);
}

TEST(NormalOutputDispatcherTest, DraftTailLogitsStayDeferredUntilProposalAcceptance) {
    auto records = torch::tensor({{8, 4, 5, 2}}, torch::kInt64);
    EXPECT_FALSE(firstNanInfo(records, 0, 1, 10, 0, 3).has_value());
    auto accepted = firstNanInfo(records, 0, 1, 11);
    ASSERT_TRUE(accepted.has_value());
    EXPECT_EQ(accepted->stage, 4);
    EXPECT_EQ(accepted->token_index, 11);
}

}  // namespace
}  // namespace rtp_llm
