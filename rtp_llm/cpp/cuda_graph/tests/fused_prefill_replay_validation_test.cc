#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cuda_graph/fused_prefill_replay_validation.h"

namespace rtp_llm {
namespace {

torch::Tensor lengths(std::initializer_list<int32_t> values) {
    return torch::tensor(std::vector<int32_t>(values), torch::TensorOptions().dtype(torch::kInt32));
}

TEST(FusedPrefillReplayValidationTest, AcceptsShorterLengthsWithStablePrefixPresence) {
    const auto result =
        validateFusedPrefillReplayLengths(lengths({12, 4}), lengths({16, 16}), lengths({8, 8}), lengths({8, 16}));

    EXPECT_TRUE(result.compatible());
    EXPECT_EQ(result.captured_max_input_length, 12);
    EXPECT_EQ(result.replay_max_input_length, 8);
    EXPECT_EQ(result.captured_max_prefix_length, 16);
    EXPECT_EQ(result.replay_max_prefix_length, 16);
}

TEST(FusedPrefillReplayValidationTest, AcceptsPerRequestRedistributionWithinCapturedMaxima) {
    const auto result =
        validateFusedPrefillReplayLengths(lengths({12, 4}), lengths({16, 8}), lengths({4, 12}), lengths({8, 16}));

    EXPECT_TRUE(result.compatible());
    EXPECT_EQ(result.replay_max_input_length, 12);
    EXPECT_EQ(result.replay_max_prefix_length, 16);
}

TEST(FusedPrefillReplayValidationTest, EmptyPrefixVectorsAreEquivalentToNoPrefix) {
    const auto empty  = lengths({});
    const auto result = validateFusedPrefillReplayLengths(lengths({8, 8}), empty, lengths({4, 8}), empty);

    EXPECT_TRUE(result.compatible());
}

TEST(FusedPrefillReplayValidationTest, RejectsPerRequestInputLengthAboveCapture) {
    const auto result =
        validateFusedPrefillReplayLengths(lengths({8, 8}), lengths({8, 8}), lengths({15, 1}), lengths({8, 8}));

    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInputLengthExceedsCapture);
}

TEST(FusedPrefillReplayValidationTest, RejectsPrefixPresenceChangesInEitherDirection) {
    auto result = validateFusedPrefillReplayLengths(lengths({8, 8}), lengths({8, 8}), lengths({8, 8}), lengths({0, 0}));
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kPrefixPresenceChanged);

    result = validateFusedPrefillReplayLengths(lengths({8, 8}), lengths({0, 0}), lengths({8, 8}), lengths({8, 1}));
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kPrefixPresenceChanged);
}

TEST(FusedPrefillReplayValidationTest, RejectsPrefixLengthAboveCapture) {
    const auto result =
        validateFusedPrefillReplayLengths(lengths({8, 8}), lengths({8, 8}), lengths({8, 8}), lengths({16, 1}));

    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kPrefixLengthExceedsCapture);
}

TEST(FusedPrefillReplayValidationTest, RejectsMalformedOrNegativeLengths) {
    const auto valid  = lengths({8, 8});
    auto       result = validateFusedPrefillReplayLengths(torch::Tensor(), valid, valid, valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths);

    result = validateFusedPrefillReplayLengths(valid, valid, lengths({}), valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidReplayInputLengths);

    result = validateFusedPrefillReplayLengths(valid, valid, valid, lengths({8, -1}));
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidReplayPrefixLengths);

    result = validateFusedPrefillReplayLengths(valid, torch::Tensor(), valid, valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedPrefixLengths);

    result = validateFusedPrefillReplayLengths(valid.to(torch::kInt64), valid, valid, valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths);

    result = validateFusedPrefillReplayLengths(valid.reshape({1, 2}), valid, valid, valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths);

    const auto non_contiguous = torch::arange(4, torch::TensorOptions().dtype(torch::kInt32)).slice(0, 0, 4, 2);
    ASSERT_FALSE(non_contiguous.is_contiguous());
    result = validateFusedPrefillReplayLengths(non_contiguous, valid, valid, valid);
    EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths);

    if (torch::cuda::is_available()) {
        result = validateFusedPrefillReplayLengths(valid.cuda(), valid, valid, valid);
        EXPECT_EQ(result.status, FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths);
    }
}

TEST(FusedPrefillReplayValidationTest, EveryStatusHasAnObservableName) {
    const std::vector<FusedPrefillReplayValidationStatus> statuses = {
        FusedPrefillReplayValidationStatus::kCompatible,
        FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths,
        FusedPrefillReplayValidationStatus::kInvalidReplayInputLengths,
        FusedPrefillReplayValidationStatus::kInvalidCapturedPrefixLengths,
        FusedPrefillReplayValidationStatus::kInvalidReplayPrefixLengths,
        FusedPrefillReplayValidationStatus::kInputLengthExceedsCapture,
        FusedPrefillReplayValidationStatus::kPrefixLengthExceedsCapture,
        FusedPrefillReplayValidationStatus::kPrefixPresenceChanged,
    };
    for (const auto status : statuses) {
        EXPECT_STRNE(fusedPrefillReplayValidationStatusName(status), "unknown");
    }
}

}  // namespace
}  // namespace rtp_llm
