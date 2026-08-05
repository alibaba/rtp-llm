#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/normal_engine/PrefillWarmupBatchSizing.h"

namespace rtp_llm {
namespace {

TEST(PrefillWarmupBatchSizingTest, UsesConfiguredTokenBudgetWithCeilingDivision) {
    auto result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                                    /*configured_max_batch_tokens=*/8192,
                                                    /*max_context_batch_size=*/8);
    EXPECT_EQ(result.max_batch_tokens, 8192u);
    EXPECT_EQ(result.num_sequences, 2u);

    result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                               /*configured_max_batch_tokens=*/8193,
                                               /*max_context_batch_size=*/8);
    EXPECT_EQ(result.num_sequences, 3u);
}

TEST(PrefillWarmupBatchSizingTest, UsesContextBatchFallbackAndAlwaysRunsOneSequence) {
    auto result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                                    /*configured_max_batch_tokens=*/0,
                                                    /*max_context_batch_size=*/8);
    EXPECT_EQ(result.max_batch_tokens, 32768u);
    EXPECT_EQ(result.num_sequences, 8u);

    result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                               /*configured_max_batch_tokens=*/0,
                                               /*max_context_batch_size=*/0);
    EXPECT_EQ(result.max_batch_tokens, 4096u);
    EXPECT_EQ(result.num_sequences, 1u);
}

TEST(PrefillWarmupBatchSizingTest, CapsConfiguredTokenBudgetByContextBatchSize) {
    const auto result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                                          /*configured_max_batch_tokens=*/32769,
                                                          /*max_context_batch_size=*/4);
    EXPECT_EQ(result.num_sequences, 4u);
    EXPECT_EQ(result.max_batch_tokens, 16384u);
}

// max_context_batch_size == 0 means "unset", and it means two different things to the
// two branches that read it (see the header comment). This pins the capping branch's
// reading -- no cap -- so nobody "unifies" it with the token-budget branch's
// treat-zero-as-one and silently truncates a configured budget to a single sequence.
TEST(PrefillWarmupBatchSizingTest, ZeroContextBatchSizeDoesNotCapConfiguredTokenBudget) {
    const auto result = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/32768,
                                                          /*configured_max_batch_tokens=*/65536,
                                                          /*max_context_batch_size=*/0);
    EXPECT_EQ(result.max_batch_tokens, 65536u);
    EXPECT_EQ(result.num_sequences, 2u);

    // Same reading with a budget that is not a whole multiple of max_seq_len: the
    // ceiling stands because there is no cap to apply.
    const auto rounded = calculatePrefillWarmupBatchSizing(/*max_seq_len=*/4096,
                                                           /*configured_max_batch_tokens=*/32769,
                                                           /*max_context_batch_size=*/0);
    EXPECT_EQ(rounded.max_batch_tokens, 32769u);
    EXPECT_EQ(rounded.num_sequences, 9u);
}

TEST(PrefillWarmupBatchSizingTest, RejectsInvalidOrOverflowingInputs) {
    EXPECT_THROW(calculatePrefillWarmupBatchSizing(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(calculatePrefillWarmupBatchSizing(2, 0, std::numeric_limits<size_t>::max()), std::overflow_error);
}

}  // namespace
}  // namespace rtp_llm
