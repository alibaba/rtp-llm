#include "rtp_llm/cpp/cuda_graph/decode_execution.h"

#include <stdexcept>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

const std::vector<int> kBuckets{1, 2, 4, 8};

TEST(DecodeExecutionTest, AbsentDecisionPreservesLegacyBucketSelection) {
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, std::nullopt, 1, 3, std::nullopt), 4);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, std::nullopt, 1, 8, std::nullopt), 8);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, std::nullopt, 1, 9, std::nullopt), std::nullopt);
}

TEST(DecodeExecutionTest, ExplicitGraphUsesTheAgreedBucketOnly) {
    const auto            phase = DecodeExecutionPhase::TARGET_VERIFY;
    const DecodeExecution execution{DecodeExecutionMode::GRAPH, phase, 8, 4};
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 1, execution), 8);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 8, execution), 8);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 9, execution), std::nullopt);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 0, execution), std::nullopt);

    const DecodeExecution missing_bucket{DecodeExecutionMode::GRAPH, phase, 3, 4};
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 1, missing_bucket), std::nullopt);
}

TEST(DecodeExecutionTest, ExplicitGraphRequiresMatchingPhaseAndWidth) {
    const auto            phase = DecodeExecutionPhase::DSPARK_PROPOSAL;
    const DecodeExecution execution{DecodeExecutionMode::GRAPH, phase, 4, 3};
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 3, 1, execution), 4);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, DecodeExecutionPhase::DSPARK_COMMIT, 3, 1, execution), std::nullopt);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 4, 1, execution), std::nullopt);
}

TEST(DecodeExecutionTest, ExplicitModeAndRunnerOptInAreRequiredTogether) {
    const auto phase = DecodeExecutionPhase::TARGET;
    for (const auto mode : {DecodeExecutionMode::EAGER, DecodeExecutionMode::FIXED}) {
        const DecodeExecution execution{mode, phase, 1, 1};
        EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 1, 1, execution), std::nullopt);
    }
    const DecodeExecution execution{DecodeExecutionMode::GRAPH, phase, 1, 1};
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, std::nullopt, 1, 1, execution), std::nullopt);
    EXPECT_EQ(selectDecodeGraphBucket(kBuckets, phase, 1, 1, std::nullopt), std::nullopt);
}

TEST(AgreedDecodeGraphPreparationTest, PublishesOnlyAfterPreparationAndConsumesOnce) {
    const auto                   phase = DecodeExecutionPhase::MTP_COMMIT;
    const DecodeExecution        execution{DecodeExecutionMode::GRAPH, phase, 4, 3};
    AgreedDecodeGraphPreparation preparation;
    EXPECT_TRUE(preparation.prepare([&] {
        // The fixed request-lane prefill graph is keyed by tokens, not requests.
        EXPECT_FALSE(preparation.consume(phase, execution, 3, 4, 12));
        return 12;
    }));
    EXPECT_TRUE(preparation.consume(phase, execution, 3, 4, 12));
    EXPECT_FALSE(preparation.consume(phase, execution, 3, 4, 12));
}

TEST(AgreedDecodeGraphPreparationTest, RejectsWrongDecisionAndDiscardsItsKey) {
    const auto                   phase = DecodeExecutionPhase::MTP_COMMIT;
    const DecodeExecution        execution{DecodeExecutionMode::GRAPH, phase, 4, 3};
    AgreedDecodeGraphPreparation preparation;
    for (const DecodeExecution wrong_execution :
         {DecodeExecution{DecodeExecutionMode::EAGER, phase, 4, 3},
          DecodeExecution{DecodeExecutionMode::GRAPH, DecodeExecutionPhase::DSPARK_COMMIT, 4, 3},
          DecodeExecution{DecodeExecutionMode::GRAPH, phase, 2, 3},
          DecodeExecution{DecodeExecutionMode::GRAPH, phase, 4, 1}}) {
        ASSERT_TRUE(preparation.prepare([] { return 12; }));
        EXPECT_FALSE(preparation.consume(phase, wrong_execution, 3, 4, 12));
        EXPECT_FALSE(preparation.consume(phase, execution, 3, 4, 12));
    }
    ASSERT_TRUE(preparation.prepare([] { return 12; }));
    EXPECT_FALSE(preparation.consume(phase, execution, 3, 4, 4));
    EXPECT_FALSE(preparation.consume(phase, execution, 3, 4, 12));
}

TEST(AgreedDecodeGraphPreparationTest, RejectionFailureAndFallbackInvalidatePreparedGraph) {
    const auto                   phase = DecodeExecutionPhase::TARGET;
    const DecodeExecution        execution{DecodeExecutionMode::GRAPH, phase, 4, 1};
    AgreedDecodeGraphPreparation preparation;

    ASSERT_TRUE(preparation.prepare([] { return 4; }));
    EXPECT_FALSE(preparation.prepare([] { return 0; }));
    EXPECT_FALSE(preparation.consume(phase, execution, 1, 4, 4));

    ASSERT_TRUE(preparation.prepare([] { return 4; }));
    EXPECT_THROW(preparation.prepare([]() -> int { throw std::runtime_error("device staging failed"); }),
                 std::runtime_error);
    EXPECT_FALSE(preparation.consume(phase, execution, 1, 4, 4));

    ASSERT_TRUE(preparation.prepare([] { return 4; }));
    preparation.discard();
    EXPECT_FALSE(preparation.consume(phase, execution, 1, 4, 4));
}

}  // namespace
}  // namespace rtp_llm
