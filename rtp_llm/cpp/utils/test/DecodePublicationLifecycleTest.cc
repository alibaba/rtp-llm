#include "rtp_llm/cpp/utils/DecodePublicationLifecycle.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(DecodePublicationLifecycleTest, MissingBoundaryDiffersFromComputedZero) {
    DecodePublicationLifecycle state;
    EXPECT_FALSE(state.prefillComputedEnd());
    EXPECT_TRUE(state.recordPrefillComputedEnd(0));
    EXPECT_EQ(state.prefillComputedEnd(), 0);
}

TEST(DecodePublicationLifecycleTest, RejectsNegativeAndConflictingEndpoints) {
    DecodePublicationLifecycle state;
    EXPECT_FALSE(state.recordPrefillComputedEnd(-1));
    EXPECT_FALSE(state.prefillComputedEnd());
    ASSERT_TRUE(state.recordPrefillComputedEnd(1536));
    EXPECT_TRUE(state.recordPrefillComputedEnd(1536));
    EXPECT_FALSE(state.recordPrefillComputedEnd(1537));
    EXPECT_FALSE(state.recordPrefillComputedEnd(1535));
    EXPECT_EQ(state.prefillComputedEnd(), 1536);
}

TEST(DecodePublicationLifecycleTest, LogicalResetDoesNotMutateRetainedCopy) {
    DecodePublicationLifecycle state;
    ASSERT_TRUE(state.recordPrefillComputedEnd(1536));
    auto retained = state;
    state.reset();
    EXPECT_FALSE(state.prefillComputedEnd());
    EXPECT_EQ(retained.prefillComputedEnd(), 1536);
    EXPECT_TRUE(state.recordPrefillComputedEnd(4096));
    EXPECT_EQ(retained.prefillComputedEnd(), 1536);
}

}  // namespace rtp_llm
