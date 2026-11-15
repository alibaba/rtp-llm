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

TEST(DecodePublicationLifecycleTest, ResourceSubmissionFactPreventsDuplicateAdmission) {
    DecodePublicationLifecycle state;
    bool submitted = false;
    int calls = 0;
    using A = DecodePublicationLifecycle::Admission;
    EXPECT_EQ(state.tryAdmit(submitted, [&] { ++calls; return true; }), A::ADMITTED);
    EXPECT_TRUE(submitted);
    EXPECT_EQ(state.tryAdmit(submitted, [&] { ++calls; return true; }), A::ALREADY_ADMITTED);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(state.admissionAttempts(), 1u);
}

TEST(DecodePublicationLifecycleTest, AcceptedResourceRemainsSubmittedIfPostAdmissionWorkThrows) {
    DecodePublicationLifecycle state;
    bool submitted = false;
    using A = DecodePublicationLifecycle::Admission;
    EXPECT_THROW(state.tryAdmit(submitted, [&]() -> bool {
        submitted = true; // The task was accepted before subsequent bookkeeping threw.
        throw 1;
    }), int);
    EXPECT_TRUE(submitted);
    EXPECT_EQ(state.tryAdmit(submitted, [] { return true; }), A::ALREADY_ADMITTED);
    EXPECT_EQ(state.admissionAttempts(), 1u);
}

TEST(DecodePublicationLifecycleTest, MayAttemptQueryNeverSpendsOrRestartsAdmission) {
    DecodePublicationLifecycle state;
    bool submitted = false;
    using A = DecodePublicationLifecycle::Admission;
    const auto now = DecodePublicationLifecycle::Clock::now();
    EXPECT_TRUE(state.mayAttempt(submitted, now));
    EXPECT_EQ(state.admissionAttempts(), 0u);
    EXPECT_FALSE(state.mayAttempt(true, now));
    EXPECT_EQ(state.tryAdmit(submitted, [&] {
        EXPECT_FALSE(state.mayAttempt(submitted, now)); // Reentrant preparation.
        return false;
    }, now), A::DECLINED);
    EXPECT_TRUE(state.mayAttempt(submitted, now));
    EXPECT_EQ(state.tryAdmit(submitted, [] { return false; }, now), A::DECLINED);
    EXPECT_FALSE(state.mayAttempt(submitted, now));
    EXPECT_EQ(state.admissionAttempts(), 2u);
    state.reset();
    state.recordEos(now);
    const auto deadline = state.admissionDeadline(std::chrono::milliseconds(10), now);
    EXPECT_TRUE(state.mayAttempt(submitted, now));
    EXPECT_FALSE(state.mayAttempt(submitted, deadline));
    EXPECT_EQ(state.admissionAttempts(), 0u);
    EXPECT_EQ(state.admissionDeadline(std::chrono::milliseconds(99), deadline), deadline);
}

TEST(DecodePublicationLifecycleTest, ReentryExceptionAndDeadlineKeepOriginalAdmissionBudget) {
    DecodePublicationLifecycle state;
    bool submitted = false;
    using A = DecodePublicationLifecycle::Admission;
    using Clock = DecodePublicationLifecycle::Clock;
    const auto now = Clock::now();
    state.recordEos(now);
    EXPECT_EQ(state.admissionDeadline(std::chrono::milliseconds(10), now), now + std::chrono::milliseconds(10));
    EXPECT_EQ(state.tryAdmit(submitted, [&] {
        EXPECT_EQ(state.tryAdmit(submitted, [] { return true; }, now), A::IN_PROGRESS);
        return false;
    }, now), A::DECLINED);
    EXPECT_THROW(state.tryAdmit(submitted, []() -> bool { throw 1; }, now), int);
    EXPECT_FALSE(submitted);
    EXPECT_EQ(state.admissionAttempts(), 2u);
    EXPECT_EQ(state.tryAdmit(submitted, [] { return true; }, now), A::EXHAUSTED);
    state.reset();
    state.recordEos(now);
    state.admissionDeadline(std::chrono::milliseconds(10), now);
    EXPECT_EQ(state.tryAdmit(submitted, [] { return true; }, now + std::chrono::milliseconds(10)), A::EXHAUSTED);
    EXPECT_EQ(state.admissionAttempts(), 0u);
}

}  // namespace rtp_llm
