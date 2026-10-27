#include "rtp_llm/cpp/normal_engine/speculative/DSparkAdaptiveBudget.h"

#include <limits>
#include <numeric>
#include "gtest/gtest.h"

namespace rtp_llm {
namespace speculative {
namespace {

void populateCosts(DSparkAdaptiveBudget& policy, size_t batch, const std::vector<double>& costs) {
    for (size_t budget = 0; budget < costs.size(); ++budget) {
        for (int sample = 0; sample < 5; ++sample) {
            policy.observeCost(batch, budget, costs[budget]);
        }
    }
}

TEST(DSparkAdaptiveBudgetTest, RejectsInvalidConfigurationAndDuplicateIdentities) {
    EXPECT_THROW(DSparkAdaptiveBudget(0, 0), std::invalid_argument);
    EXPECT_THROW(DSparkAdaptiveBudget(-1, 0), std::invalid_argument);
    EXPECT_THROW(DSparkAdaptiveBudget(3, -1), std::invalid_argument);
    EXPECT_THROW(DSparkAdaptiveBudget(3, 4), std::invalid_argument);
    DSparkAdaptiveBudget policy(3, 1);
    EXPECT_THROW(policy.choose({7, 7}), std::invalid_argument);
    EXPECT_THROW(policy.observeConfidence({7, 7}, {{1, 1, 1}, {1, 1, 1}}), std::invalid_argument);
    EXPECT_EQ(policy.choose({}).total_draft_budget, 0);
}

TEST(DSparkAdaptiveBudgetTest, ColdStartRotatesFromFallbackUntilEveryCandidateHasFiveSamples) {
    DSparkAdaptiveBudget policy(2, 1);
    for (int round = 0; round < 15; ++round) {
        const auto plan = policy.choose({10});
        EXPECT_TRUE(plan.exploring);
        EXPECT_EQ(plan.budget_per_request, (round + 1) % 3);
        EXPECT_EQ(plan.estimated_ms, 0.0);
        policy.observeCost(1, plan.budget_per_request, 1.0 + plan.budget_per_request);
    }
    const auto learned = policy.choose({10});
    EXPECT_FALSE(learned.exploring);
    EXPECT_EQ(learned.budget_per_request, 0);  // Throughputs tie; prefer smaller budget.
}

TEST(DSparkAdaptiveBudgetTest, MissingObservationDoesNotTurnIntoAnInventedCost) {
    DSparkAdaptiveBudget policy(2, 1);
    populateCosts(policy, 1, {1.0, 1.0});
    for (int round = 0; round < 20; ++round) {
        const auto plan = policy.choose({1});
        EXPECT_TRUE(plan.exploring);
        EXPECT_EQ(plan.budget_per_request, 2);
        EXPECT_EQ(plan.estimated_ms, 0.0);
    }
}

TEST(DSparkAdaptiveBudgetTest, ConfidenceChangesOptimalBudgetOnTheSameMeasuredCurve) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 1, {1.0, 1.2, 1.8, 2.0});
    policy.observeConfidence({11}, {{0.9f, 0.9f, 0.9f}});
    const auto high = policy.choose({11});
    EXPECT_EQ(high.budget_per_request, 3);
    EXPECT_NEAR(high.expected_tokens, 3.439, 1e-6);
    policy.observeConfidence({11}, {{0.1f, 0.1f, 0.1f}});
    const auto low = policy.choose({11});
    EXPECT_EQ(low.budget_per_request, 0);
    EXPECT_DOUBLE_EQ(low.expected_tokens, 1.0);
    // Publishing observations never mutates a previously frozen decision.
    EXPECT_EQ(high.total_draft_budget, 3);
}

TEST(DSparkAdaptiveBudgetTest, FlatAndSteepCurvesProduceDifferentBudgets) {
    DSparkAdaptiveBudget flat(3, 1);
    DSparkAdaptiveBudget steep(3, 1);
    populateCosts(flat, 1, {1.0, 1.0, 1.0, 1.0});
    populateCosts(steep, 1, {1.0, 1.2, 3.0, 4.0});
    EXPECT_EQ(flat.choose({1}).budget_per_request, 3);
    EXPECT_EQ(steep.choose({1}).budget_per_request, 1);
}

TEST(DSparkAdaptiveBudgetTest, WholeRoundOverheadCanChangeTheOptimalBudget) {
    DSparkAdaptiveBudget target_only(3, 1);
    DSparkAdaptiveBudget full_round(3, 1);
    populateCosts(target_only, 2, {1.0, 1.1, 5.0, 5.0});
    populateCosts(full_round, 2, {21.0, 21.1, 25.0, 25.0});
    for (auto* policy : {&target_only, &full_round}) {
        policy->observeConfidence({10, 20}, {{0.9f, 0.9f, 0.9f}, {0.5f, 0.5f, 0.5f}});
    }
    EXPECT_EQ(target_only.choose({10, 20}).budget_per_request, 1);
    EXPECT_EQ(full_round.choose({10, 20}).budget_per_request, 3);
}

TEST(DSparkAdaptiveBudgetTest, ExpectedAcceptanceUsesGlobalTopKOfCumulativeRates) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 2, {1.0, 1.1, 5.0, 5.0});
    policy.observeConfidence({10, 20}, {{0.9f, 0.9f, 0.9f}, {0.5f, 0.5f, 0.5f}});
    const auto plan = policy.choose({10, 20});
    EXPECT_EQ(plan.total_draft_budget, 2);
    EXPECT_NEAR(plan.expected_tokens, 2.0 + 0.9 + 0.81, 1e-6);
}

TEST(DSparkAdaptiveBudgetTest, HistoryTracksRequestIdentityAcrossBatchReordering) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 2, {1.0, 1.1, 5.0, 5.0});
    policy.observeConfidence({10, 20}, {{0.9f, 0.9f, 0.9f}, {0.5f, 0.5f, 0.5f}});
    const auto original  = policy.choose({10, 20});
    const auto reordered = policy.choose({20, 10});
    EXPECT_EQ(original.budget_per_request, reordered.budget_per_request);
    EXPECT_DOUBLE_EQ(original.expected_tokens, reordered.expected_tokens);
}

TEST(DSparkAdaptiveBudgetTest, FreshRequestsUseOnePriorAndPrunedIdsCannotInheritOldHistory) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 2, {1.0, 1.0, 1.0, 1.0});
    policy.observeConfidence({10, 20}, {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f}});
    const auto initial = policy.choose({10, 20});
    EXPECT_DOUBLE_EQ(initial.expected_tokens, 2.875);
    policy.pruneHistory({10});
    const auto replaced = policy.choose({10, 20});
    EXPECT_DOUBLE_EQ(replaced.expected_tokens, 5.875);
    EXPECT_EQ(replaced.budget_per_request, 3);
    // An empty active batch also releases history.
    policy.choose({});
    EXPECT_DOUBLE_EQ(policy.choose({10, 20}).expected_tokens, 8.0);
}

TEST(DSparkAdaptiveBudgetTest, InvalidConfidenceCannotPoisonExpectedTokens) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 1, {1.0, 1.0, 1.0, 1.0});
    for (const float invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -0.1f, 1.1f}) {
        policy.observeConfidence({1}, {{0.5f, invalid, 0.9f}});
        const auto plan = policy.choose({1});
        EXPECT_EQ(plan.budget_per_request, 1);
        EXPECT_DOUBLE_EQ(plan.expected_tokens, 1.5);
    }
}

TEST(DSparkAdaptiveBudgetTest, InvalidConfidenceShapeDoesNotPartiallyPublishRows) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 2, {1.0, 1.0, 1.0, 1.0});
    policy.observeConfidence({1, 2}, {{0, 0, 0}, {0, 0, 0}});
    EXPECT_THROW(policy.observeConfidence({1}, {}), std::invalid_argument);
    EXPECT_THROW(policy.observeConfidence({1, 2}, {{1, 1, 1}, {1}}), std::invalid_argument);
    EXPECT_EQ(policy.choose({1, 2}).budget_per_request, 0);
}

TEST(DSparkAdaptiveBudgetTest, CostsRemainIsolatedByActualBatchSize) {
    DSparkAdaptiveBudget policy(3, 1);
    populateCosts(policy, 1, {1.0, 1.0, 1.0, 1.0});
    populateCosts(policy, 2, {1.0, 3.0, 6.0, 10.0});
    EXPECT_EQ(policy.choose({1}).budget_per_request, 3);
    EXPECT_EQ(policy.choose({1, 2}).budget_per_request, 0);
    const auto cold_batch = policy.choose({1, 2, 3});
    EXPECT_TRUE(cold_batch.exploring);
    EXPECT_EQ(cold_batch.budget_per_request, 1);
}

TEST(DSparkAdaptiveBudgetTest, InvalidTimingsDoNotCountTowardWarmup) {
    DSparkAdaptiveBudget policy(1, 1);
    for (int sample = 0; sample < 5; ++sample) {
        policy.observeCost(1, 0, 1.0);
    }
    for (int sample = 0; sample < 4; ++sample) {
        policy.observeCost(1, 1, 1.0);
    }
    for (double value :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        policy.observeCost(1, 1, value);
    }
    policy.observeCost(0, 1, 1.0);
    policy.observeCost(1, -1, 1.0);
    policy.observeCost(1, 2, 1.0);
    EXPECT_TRUE(policy.choose({1}).exploring);
    policy.observeCost(1, 1, 1.0);
    EXPECT_FALSE(policy.choose({1}).exploring);
}

TEST(DSparkAdaptiveBudgetTest, SlidingMedianResistsOutliersAndAdaptsToNewCosts) {
    DSparkAdaptiveBudget policy(1, 1);
    populateCosts(policy, 1, {1.0});
    for (const double sample : {1.0, 1.0, 1.0, 1000.0, 1000.0}) {
        policy.observeCost(1, 1, sample);
    }
    EXPECT_EQ(policy.choose({1}).budget_per_request, 1);
    for (int sample = 0; sample < 4; ++sample) {
        policy.observeCost(1, 1, 1000.0);
    }
    EXPECT_EQ(policy.choose({1}).budget_per_request, 0);
    for (int sample = 0; sample < 9; ++sample) {
        policy.observeCost(1, 1, 0.5);
    }
    EXPECT_EQ(policy.choose({1}).budget_per_request, 1);
}

TEST(DSparkAdaptiveBudgetTest, LearnedPolicyPeriodicallyProbesEveryCandidate) {
    DSparkAdaptiveBudget policy(2, 2);
    populateCosts(policy, 1, {1.0, 10.0, 100.0});
    for (int period = 0; period < 3; ++period) {
        for (int round = 1; round < 32; ++round) {
            const auto plan = policy.choose({1});
            EXPECT_FALSE(plan.exploring);
            EXPECT_EQ(plan.budget_per_request, 0);
        }
        const auto probe = policy.choose({1});
        EXPECT_TRUE(probe.exploring);
        EXPECT_EQ(probe.budget_per_request, (period + 2) % 3);
    }
}

TEST(DSparkAdaptiveBudgetTest, CostMapEvictsTheLeastRecentlyUsedBatchAfter64Entries) {
    DSparkAdaptiveBudget policy(1, 1);
    for (size_t batch = 1; batch <= 64; ++batch) {
        populateCosts(policy, batch, {1.0, 1.0});
    }
    EXPECT_FALSE(policy.choose({1}).exploring);  // Keep batch 1 recent; evict batch 2.
    populateCosts(policy, 65, {1.0, 1.0});
    EXPECT_FALSE(policy.choose({1}).exploring);
    EXPECT_TRUE(policy.choose({1, 2}).exploring);
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
