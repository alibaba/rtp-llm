// Pure-CPU unit test for the frozen EPLB window reducer (reduceExpertLoadWindow).
// This is an independent numerical oracle: every expected value below is computed
// by hand from the contract semantics, per layer:
//   moe_lb        = sum_t(max_rank x[t,l,r]) / sum_t(mean_rank x[t,l,r])
//   moe_lb_window = max_rank(sum_t x[t,l,r]) / mean_rank(sum_t x[t,l,r])
// Windows are independent stateless calls; layers are never summed before max.
// No production lifecycle/monitor code is exercised here.

#include "rtp_llm/cpp/models/eplb/ExpertLoadMetrics.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

constexpr double kEps = 1e-12;

// Contract acceptance case 1: W=2 over steps [4,0],[0,4],[2,2],[0,4] must yield
// window ratios (2,1) then (1.5,1.5), with no cross-window accumulation.
TEST(ExpertLoadMetricsTest, TwoWindowsDoNotAccumulate) {
    const std::vector<int64_t> window1 = {4, 0, 0, 4};
    const std::vector<int64_t> window2 = {2, 2, 0, 4};

    const auto first = reduceExpertLoadWindow(window1, /*sample_count=*/2, /*layer_count=*/1, /*ep_size=*/2);
    ASSERT_EQ(first.size(), 1u);
    EXPECT_TRUE(first[0].valid);
    EXPECT_EQ(first[0].total_activations, 8);
    EXPECT_EQ(first[0].rank_activations, (std::vector<int64_t>{4, 4}));
    EXPECT_DOUBLE_EQ(first[0].moe_lb, 2.0);         // (4+4) / (2+2)
    EXPECT_DOUBLE_EQ(first[0].moe_lb_window, 1.0);  // max(4,4) / (8/2)

    const auto second = reduceExpertLoadWindow(window2, 2, 1, 2);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_TRUE(second[0].valid);
    EXPECT_EQ(second[0].total_activations, 8);
    EXPECT_EQ(second[0].rank_activations, (std::vector<int64_t>{2, 6}));
    EXPECT_DOUBLE_EQ(second[0].moe_lb, 1.5);         // (2+4) / (2+2)
    EXPECT_DOUBLE_EQ(second[0].moe_lb_window, 1.5);  // max(2,6) / (8/2)

    // Replaying the first window must reproduce it exactly: calls are stateless
    // and the second window left no residue behind.
    const auto replay = reduceExpertLoadWindow(window1, 2, 1, 2);
    ASSERT_EQ(replay.size(), 1u);
    EXPECT_EQ(replay[0].total_activations, first[0].total_activations);
    EXPECT_EQ(replay[0].rank_activations, first[0].rank_activations);
    EXPECT_DOUBLE_EQ(replay[0].moe_lb, first[0].moe_lb);
    EXPECT_DOUBLE_EQ(replay[0].moe_lb_window, first[0].moe_lb_window);
}

// moe_lb is sum-of-max over sum-of-mean: (8+1)/(5+1) = 1.5. It is NOT the
// unweighted average of per-step ratios, which would be (8/5 + 1)/2 = 1.3.
TEST(ExpertLoadMetricsTest, MoeLbIsWeightedNotUnweightedAverage) {
    const auto result = reduceExpertLoadWindow({8, 2, 1, 1}, 2, 1, 2);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].valid);
    EXPECT_EQ(result[0].total_activations, 12);
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{9, 3}));
    EXPECT_DOUBLE_EQ(result[0].moe_lb, 1.5);
    EXPECT_DOUBLE_EQ(result[0].moe_lb_window, 1.5);           // max(9,3) / (12/2)
    const double naive_unweighted = (8.0 / 5.0 + 1.0) / 2.0;  // 1.3
    EXPECT_GT(std::fabs(result[0].moe_lb - naive_unweighted), 0.1);
}

// Empty sample count and all-zero layers report valid=false with zeroed ratios,
// never a fabricated 1.
TEST(ExpertLoadMetricsTest, EmptyAndAllZeroWindowsAreInvalid) {
    const auto zero = reduceExpertLoadWindow({0, 0, 0, 0}, 2, 1, 2);
    ASSERT_EQ(zero.size(), 1u);
    EXPECT_FALSE(zero[0].valid);
    EXPECT_EQ(zero[0].total_activations, 0);
    EXPECT_EQ(zero[0].rank_activations, (std::vector<int64_t>{0, 0}));
    EXPECT_DOUBLE_EQ(zero[0].moe_lb, 0.0);
    EXPECT_DOUBLE_EQ(zero[0].moe_lb_window, 0.0);

    const auto empty = reduceExpertLoadWindow({}, /*sample_count=*/0, 1, 2);
    ASSERT_EQ(empty.size(), 1u);
    EXPECT_FALSE(empty[0].valid);
    EXPECT_EQ(empty[0].total_activations, 0);
    EXPECT_EQ(empty[0].rank_activations.size(), 2u);
    EXPECT_DOUBLE_EQ(empty[0].moe_lb, 0.0);
    EXPECT_DOUBLE_EQ(empty[0].moe_lb_window, 0.0);

    // A zero layer stays invalid next to a valid one: layers are independent.
    const auto mixed = reduceExpertLoadWindow({2, 2, 0, 0}, 1, 2, 2);
    ASSERT_EQ(mixed.size(), 2u);
    EXPECT_TRUE(mixed[0].valid);
    EXPECT_DOUBLE_EQ(mixed[0].moe_lb, 1.0);
    EXPECT_DOUBLE_EQ(mixed[0].moe_lb_window, 1.0);
    EXPECT_FALSE(mixed[1].valid);
}

// Two layers with opposite hotspots. Summing layers before max would merge
// [4,0]+[0,4] into a flat [4,4] and report 1.0; each layer must report 2.0.
TEST(ExpertLoadMetricsTest, LayersAreReducedIndependently) {
    const auto result = reduceExpertLoadWindow({4, 0, 0, 4, 4, 0, 0, 4}, 2, 2, 2);
    ASSERT_EQ(result.size(), 2u);
    for (size_t layer = 0; layer < 2; ++layer) {
        EXPECT_TRUE(result[layer].valid);
        EXPECT_EQ(result[layer].total_activations, 8);
        EXPECT_DOUBLE_EQ(result[layer].moe_lb, 2.0);
        EXPECT_DOUBLE_EQ(result[layer].moe_lb_window, 2.0);
    }
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{8, 0}));
    EXPECT_EQ(result[1].rank_activations, (std::vector<int64_t>{0, 8}));
}

// EP=3 with an idle rank: the mean denominator stays the full EP size (3), so a
// single active rank at 3/step yields moe_lb 3.0, not 1.0.
TEST(ExpertLoadMetricsTest, IdleRankKeepsFullEpDenominator) {
    const auto result = reduceExpertLoadWindow({3, 0, 0, 0, 3, 0}, 2, 1, 3);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].valid);
    EXPECT_EQ(result[0].total_activations, 6);
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{3, 3, 0}));
    EXPECT_DOUBLE_EQ(result[0].moe_lb, 3.0);         // (3+3) / (1+1)
    EXPECT_DOUBLE_EQ(result[0].moe_lb_window, 1.5);  // max(3,3,0) / (6/3)
}

// Input is already globally SUM-reduced across sources: [2,4] gives both ratios
// 4/3, matching the frozen artifact schema example (1.3333333333333333).
TEST(ExpertLoadMetricsTest, CrossSourceSummedWindow) {
    const auto result = reduceExpertLoadWindow({2, 4}, 1, 1, 2);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].valid);
    EXPECT_EQ(result[0].total_activations, 6);
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{2, 4}));
    EXPECT_NEAR(result[0].moe_lb, 4.0 / 3.0, kEps);
    EXPECT_NEAR(result[0].moe_lb_window, 4.0 / 3.0, kEps);
}

// Counts beyond INT32 range must stay exact in int64 accumulators.
TEST(ExpertLoadMetricsTest, CountsAboveInt32StayExact) {
    constexpr int64_t kThreeBillion = 3000000000LL;
    constexpr int64_t kOneBillion   = 1000000000LL;
    const auto        result        = reduceExpertLoadWindow({kThreeBillion, kOneBillion}, 1, 1, 2);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].valid);
    EXPECT_EQ(result[0].total_activations, kThreeBillion + kOneBillion);  // > INT32_MAX
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{kThreeBillion, kOneBillion}));
    EXPECT_DOUBLE_EQ(result[0].moe_lb, 1.5);
    EXPECT_DOUBLE_EQ(result[0].moe_lb_window, 1.5);
}

TEST(ExpertLoadMetricsTest, RejectsInvalidInputs) {
    const int64_t kMax = std::numeric_limits<int64_t>::max();

    // Negative counts.
    const std::vector<int64_t> negative = {kMax - 1, -5};
    EXPECT_THROW(reduceExpertLoadWindow(negative, 1, 1, 2), std::invalid_argument);

    // Shape mismatch: too few and too many elements for [step, layer, ep_rank].
    const std::vector<int64_t> too_short = {4, 0};
    const std::vector<int64_t> too_long  = {4, 0, 0, 4, 1};
    EXPECT_THROW(reduceExpertLoadWindow(too_short, 2, 1, 2), std::invalid_argument);
    EXPECT_THROW(reduceExpertLoadWindow(too_long, 2, 1, 2), std::invalid_argument);

    // Non-positive structural dimensions.
    const std::vector<int64_t> empty;
    EXPECT_THROW(reduceExpertLoadWindow(empty, 1, 0, 2), std::invalid_argument);
    EXPECT_THROW(reduceExpertLoadWindow(empty, 1, 1, 0), std::invalid_argument);

    // size_t dimension product overflow.
    const size_t kHuge = std::numeric_limits<size_t>::max() / 2 + 1;
    EXPECT_THROW(reduceExpertLoadWindow(empty, kHuge, 2, 2), std::invalid_argument);

    // int64 accumulation overflow, across ranks within a step and across steps.
    const std::vector<int64_t> overflow_rank = {kMax, 1};
    const std::vector<int64_t> overflow_step = {kMax, 0, 1, 0};
    EXPECT_THROW(reduceExpertLoadWindow(overflow_rank, 1, 1, 2), std::overflow_error);
    EXPECT_THROW(reduceExpertLoadWindow(overflow_step, 2, 1, 2), std::overflow_error);
}

// Single EP rank: max == mean at every step, so both ratios are exactly 1.
TEST(ExpertLoadMetricsTest, SingleRankIsPerfectlyBalanced) {
    const auto result = reduceExpertLoadWindow({5, 7}, 2, 1, 1);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].valid);
    EXPECT_EQ(result[0].total_activations, 12);
    EXPECT_EQ(result[0].rank_activations, (std::vector<int64_t>{12}));
    EXPECT_DOUBLE_EQ(result[0].moe_lb, 1.0);
    EXPECT_DOUBLE_EQ(result[0].moe_lb_window, 1.0);
}

}  // namespace
}  // namespace rtp_llm
