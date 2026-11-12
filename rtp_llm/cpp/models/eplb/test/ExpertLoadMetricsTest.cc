#include "rtp_llm/cpp/models/eplb/test/ExpertLoadMetricsReference.h"

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(ExpertLoadMetricsTest, DividesSumOfStepMaximaBySumOfStepMeans) {
    // Heavy balanced step [4,4], then light unbalanced step [2,0].
    const auto result = reduceExpertLoadWindowReference({4, 2, 4, 0}, 2, 1, 2, 2);
    EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 1.2);
    EXPECT_EQ(result.aggregate.max_rank_activations, 6);
    EXPECT_EQ(result.aggregate.total_activations, 10);
    EXPECT_EQ(result.aggregate.rank_activations, (std::vector<int64_t>{6, 4}));
    EXPECT_DOUBLE_EQ(result.layers[0].moe_lb, 1.2);
    // An activation-weighted inverse balance would be 10/9, not 6/5.
}

TEST(ExpertLoadMetricsTest, AlternatingHotspotsAreNotFlattenedAcrossSteps) {
    const auto result = reduceExpertLoadWindowReference({8, 0, 0, 8}, 2, 1, 2, 2);
    EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 2.0);
    EXPECT_EQ(result.aggregate.rank_activations, (std::vector<int64_t>{8, 8}));
}

TEST(ExpertLoadMetricsTest, AggregatesLayerMaximaBeforeTakingTheRatio) {
    // Alternating hotspots in layer 0, balanced layer 1, and zero-load layer 2.
    const auto result = reduceExpertLoadWindowReference({8, 2, 0, 0, 0, 0, 0, 2, 0, 8, 0, 0}, 2, 3, 2, 2);
    ASSERT_EQ(result.layers.size(), 3u);
    EXPECT_DOUBLE_EQ(result.layers[0].moe_lb, 2.0);
    EXPECT_DOUBLE_EQ(result.layers[1].moe_lb, 1.0);
    EXPECT_DOUBLE_EQ(result.layers[2].moe_lb, 1.0);
    EXPECT_EQ(result.layers[2].rank_activations, (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(result.aggregate.max_rank_activations, 18);
    EXPECT_EQ(result.aggregate.total_activations, 20);
    EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 1.8);
}

TEST(ExpertLoadMetricsTest, ZeroWorkAndEmptyWindowsYieldOne) {
    for (const auto& result :
         {reduceExpertLoadWindowReference({}, 0, 2, 2, 2), reduceExpertLoadWindowReference({0, 0, 0, 0}, 1, 2, 2, 2)}) {
        EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 1.0);
        EXPECT_EQ(result.aggregate.total_activations, 0);
        EXPECT_EQ(result.aggregate.max_rank_activations, 0);
        for (const auto& layer : result.layers) {
            EXPECT_DOUBLE_EQ(layer.moe_lb, 1.0);
            EXPECT_EQ(layer.rank_activations, (std::vector<int64_t>{0, 0}));
        }
    }
}

TEST(ExpertLoadMetricsTest, ReplicatedTpRanksUseOneEpOwner) {
    const auto result = reduceExpertLoadWindowReference({8, 4, 0, 0, 0, 0}, 2, 1, 3, 1);
    EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 1.0);
    EXPECT_EQ(result.aggregate.total_activations, 12);
    EXPECT_EQ(result.aggregate.rank_activations, (std::vector<int64_t>{12}));
}

TEST(ExpertLoadMetricsTest, IndependentWindowsKeepTheirOwnStatistics) {
    const auto first  = reduceExpertLoadWindowReference({4, 0, 0, 4}, 2, 1, 2, 2);
    const auto second = reduceExpertLoadWindowReference({2, 0, 2, 4}, 2, 1, 2, 2);
    EXPECT_DOUBLE_EQ(first.aggregate.moe_lb, 2.0);
    EXPECT_DOUBLE_EQ(second.aggregate.moe_lb, 1.5);
    EXPECT_EQ(second.aggregate.total_activations, 8);
}

TEST(ExpertLoadMetricsTest, CountsStayInt64ThroughReduction) {
    const auto result = reduceExpertLoadWindowReference({3000000000LL, 1000000000LL}, 1, 1, 2, 2);
    EXPECT_EQ(result.aggregate.total_activations, 4000000000LL);
    EXPECT_DOUBLE_EQ(result.aggregate.moe_lb, 1.5);
}

TEST(ExpertLoadMetricsTest, RejectsMismatchedShapesAndUnsupportedEpLayouts) {
    EXPECT_THROW(reduceExpertLoadWindowReference({1}, 1, 1, 2, 2), std::invalid_argument);
    EXPECT_THROW(reduceExpertLoadWindowReference({}, 0, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(reduceExpertLoadWindowReference({}, 0, 1, 0, 1), std::invalid_argument);
    EXPECT_THROW(reduceExpertLoadWindowReference({}, 0, 1, 4, 2), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
