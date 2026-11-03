#include <gtest/gtest.h>

#include <limits>
#include <utility>
#include "rtp_llm/cpp/cuda_graph/dspark_cuda_graph_budget.h"

namespace rtp_llm {

TEST(DSparkCudaGraphBudgetTest, FixedBudgetCapturesOnlyItsOwnGeometry) {
    for (int configured = -1; configured <= 5; ++configured) {
        const auto plans = makeDSparkCudaGraphBudgets(5, configured, false);
        ASSERT_EQ(plans.size(), 1u);
        const int expected = configured < 0 ? 5 : configured;
        EXPECT_EQ(plans[0].budget, expected);
        EXPECT_EQ(plans[0].packed, expected < 5);
        for (int budget = 0; budget <= 5; ++budget) {
            const auto selected = selectDSparkCudaGraphBudget(plans, budget < 5, 4, 4 * (budget + 1));
            EXPECT_EQ(selected.has_value(), budget == expected);
        }
    }
}

TEST(DSparkCudaGraphBudgetTest, AdaptiveCaptureOrderDoesNotDependOnRankFallback) {
    for (int fallback = 0; fallback <= 5; ++fallback) {
        const auto plans = makeDSparkCudaGraphBudgets(5, fallback, true);
        ASSERT_EQ(plans.size(), 6u);
        for (int budget = 0; budget <= 5; ++budget) {
            EXPECT_EQ(plans[budget].budget, budget);
            EXPECT_EQ(plans[budget].tokensPerBatch(), budget + 1);
            EXPECT_EQ(plans[budget].packed, budget < 5);
        }
    }
}

TEST(DSparkCudaGraphBudgetTest, IndependentRanksCanSwitchZeroFullAndPackedWithoutKeyCollision) {
    const auto rank0 = makeDSparkCudaGraphBudgets(5, 3, true);
    const auto rank1 = makeDSparkCudaGraphBudgets(5, 1, true);
    for (int budget : {3, 0, 5, 3, 1, 4, 2}) {
        for (int64_t batch : {1, 2, 3, 4}) {
            ASSERT_EQ(selectDSparkCudaGraphBudget(rank0, budget < 5, batch, batch * (budget + 1)), budget);
            const int peer_budget = 5 - budget;
            ASSERT_EQ(selectDSparkCudaGraphBudget(rank1, peer_budget < 5, batch, batch * (peer_budget + 1)),
                      peer_budget);
        }
    }
    // Equal row counts do not imply equal captured request/width geometry.
    EXPECT_EQ(selectDSparkCudaGraphBudget(rank0, false, 2, 12), 5u);
    EXPECT_EQ(selectDSparkCudaGraphBudget(rank0, true, 4, 12), 2u);
    EXPECT_EQ(selectDSparkCudaGraphBudget(rank0, true, 3, 12), 3u);
}

TEST(DSparkCudaGraphBudgetTest, RejectsMissingOrMismatchedGeometry) {
    const auto plans = makeDSparkCudaGraphBudgets(5, 3, true);
    for (const auto& shape : std::vector<std::pair<int64_t, int64_t>>{{0, 0}, {-1, 4}, {2, 0}, {3, 8}, {1, 7}}) {
        EXPECT_FALSE(selectDSparkCudaGraphBudget(plans, true, shape.first, shape.second));
    }
    EXPECT_FALSE(selectDSparkCudaGraphBudget(plans, false, 4, 4));
    EXPECT_FALSE(selectDSparkCudaGraphBudget(plans, true, 4, 24));
    EXPECT_FALSE(selectDSparkCudaGraphBudget({}, true, 4, 16));
    EXPECT_FALSE(selectDSparkCudaGraphBudget(plans, true, 2, std::numeric_limits<int64_t>::max()));
    EXPECT_THROW(makeDSparkCudaGraphBudgets(0, 0, false), std::invalid_argument);
    EXPECT_THROW(makeDSparkCudaGraphBudgets(5, -2, false), std::invalid_argument);
    EXPECT_THROW(makeDSparkCudaGraphBudgets(5, 6, true), std::invalid_argument);
    EXPECT_THROW(makeDSparkCudaGraphBudgets(std::numeric_limits<int>::max(), 3, false), std::invalid_argument);
}

}  // namespace rtp_llm
