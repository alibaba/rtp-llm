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

TEST(DSparkCudaGraphBudgetTest, CapacityPlanSupportsOtherProperPackedWidths) {
    struct Case {
        int     gamma, budget;
        int64_t batch, graph_batch, rows;
    };
    for (const auto& shape : std::vector<Case>{
             {1, 0, 1, 1, 1}, {1, 0, 3, 4, 4}, {3, 0, 3, 4, 4}, {3, 1, 3, 4, 8}, {5, 1, 1, 4, 6}, {5, 4, 3, 4, 18}}) {
        const int64_t width    = shape.budget + 1;
        const int64_t capacity = shape.graph_batch * width;
        auto plan = makeDSparkCapacityPlan({shape.graph_batch, capacity, shape.budget}, shape.batch, shape.gamma);
        ASSERT_TRUE(plan);
        EXPECT_EQ(plan->query_rows, shape.rows);
        EXPECT_TRUE(isDSparkCapacityShape(*plan, shape.rows));
        EXPECT_TRUE(isDSparkCapacityShape(*plan, shape.batch * width));
        int64_t total = shape.rows;
        for (int64_t ghost = 0; ghost < shape.graph_batch - shape.batch; ++ghost) {
            total += dsparkDummyQueryLength(shape.rows, capacity, ghost, width);
        }
        EXPECT_EQ(total, capacity);
    }
    EXPECT_FALSE(makeDSparkCapacityPlan({4, 4, 0}, 3, std::numeric_limits<int>::max()));
    EXPECT_FALSE(makeDSparkCapacityPlan({4, 4, -1}, 3, 5));
    EXPECT_FALSE(makeDSparkCapacityPlan({4, 24, 5}, 3, 5));
    EXPECT_FALSE(makeDSparkCapacityPlan({std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 0},
                                        std::numeric_limits<int64_t>::max(),
                                        1));
}

TEST(DSparkCudaGraphBudgetTest, CapacityPlanSpendsOnlyTheActualCapturedTokenCapacity) {
    struct Case {
        int64_t batch, graph_batch, rows, draft_budget;
    };
    for (const auto& shape : std::vector<Case>{{1, 1, 4, 3},
                                               {2, 2, 8, 6},
                                               {3, 4, 16, 13},
                                               {4, 4, 16, 12},
                                               {5, 8, 30, 25},
                                               {6, 8, 32, 26},
                                               {9, 16, 54, 45}}) {
        auto plan = makeDSparkCapacityPlan({shape.graph_batch, shape.graph_batch * 4, 3}, shape.batch, 5);
        ASSERT_TRUE(plan);
        EXPECT_EQ(plan->query_rows, shape.rows);
        EXPECT_EQ(plan->draftBudget(), shape.draft_budget);
        EXPECT_TRUE(isDSparkCapacityShape(*plan, shape.rows));
        EXPECT_TRUE(isDSparkCapacityShape(*plan, 4 * shape.batch));  // unchanged whole-idle DP fake shape
        EXPECT_FALSE(isDSparkCapacityShape(*plan, shape.rows + 1));
        int64_t sum = shape.rows;
        for (int64_t ghost = 0; ghost < shape.graph_batch - shape.batch; ++ghost) {
            const auto length = dsparkDummyQueryLength(shape.rows, shape.graph_batch * 4, ghost, 4);
            EXPECT_GE(length, 0);
            EXPECT_LE(length, 4);
            sum += length;
        }
        EXPECT_EQ(sum, shape.graph_batch * 4);
    }
    // A partial Q/B does not identify a new bank; physical bank3 remains packed.
    auto fractional = makeDSparkCapacityPlan({4, 16, 3}, 3, 5);
    ASSERT_TRUE(fractional);
    EXPECT_EQ(fractional->query_rows % fractional->real_batch, 1);
    EXPECT_EQ(fractional->graph.capture_budget, 3);
    EXPECT_EQ(dsparkDummyQueryLength(16, 16, 0, 4), 0);
    EXPECT_EQ(dsparkDummyQueryLength(30, 32, 0, 4), 2);
    EXPECT_EQ(dsparkDummyQueryLength(30, 32, 1, 4), 0);
    for (const auto& geometry : std::vector<DSparkCapturedCapacity>{
             {4, 15, 3}, {4, 16, 2}, {2, 8, 3}, {std::numeric_limits<int64_t>::max(), 16, 3}}) {
        EXPECT_FALSE(makeDSparkCapacityPlan(geometry, 3, 5));
    }
    EXPECT_FALSE(makeDSparkCapacityPlan({4, 16, 3}, 0, 5));
    EXPECT_FALSE(makeDSparkCapacityPlan({4, 16, 3}, 3, 3));
}

}  // namespace rtp_llm
