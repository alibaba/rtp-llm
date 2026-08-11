#include "rtp_llm/cpp/model_rpc/RpcFanoutUtils.h"

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(RpcFanoutUtilsTest, EmptyFanoutHasNoCompletionQueues) {
    const auto plan = makeCompletionQueuePlan(0);
    EXPECT_EQ(plan.queue_count, 0);
    EXPECT_TRUE(plan.expected_completions.empty());
    EXPECT_THROW(plan.queueIndexForWorker(0), std::logic_error);
}

TEST(RpcFanoutUtilsTest, OddWorkerCountsKeepEveryExpectedCompletion) {
    const auto plan = makeCompletionQueuePlan(5);
    ASSERT_EQ(plan.queue_count, 3);
    EXPECT_EQ(plan.expected_completions, (std::vector<size_t>{2, 2, 1}));

    std::vector<size_t> observed(plan.queue_count, 0);
    for (size_t worker_index = 0; worker_index < 5; ++worker_index) {
        ++observed[plan.queueIndexForWorker(worker_index)];
    }
    EXPECT_EQ(observed, plan.expected_completions);
}

TEST(RpcFanoutUtilsTest, EvenWorkerCountsRemainBalanced) {
    const auto plan = makeCompletionQueuePlan(6);
    ASSERT_EQ(plan.queue_count, 3);
    EXPECT_EQ(plan.expected_completions, (std::vector<size_t>{2, 2, 2}));
}

}  // namespace rtp_llm
