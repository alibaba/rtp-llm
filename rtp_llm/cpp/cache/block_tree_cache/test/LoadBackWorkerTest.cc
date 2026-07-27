#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackWorker.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {

TEST(LoadBackWorkerTest, CreateTaskAllowsNoTransferItems) {
    LoadBackWorker    worker;
    ComponentGroupPtr group   = std::make_shared<FullComponentGroup>();
    group->component_group_id = 0;

    LoadBackTicket::PendingLoadBackItem joined_item;
    joined_item.group_id                                = 0;
    joined_item.source_tier                             = Tier::HOST;
    joined_item.joined_load_back                        = true;
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(1);
    LoadBackWorker::TaskPtr                     task    = std::make_shared<LoadBackWorker::Task>();

    ASSERT_TRUE(worker.createTask({joined_item}, {group}, context, task));
    EXPECT_EQ(task, nullptr);
}

TEST(LoadBackWorkerTest, FinishLoadingNotifiesJoinedContext) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{1, 2};
    const std::shared_ptr<LoadBackAsyncContext> first_context  = std::make_shared<LoadBackAsyncContext>(1);
    const std::shared_ptr<LoadBackAsyncContext> joined_context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, target_blocks, first_context));
    const std::optional<std::vector<BlockIdxType>> joined_blocks = worker.joinLoading(&node, 0, joined_context);
    ASSERT_TRUE(joined_blocks.has_value());
    EXPECT_EQ(joined_blocks.value(), target_blocks);
    EXPECT_FALSE(first_context->done());
    EXPECT_FALSE(joined_context->done());

    EXPECT_TRUE(worker.finishLoading(&node, 0, true));
    EXPECT_TRUE(first_context->done());
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->done());
    EXPECT_TRUE(joined_context->success());
    EXPECT_FALSE(worker.finishLoading(&node, 0, true));
}

TEST(LoadBackWorkerTest, DuplicateJoinOnlyCompletesOnce) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{3};
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 1, target_blocks, context));
    ASSERT_TRUE(worker.joinLoading(&node, 1, context).has_value());
    ASSERT_TRUE(worker.joinLoading(&node, 1, context).has_value());

    EXPECT_TRUE(worker.finishLoading(&node, 1, true));
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
}

TEST(LoadBackWorkerTest, CancelingFirstContextDoesNotFailJoinedContext) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadBackAsyncContext> first_context  = std::make_shared<LoadBackAsyncContext>(1);
    const std::shared_ptr<LoadBackAsyncContext> joined_context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {4}, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, joined_context).has_value());
    ASSERT_TRUE(first_context->requestCancel());
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));

    EXPECT_TRUE(first_context->done());
    EXPECT_FALSE(first_context->success());
    EXPECT_TRUE(joined_context->done());
    EXPECT_TRUE(joined_context->success());
}

TEST(LoadBackWorkerTest, CancelingJoinedContextDoesNotFailFirstContext) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadBackAsyncContext> first_context  = std::make_shared<LoadBackAsyncContext>(1);
    const std::shared_ptr<LoadBackAsyncContext> joined_context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {5}, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, joined_context).has_value());
    ASSERT_TRUE(joined_context->requestCancel());
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));

    EXPECT_TRUE(first_context->done());
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->done());
    EXPECT_FALSE(joined_context->success());
}

TEST(LoadBackWorkerTest, ContextAggregatesMultipleLoadingRecords) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(2);

    ASSERT_TRUE(worker.startLoading(&node, 0, {5}, context));
    ASSERT_TRUE(worker.startLoading(&node, 1, {6}, context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, context).has_value());
    ASSERT_TRUE(worker.joinLoading(&node, 1, context).has_value());

    EXPECT_TRUE(worker.finishLoading(&node, 0, true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(worker.finishLoading(&node, 1, false));
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST(LoadBackWorkerTest, ContextWaitsForJoinedAndNewLoadingRecords) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadBackAsyncContext> first_context = std::make_shared<LoadBackAsyncContext>(1);
    const std::shared_ptr<LoadBackAsyncContext> mixed_context = std::make_shared<LoadBackAsyncContext>(2);

    ASSERT_TRUE(worker.startLoading(&node, 0, {7}, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, mixed_context).has_value());
    ASSERT_TRUE(worker.startLoading(&node, 1, {8}, mixed_context));

    EXPECT_TRUE(worker.finishLoading(&node, 1, true));
    EXPECT_FALSE(mixed_context->done());
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));
    EXPECT_TRUE(first_context->done());
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(mixed_context->done());
    EXPECT_TRUE(mixed_context->success());
}

TEST(LoadBackWorkerTest, EraseLoadingOnlyRemovesSelectedContext) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{7};
    const std::shared_ptr<LoadBackAsyncContext> first_context  = std::make_shared<LoadBackAsyncContext>(1);
    const std::shared_ptr<LoadBackAsyncContext> second_context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, target_blocks, first_context));
    EXPECT_FALSE(worker.startLoading(&node, 0, target_blocks, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, first_context).has_value());
    ASSERT_TRUE(worker.joinLoading(&node, 0, second_context).has_value());

    EXPECT_TRUE(worker.eraseLoadingForOneContext(&node, 0, second_context));
    EXPECT_FALSE(worker.eraseLoadingForOneContext(&node, 0, second_context));
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));

    EXPECT_TRUE(first_context->done());
    EXPECT_TRUE(first_context->success());
    EXPECT_FALSE(second_context->done());
    EXPECT_TRUE(second_context->onTaskFail());
}

TEST(LoadBackWorkerTest, EraseLastLoadingContextRemovesRecord) {
    LoadBackWorker                              worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadBackAsyncContext> context = std::make_shared<LoadBackAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {8}, context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, context).has_value());
    EXPECT_TRUE(worker.eraseLoadingForOneContext(&node, 0, context));
    EXPECT_FALSE(worker.joinLoading(&node, 0, context).has_value());
    EXPECT_FALSE(worker.finishLoading(&node, 0, true));
    EXPECT_TRUE(context->onTaskFail());
}

}  // namespace rtp_llm
