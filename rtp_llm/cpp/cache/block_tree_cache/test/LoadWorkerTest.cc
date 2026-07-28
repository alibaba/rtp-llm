#include "rtp_llm/cpp/cache/block_tree_cache/LoadWorker.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

GroupSetPtr makeWorkerTestGroupSet() {
    using namespace block_transfer_engine_test;

    TestGroupSpec spec;
    spec.tag                                            = "load_worker";
    spec.policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    spec.policy.enable_prefix_reuse                     = true;
    spec.layer_ids                                      = {0};
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({spec});
    DeviceBlockPoolPtr pool = makeTestDevicePool({{spec.kv_block_stride_bytes, spec.kv_scale_stride_bytes}},
                                                 /*usable_count=*/1,
                                                 "load_worker");
    return makeTestGroupSet(0, topology, {0}, {std::move(pool)});
}

TEST(LoadWorkerTest, CreateTaskAllowsNoTransferItems) {
    LoadWorker     worker;
    GroupSetPtr    group = makeWorkerTestGroupSet();

    LoadTicket::PendingLoadItem joined_item;
    joined_item.group_set_id                            = 0;
    joined_item.source_tier                             = Tier::HOST;
    joined_item.joined_load                             = true;
    const std::shared_ptr<LoadAsyncContext> context     = std::make_shared<LoadAsyncContext>(1);
    LoadWorker::TaskPtr                     task        = std::make_shared<LoadWorker::Task>();

    ASSERT_TRUE(worker.createTask({joined_item}, {group}, context, task));
    EXPECT_EQ(task, nullptr);
}

TEST(LoadWorkerTest, FinishLoadingNotifiesJoinedContext) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{1, 2};
    const std::shared_ptr<LoadAsyncContext>     first_context  = std::make_shared<LoadAsyncContext>(1);
    const std::shared_ptr<LoadAsyncContext>     joined_context = std::make_shared<LoadAsyncContext>(1);

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

TEST(LoadWorkerTest, DuplicateJoinOnlyCompletesOnce) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{3};
    const std::shared_ptr<LoadAsyncContext>     context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 1, target_blocks, context));
    ASSERT_TRUE(worker.joinLoading(&node, 1, context).has_value());
    ASSERT_TRUE(worker.joinLoading(&node, 1, context).has_value());

    EXPECT_TRUE(worker.finishLoading(&node, 1, true));
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
}

TEST(LoadWorkerTest, CancelingFirstContextDoesNotFailJoinedContext) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadAsyncContext>     first_context  = std::make_shared<LoadAsyncContext>(1);
    const std::shared_ptr<LoadAsyncContext>     joined_context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {4}, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, joined_context).has_value());
    ASSERT_TRUE(first_context->requestCancel());
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));

    EXPECT_TRUE(first_context->done());
    EXPECT_FALSE(first_context->success());
    EXPECT_TRUE(joined_context->done());
    EXPECT_TRUE(joined_context->success());
}

TEST(LoadWorkerTest, CancelingJoinedContextDoesNotFailFirstContext) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadAsyncContext>     first_context  = std::make_shared<LoadAsyncContext>(1);
    const std::shared_ptr<LoadAsyncContext>     joined_context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {5}, first_context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, joined_context).has_value());
    ASSERT_TRUE(joined_context->requestCancel());
    EXPECT_TRUE(worker.finishLoading(&node, 0, true));

    EXPECT_TRUE(first_context->done());
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->done());
    EXPECT_FALSE(joined_context->success());
}

TEST(LoadWorkerTest, ContextAggregatesMultipleLoadingRecords) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadAsyncContext>     context = std::make_shared<LoadAsyncContext>(2);

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

TEST(LoadWorkerTest, ContextWaitsForJoinedAndNewLoadingRecords) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadAsyncContext>     first_context = std::make_shared<LoadAsyncContext>(1);
    const std::shared_ptr<LoadAsyncContext>     mixed_context = std::make_shared<LoadAsyncContext>(2);

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

TEST(LoadWorkerTest, EraseLoadingOnlyRemovesSelectedContext) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::vector<BlockIdxType>             target_blocks{7};
    const std::shared_ptr<LoadAsyncContext>     first_context  = std::make_shared<LoadAsyncContext>(1);
    const std::shared_ptr<LoadAsyncContext>     second_context = std::make_shared<LoadAsyncContext>(1);

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

TEST(LoadWorkerTest, EraseLastLoadingContextRemovesRecord) {
    LoadWorker                                  worker;
    TreeNode                                    node;
    const std::shared_ptr<LoadAsyncContext>     context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(worker.startLoading(&node, 0, {8}, context));
    ASSERT_TRUE(worker.joinLoading(&node, 0, context).has_value());
    EXPECT_TRUE(worker.eraseLoadingForOneContext(&node, 0, context));
    EXPECT_FALSE(worker.joinLoading(&node, 0, context).has_value());
    EXPECT_FALSE(worker.finishLoading(&node, 0, true));
    EXPECT_TRUE(context->onTaskFail());
}

}  // namespace
}  // namespace rtp_llm
