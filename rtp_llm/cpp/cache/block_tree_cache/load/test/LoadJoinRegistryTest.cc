#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class LoadJoinRegistryTest: public ::testing::Test {
protected:
    void SetUp() override {
        coordinator_ = std::make_shared<LoadContextCoordinator>(LoadContextCoordinator::CommitCallback{},
                                                                LoadContextCoordinator::AbortCallback{});
    }

    void TearDown() override {
        coordinator_->shutdown();
    }

    std::shared_ptr<LoadAsyncContext> makeContext(size_t pending_transfer_count) {
        TransferDescriptor desc;
        desc.source_tier = Tier::HOST;
        desc.target_tier = Tier::DEVICE;
        return coordinator_->create({desc}, {false}, 1, pending_transfer_count);
    }

    std::shared_ptr<LoadContextCoordinator> coordinator_;
};

TEST_F(LoadJoinRegistryTest, FinishNotifiesJoinedContext) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::vector<BlockIdxType>         target_blocks{1, 2};
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1);

    ASSERT_TRUE(registry.start(&node, 0, target_blocks, first_context));
    std::vector<BlockIdxType> joined_blocks;
    ASSERT_TRUE(registry.join(&node, 0, joined_context, joined_blocks));
    EXPECT_EQ(joined_blocks, target_blocks);
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->success());
    EXPECT_FALSE(registry.finish(&node, 0, true));
}

TEST_F(LoadJoinRegistryTest, DuplicateJoinOnlyCompletesOnce) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);

    ASSERT_TRUE(registry.start(&node, 1, {3}, context));
    std::vector<BlockIdxType> target_blocks;
    ASSERT_TRUE(registry.join(&node, 1, context, target_blocks));
    ASSERT_TRUE(registry.join(&node, 1, context, target_blocks));
    EXPECT_TRUE(registry.finish(&node, 1, true));
    EXPECT_TRUE(context->success());
}

TEST_F(LoadJoinRegistryTest, CancellationIsPerContext) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1);

    ASSERT_TRUE(registry.start(&node, 0, {4}, first_context));
    std::vector<BlockIdxType> target_blocks;
    ASSERT_TRUE(registry.join(&node, 0, joined_context, target_blocks));
    ASSERT_TRUE(first_context->requestCancel());
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_FALSE(first_context->success());
    EXPECT_TRUE(joined_context->success());
}

TEST_F(LoadJoinRegistryTest, ContextAggregatesMultipleRecords) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> context = makeContext(2);

    ASSERT_TRUE(registry.start(&node, 0, {5}, context));
    ASSERT_TRUE(registry.start(&node, 1, {6}, context));
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(registry.finish(&node, 1, false));
    EXPECT_FALSE(context->success());
}

TEST_F(LoadJoinRegistryTest, EraseForContextPreservesOtherContexts) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> second_context = makeContext(1);

    ASSERT_TRUE(registry.start(&node, 0, {7}, first_context));
    std::vector<BlockIdxType> target_blocks;
    ASSERT_TRUE(registry.join(&node, 0, second_context, target_blocks));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, second_context->contextId()));
    EXPECT_FALSE(registry.eraseForContext(&node, 0, second_context->contextId()));
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_FALSE(second_context->done());
}

TEST_F(LoadJoinRegistryTest, ExpiredJoinedContextIsNotKeptAlive) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context = makeContext(1);
    std::weak_ptr<LoadAsyncContext>         weak_joined_context;

    ASSERT_TRUE(registry.start(&node, 0, {8}, first_context));
    {
        const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1);
        weak_joined_context                                    = joined_context;
        std::vector<BlockIdxType> target_blocks;
        ASSERT_TRUE(registry.join(&node, 0, joined_context, target_blocks));
    }

    EXPECT_TRUE(weak_joined_context.expired());
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
}

TEST_F(LoadJoinRegistryTest, EraseExpiredContextById) {
    LoadJoinRegistry                registry;
    TreeNode                        node;
    std::weak_ptr<LoadAsyncContext> weak_context;
    uint64_t                        context_id = 0;

    {
        const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
        weak_context                                    = context;
        context_id                                      = context->contextId();
        ASSERT_TRUE(registry.start(&node, 0, {9}, context));
    }

    ASSERT_TRUE(weak_context.expired());
    EXPECT_TRUE(registry.eraseForContext(&node, 0, context_id));
    std::vector<BlockIdxType> target_blocks;
    EXPECT_FALSE(registry.getTargetBlocks(&node, 0, target_blocks));
}

TEST_F(LoadJoinRegistryTest, EraseLastContextRemovesRecord) {
    LoadJoinRegistry                        registry;
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);

    ASSERT_TRUE(registry.start(&node, 0, {10}, context));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, context->contextId()));
    std::vector<BlockIdxType> target_blocks;
    EXPECT_FALSE(registry.join(&node, 0, context, target_blocks));
    EXPECT_FALSE(registry.finish(&node, 0, true));
}

}  // namespace
}  // namespace rtp_llm
