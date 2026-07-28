#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(LoadJoinRegistryTest, FinishNotifiesJoinedContext) {
    LoadJoinRegistry                registry;
    TreeNode                        node;
    const std::vector<BlockIdxType> target_blocks{1, 2};
    const auto                      first_context  = std::make_shared<LoadAsyncContext>(1);
    const auto                      joined_context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(registry.start(&node, 0, target_blocks, first_context));
    const auto joined_blocks = registry.join(&node, 0, joined_context);
    ASSERT_TRUE(joined_blocks.has_value());
    EXPECT_EQ(joined_blocks.value(), target_blocks);
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->success());
    EXPECT_FALSE(registry.finish(&node, 0, true));
}

TEST(LoadJoinRegistryTest, DuplicateJoinOnlyCompletesOnce) {
    LoadJoinRegistry registry;
    TreeNode         node;
    const auto       context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(registry.start(&node, 1, {3}, context));
    ASSERT_TRUE(registry.join(&node, 1, context).has_value());
    ASSERT_TRUE(registry.join(&node, 1, context).has_value());
    EXPECT_TRUE(registry.finish(&node, 1, true));
    EXPECT_TRUE(context->success());
}

TEST(LoadJoinRegistryTest, CancellationIsPerContext) {
    LoadJoinRegistry registry;
    TreeNode         node;
    const auto       first_context  = std::make_shared<LoadAsyncContext>(1);
    const auto       joined_context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(registry.start(&node, 0, {4}, first_context));
    ASSERT_TRUE(registry.join(&node, 0, joined_context).has_value());
    ASSERT_TRUE(first_context->requestCancel());
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_FALSE(first_context->success());
    EXPECT_TRUE(joined_context->success());
}

TEST(LoadJoinRegistryTest, ContextAggregatesMultipleRecords) {
    LoadJoinRegistry registry;
    TreeNode         node;
    const auto       context = std::make_shared<LoadAsyncContext>(2);

    ASSERT_TRUE(registry.start(&node, 0, {5}, context));
    ASSERT_TRUE(registry.start(&node, 1, {6}, context));
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(registry.finish(&node, 1, false));
    EXPECT_FALSE(context->success());
}

TEST(LoadJoinRegistryTest, EraseForContextPreservesOtherContexts) {
    LoadJoinRegistry registry;
    TreeNode         node;
    const auto       first_context  = std::make_shared<LoadAsyncContext>(1);
    const auto       second_context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(registry.start(&node, 0, {7}, first_context));
    ASSERT_TRUE(registry.join(&node, 0, second_context).has_value());
    EXPECT_TRUE(registry.eraseForContext(&node, 0, second_context));
    EXPECT_FALSE(registry.eraseForContext(&node, 0, second_context));
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_FALSE(second_context->done());
}

TEST(LoadJoinRegistryTest, EraseLastContextRemovesRecord) {
    LoadJoinRegistry registry;
    TreeNode         node;
    const auto       context = std::make_shared<LoadAsyncContext>(1);

    ASSERT_TRUE(registry.start(&node, 0, {8}, context));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, context));
    EXPECT_FALSE(registry.join(&node, 0, context).has_value());
    EXPECT_FALSE(registry.finish(&node, 0, true));
}

}  // namespace
}  // namespace rtp_llm
