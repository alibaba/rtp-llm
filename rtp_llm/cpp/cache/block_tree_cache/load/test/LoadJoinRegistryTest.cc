#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;

class LoadJoinRegistryTest: public ::testing::Test {
protected:
    void SetUp() override {
        const GroupBase group = makeTestGroupBase();
        device_pool_ =
            makeTestDevicePool({{group.kv_block_stride_bytes, group.kv_scale_stride_bytes}}, 16, "load_join_registry");
        const GroupSetPtr group_set = makeTestGroupSet(0, makeTestTopology({group}), {0}, {device_pool_});
        tree_                       = std::make_unique<BlockTree>(std::vector<GroupSetPtr>{group_set});
        target_blocks_              = device_pool_->malloc(10).value();
        coordinator_ = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    }

    void TearDown() override {
        coordinator_->shutdown();
    }

    std::shared_ptr<LoadAsyncContext> makeContext(size_t    transfer_count,
                                                  TreeNode* node                    = nullptr,
                                                  size_t    group_set_id            = 0,
                                                  bool      joined                  = false,
                                                  bool      install_target_in_cache = true) {
        std::vector<TransferDescriptor> load_descs(transfer_count);
        for (TransferDescriptor& desc : load_descs) {
            desc.node                    = node;
            desc.group_set_id            = group_set_id;
            desc.source_tier             = Tier::HOST;
            desc.target_tier             = Tier::DEVICE;
            desc.install_target_in_cache = install_target_in_cache;
        }
        auto context = coordinator_->create(load_descs, std::vector<bool>(load_descs.size(), joined), 1);
        EXPECT_NE(context, nullptr);
        EXPECT_TRUE(coordinator_->registerContext(context));
        EXPECT_TRUE(context->commit());
        return context;
    }

    DeviceBlockPoolPtr                      device_pool_;
    std::unique_ptr<BlockTree>              tree_;
    std::vector<BlockIdxType>               target_blocks_;
    std::shared_ptr<LoadContextCoordinator> coordinator_;
};

TEST_F(LoadJoinRegistryTest, JoinSkipsNonJoinedDescriptors) {
    LoadJoinRegistry                        registry(tree_.get());
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);

    EXPECT_TRUE(registry.join(context));
}

TEST_F(LoadJoinRegistryTest, FinishNotifiesJoinedContext) {
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::vector<BlockIdxType>         target_blocks{target_blocks_[0]};
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1, &node, 0, true);

    ASSERT_TRUE(registry.start(&node, 0, target_blocks, first_context));
    device_pool_->incTreeRef(target_blocks[0], BlockTreeRefType::LOAD);
    EXPECT_EQ(device_pool_->refCount(target_blocks[0]), 1u);
    EXPECT_EQ(device_pool_->treeRefCount(target_blocks[0]), 1u);
    ASSERT_TRUE(registry.join(joined_context));
    EXPECT_EQ(joined_context->loadDescs()[0].target_blocks, target_blocks);
    EXPECT_EQ(device_pool_->refCount(target_blocks[0]), 2u);
    EXPECT_EQ(device_pool_->treeRefCount(target_blocks[0]), 1u);
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(joined_context->success());
    EXPECT_FALSE(registry.finish(&node, 0, true));
    device_pool_->decRef(target_blocks[0]);
    device_pool_->decTreeRef(target_blocks[0], BlockTreeRefType::LOAD);
}

TEST_F(LoadJoinRegistryTest, FailureIsPerContext) {
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1, &node, 0, true);

    ASSERT_TRUE(registry.start(&node, 0, {target_blocks_[3]}, first_context));
    ASSERT_TRUE(registry.join(joined_context));
    ASSERT_TRUE(first_context->onTaskFail());
    EXPECT_FALSE(registry.finish(&node, 0, true));
    EXPECT_FALSE(first_context->success());
    EXPECT_TRUE(joined_context->success());
}

TEST_F(LoadJoinRegistryTest, ContextAggregatesMultipleRecords) {
    LoadJoinRegistry                        registry(tree_.get());
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
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> second_context = makeContext(1, &node, 0, true);

    ASSERT_TRUE(registry.start(&node, 0, {target_blocks_[4]}, first_context));
    ASSERT_TRUE(registry.join(second_context));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, second_context->contextId()));
    EXPECT_FALSE(registry.eraseForContext(&node, 0, second_context->contextId()));
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
    EXPECT_FALSE(second_context->done());
}

TEST_F(LoadJoinRegistryTest, InstallDecisionAggregatesActiveContexts) {
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> host_only_leader =
        makeContext(1, nullptr, 0, /*joined=*/false, /*install_target_in_cache=*/false);
    const std::shared_ptr<LoadAsyncContext> device_joiner =
        makeContext(1, &node, 0, /*joined=*/true, /*install_target_in_cache=*/true);

    ASSERT_TRUE(registry.start(&node, 0, {target_blocks_[6]}, host_only_leader, /*install_target_in_cache=*/false));
    EXPECT_FALSE(registry.installTargetInCache(&node, 0));
    ASSERT_TRUE(registry.join(device_joiner));
    EXPECT_TRUE(registry.installTargetInCache(&node, 0));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, device_joiner->contextId()));
    EXPECT_FALSE(registry.installTargetInCache(&node, 0));
    EXPECT_TRUE(registry.finish(&node, 0, true));
}

TEST_F(LoadJoinRegistryTest, ExpiredJoinedContextIsNotKeptAlive) {
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> first_context = makeContext(1);
    std::weak_ptr<LoadAsyncContext>         weak_joined_context;

    ASSERT_TRUE(registry.start(&node, 0, {target_blocks_[5]}, first_context));
    {
        const std::shared_ptr<LoadAsyncContext> joined_context = makeContext(1, &node, 0, true);
        weak_joined_context                                    = joined_context;
        ASSERT_TRUE(registry.join(joined_context));
    }

    EXPECT_TRUE(weak_joined_context.expired());
    EXPECT_TRUE(registry.finish(&node, 0, true));
    EXPECT_TRUE(first_context->success());
}

TEST_F(LoadJoinRegistryTest, EraseExpiredContextById) {
    LoadJoinRegistry                registry(tree_.get());
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
    const std::shared_ptr<LoadAsyncContext> other_context = makeContext(1, &node, 0, true);
    EXPECT_FALSE(registry.join(other_context));
}

TEST_F(LoadJoinRegistryTest, EraseLastContextRemovesRecord) {
    LoadJoinRegistry                        registry(tree_.get());
    TreeNode                                node;
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1, &node, 0, true);

    ASSERT_TRUE(registry.start(&node, 0, {10}, context));
    EXPECT_TRUE(registry.eraseForContext(&node, 0, context->contextId()));
    EXPECT_FALSE(registry.join(context));
    EXPECT_FALSE(registry.finish(&node, 0, true));
}

}  // namespace
}  // namespace rtp_llm
