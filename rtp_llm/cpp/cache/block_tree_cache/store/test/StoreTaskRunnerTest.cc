#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;
using namespace block_tree_cache_test;

TEST(StoreTaskRunnerTest, PrepareTaskCreatesHostTransferAndTemporaryHolds) {
    auto policy                                            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                             = true;
    const GroupBase                            group       = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology    = makeTestTopology({group});
    DeviceBlockPoolPtr                         device_pool = makeTestDevicePool({{16, 0}}, 2, "store_task_runner");
    std::shared_ptr<HostBlockPool>             host_pool   = block_transfer_engine_test::makeHostPool(16, 1, false);
    GroupSetPtr                                group_set = makeTestGroupSet(0, topology, {0}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr>             group_sets{group_set};
    StoreTaskRunner                            runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1, BlockRefType::REQUEST);
    ASSERT_EQ(source_holder.size(), 1u);
    const BlockIdxType                         source_block = source_holder[0][0];
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    StoreTaskRunner::Task task;
    task.target_tier = Tier::HOST;
    task.cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(task, resources));

    ASSERT_EQ(task.descriptors.size(), 1u);
    EXPECT_EQ(task.descriptors[0].path_index, 0u);
    EXPECT_EQ(task.descriptors[0].group_set_id, 0u);
    EXPECT_EQ(task.descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(task.descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(task.descriptors[0].source_blocks, (BlockIndicesType{source_block}));
    ASSERT_EQ(task.descriptors[0].target_blocks.size(), 1u);
    EXPECT_NE(task.descriptors[0].target_blocks[0], NULL_BLOCK_IDX);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockRefType::STORE), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::STORE), 1u);

    runner.releaseTaskResources(task);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockRefType::STORE), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::STORE), 0u);

    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}},
                                 BlockRefType::REQUEST);
}

TEST(StoreTaskRunnerTest, PrepareTaskRecordsPathIndexInTransferDescriptors) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const GroupBase                            group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool = makeTestDevicePool({{16, 0}}, 4, "store_task_runner_path_index");
    std::shared_ptr<HostBlockPool> host_pool   = block_transfer_engine_test::makeHostPool(16, 2, false);
    GroupSetPtr                    group_set   = makeTestGroupSet(0, topology, {0}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 2, BlockRefType::REQUEST);
    ASSERT_EQ(source_holder.size(), 2u);
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];
    resources[1][0].device_blocks = source_holder[1];

    StoreTaskRunner::Task task;
    task.target_tier = Tier::HOST;
    task.cache_keys  = {100, 200};
    ASSERT_TRUE(runner.prepareTask(task, resources));

    ASSERT_EQ(task.descriptors.size(), 2u);
    EXPECT_EQ(task.descriptors[0].path_index, 0u);
    EXPECT_EQ(task.descriptors[1].path_index, 1u);

    runner.releaseTaskResources(task);
    unreferenceDeviceBlocksForTest(*group_set, source_holder, BlockRefType::REQUEST);
}

TEST(StoreTaskRunnerTest, ReleaseTaskResourcesDropsTemporaryHolds) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const GroupBase                            group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool          = makeTestDevicePool({{16, 0}}, 2, "store_task_runner_release");
    std::shared_ptr<HostBlockPool> host_pool            = block_transfer_engine_test::makeHostPool(16, 1, false);
    GroupSetPtr                    group_set            = makeTestGroupSet(0, topology, {0}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1, BlockRefType::REQUEST);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    StoreTaskRunner::Task task;
    task.target_tier = Tier::HOST;
    task.cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(task, resources));
    runner.releaseTaskResources(task);

    EXPECT_EQ(device_pool->referencedBlocksNum(BlockRefType::STORE), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockRefType::STORE), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(device_pool->refCount(source_holder[0][0]), 1u);

    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}},
                                 BlockRefType::REQUEST);
}

class RecordingStoreTransferEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingStoreTransferEngine(const std::vector<GroupSetPtr>& group_sets):
        PerRankBlockTransferEngine(group_sets) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches.push_back(descriptors);
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected copy failure"));
    }

    std::vector<std::vector<TransferDescriptor>> batches;
};

TEST(StoreTaskRunnerTest, RunTransferReturnsDispatcherFailure) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const GroupBase                            group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool = makeTestDevicePool({{16, 0}}, 2, "store_task_runner_transfer");
    std::shared_ptr<HostBlockPool> host_pool   = block_transfer_engine_test::makeHostPool(16, 1, false);
    GroupSetPtr                    group_set   = makeTestGroupSet(0, topology, {0}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1, BlockRefType::REQUEST);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    StoreTaskRunner::Task task;
    task.target_tier = Tier::HOST;
    task.cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(task, resources));

    auto engine = std::make_shared<ControlledPerRankBlockTransferEngine>(group_sets, TransferCopyAction::Fail);
    BlockTransferDispatcher       dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    EXPECT_FALSE(runner.runTransfer(task, dispatcher, metrics_reporter, 10, 20));

    runner.releaseTaskResources(task);
    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}},
                                 BlockRefType::REQUEST);
}

TEST(StoreTaskRunnerTest, TransferSubmissionFollowsTargetTier) {
    const GroupBase group = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL));
    const auto      topology = makeTestTopology({group});
    const std::vector<GroupSetPtr> group_sets{
        makeTestGroupSet(0,
                         topology,
                         {0},
                         {makeTestDevicePool({{16, 0}}, 2, "store_task_runner_submission_0")}),
        makeTestGroupSet(1,
                         topology,
                         {0},
                         {makeTestDevicePool({{16, 0}}, 2, "store_task_runner_submission_1")})};
    StoreTaskRunner                runner(group_sets);
    BlockTreeCacheMetricsReporter metrics_reporter;

    StoreTaskRunner::Task host_task;
    host_task.target_tier = Tier::HOST;
    host_task.descriptors = {TransferDescriptor::deviceToHost(0, {1}, 1),
                             TransferDescriptor::deviceToHost(1, {2}, 2)};
    auto host_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher host_dispatcher(host_engine);
    EXPECT_FALSE(runner.runTransfer(host_task, host_dispatcher, metrics_reporter, 10, 20));
    ASSERT_EQ(host_engine->batches.size(), 1u);
    ASSERT_EQ(host_engine->batches.front().size(), 2u);
    EXPECT_EQ(host_engine->batches.front()[0].group_set_id, 0u);
    EXPECT_EQ(host_engine->batches.front()[1].group_set_id, 1u);
    EXPECT_EQ(host_engine->batches.front()[0].target_tier, Tier::HOST);
    EXPECT_EQ(host_engine->batches.front()[1].target_tier, Tier::HOST);

    StoreTaskRunner::Task disk_task;
    disk_task.target_tier = Tier::DISK;
    disk_task.descriptors = {TransferDescriptor::deviceToDisk(0, {1}, 1),
                             TransferDescriptor::deviceToDisk(1, {2}, 2)};
    auto disk_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher disk_dispatcher(disk_engine);
    EXPECT_FALSE(runner.runTransfer(disk_task, disk_dispatcher, metrics_reporter, 10, 20));
    ASSERT_EQ(disk_engine->batches.size(), 2u);
    for (size_t batch_index = 0; batch_index < disk_engine->batches.size(); ++batch_index) {
        ASSERT_EQ(disk_engine->batches[batch_index].size(), 1u);
        EXPECT_EQ(disk_engine->batches[batch_index].front().group_set_id, batch_index);
        EXPECT_EQ(disk_engine->batches[batch_index].front().target_tier, Tier::DISK);
    }
}

}  // namespace
}  // namespace rtp_llm
