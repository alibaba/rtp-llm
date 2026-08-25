#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
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

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    const BlockIdxType                         source_block = source_holder[0][0];
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>();
    task->target_tier = Tier::HOST;
    task->cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(*task, resources));

    ASSERT_EQ(task->descriptors.size(), 1u);
    EXPECT_EQ(task->descriptors[0].path_index, 0u);
    EXPECT_EQ(task->descriptors[0].group_set_id, 0u);
    EXPECT_EQ(task->descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(task->descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(task->descriptors[0].source_blocks, (BlockIndicesType{source_block}));
    ASSERT_EQ(task->descriptors[0].target_blocks.size(), 1u);
    EXPECT_NE(task->descriptors[0].target_blocks[0], NULL_BLOCK_IDX);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_EQ(device_pool->refCount(source_block), 2u);
    EXPECT_EQ(device_pool->treeRefCount(source_block), 1u);
    EXPECT_EQ(host_pool->treeRefCount(task->descriptors[0].target_blocks[0]), 1u);

    runner.releaseTaskResources(*task);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);

    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
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

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 2);
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
    unreferenceDeviceBlocksForTest(*group_set, source_holder);
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

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>();
    task->target_tier = Tier::HOST;
    task->cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(*task, resources));
    runner.releaseTaskResources(*task);

    EXPECT_EQ(device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(device_pool->refCount(source_holder[0][0]), 1u);

    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
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

class PendingStoreTransferEngine final: public PerRankBlockTransferEngine {
public:
    PendingStoreTransferEngine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches.push_back(descriptors);
        auto context = std::make_shared<TransferBatchAsyncContext>();
        contexts.push_back(context);
        return context;
    }

    std::vector<std::vector<TransferDescriptor>>            batches;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
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

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>();
    task->target_tier = Tier::HOST;
    task->cache_keys  = {100};
    ASSERT_TRUE(runner.prepareTask(*task, resources));

    auto engine = std::make_shared<ControlledPerRankBlockTransferEngine>(group_sets, TransferCopyAction::Fail);
    BlockTransferDispatcher       dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    std::optional<ErrorInfo> result;
    runner.runTransfer(task, dispatcher, metrics_reporter, 10, 20,
                       [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());

    runner.releaseTaskResources(*task);
    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
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

    auto host_task = std::make_shared<StoreTaskRunner::Task>();
    host_task->target_tier = Tier::HOST;
    host_task->descriptors = {TransferDescriptor::deviceToHost(0, {1}, 1),
                              TransferDescriptor::deviceToHost(1, {2}, 2)};
    auto host_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher host_dispatcher(host_engine);
    std::optional<ErrorInfo> host_result;
    runner.runTransfer(host_task, host_dispatcher, metrics_reporter, 10, 20,
                       [&](ErrorInfo error) { host_result.emplace(std::move(error)); });
    ASSERT_TRUE(host_result.has_value());
    EXPECT_FALSE(host_result->ok());
    ASSERT_EQ(host_engine->batches.size(), 2u);
    for (size_t batch_index = 0; batch_index < host_engine->batches.size(); ++batch_index) {
        ASSERT_EQ(host_engine->batches[batch_index].size(), 1u);
        EXPECT_EQ(host_engine->batches[batch_index].front().group_set_id, batch_index);
        EXPECT_EQ(host_engine->batches[batch_index].front().target_tier, Tier::HOST);
    }

    auto disk_task = std::make_shared<StoreTaskRunner::Task>();
    disk_task->target_tier = Tier::DISK;
    disk_task->descriptors = {TransferDescriptor::deviceToDisk(0, {1}, 1),
                              TransferDescriptor::deviceToDisk(1, {2}, 2)};
    auto disk_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher disk_dispatcher(disk_engine);
    std::optional<ErrorInfo> disk_result;
    runner.runTransfer(disk_task, disk_dispatcher, metrics_reporter, 10, 20,
                       [&](ErrorInfo error) { disk_result.emplace(std::move(error)); });
    ASSERT_TRUE(disk_result.has_value());
    EXPECT_FALSE(disk_result->ok());
    ASSERT_EQ(disk_engine->batches.size(), 2u);
    for (size_t batch_index = 0; batch_index < disk_engine->batches.size(); ++batch_index) {
        ASSERT_EQ(disk_engine->batches[batch_index].size(), 1u);
        EXPECT_EQ(disk_engine->batches[batch_index].front().group_set_id, batch_index);
        EXPECT_EQ(disk_engine->batches[batch_index].front().target_tier, Tier::DISK);
    }
}

TEST(StoreTaskRunnerTest, PendingTransferDoesNotRetainOuterWorker) {
    const std::vector<GroupSetPtr> group_sets;
    StoreTaskRunner runner(group_sets);
    auto engine = std::make_shared<PendingStoreTransferEngine>();
    BlockTransferDispatcher dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    BlockTreeTaskPool outer_pool(1, 8, "AsyncStoreOuter");
    ASSERT_TRUE(outer_pool.start());

    auto first = std::make_shared<StoreTaskRunner::Task>();
    auto second = std::make_shared<StoreTaskRunner::Task>();
    first->target_tier = Tier::HOST;
    second->target_tier = Tier::HOST;
    first->descriptors = {TransferDescriptor::deviceToHost(0, {1}, 1)};
    second->descriptors = {TransferDescriptor::deviceToHost(0, {2}, 2)};
    std::atomic<size_t> started{0};
    std::atomic<size_t> settled{0};
    const auto submit_task = [&](const std::shared_ptr<StoreTaskRunner::Task>& task) {
        return outer_pool.submit([&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, 10, 20, [&](ErrorInfo) {
                EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); }));
            });
            started.fetch_add(1);
        });
    };

    ASSERT_TRUE(submit_task(first));
    ASSERT_TRUE(submit_task(second));
    for (size_t attempt = 0; attempt < 100 && started.load() != 2; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(started.load(), 2u);
    ASSERT_EQ(engine->contexts.size(), 2u);

    engine->contexts[0]->complete(ErrorInfo::OkStatus());
    engine->contexts[1]->complete(ErrorInfo::OkStatus());
    outer_pool.waitForIdle();
    EXPECT_EQ(settled.load(), 2u);
}

}  // namespace
}  // namespace rtp_llm
