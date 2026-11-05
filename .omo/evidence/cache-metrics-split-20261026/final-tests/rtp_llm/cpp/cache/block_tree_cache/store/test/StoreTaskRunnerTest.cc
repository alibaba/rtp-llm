#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
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
    const auto                                 group       = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology    = makeTestTopology({group});
    DeviceBlockPoolPtr                         device_pool = makeTestDevicePool({{16, 0}}, 2, "store_task_runner");
    std::shared_ptr<HostBlockPool>             host_pool   = block_transfer_engine_test::makeHostPool(16, 1);
    GroupSetPtr group_set = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr>             group_sets{group_set};
    StoreTaskRunner                            runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    const BlockIdxType                         source_block = source_holder[0][0];
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{100}, std::chrono::seconds(30));
    ASSERT_TRUE(runner.prepareTask(*task, resources));

    ASSERT_EQ(task->descriptors().size(), 1u);
    EXPECT_EQ(task->descriptors()[0].path_index, 0u);
    EXPECT_EQ(task->descriptors()[0].group_set_id, 0u);
    EXPECT_EQ(task->descriptors()[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(task->descriptors()[0].target_tier, Tier::HOST);
    EXPECT_EQ(task->descriptors()[0].source_blocks, (BlockIndicesType{source_block}));
    ASSERT_EQ(task->descriptors()[0].target_blocks.size(), 1u);
    EXPECT_NE(task->descriptors()[0].target_blocks[0], NULL_BLOCK_IDX);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_EQ(device_pool->refCount(source_block), 2u);
    EXPECT_EQ(device_pool->treeRefCount(source_block), 1u);
    EXPECT_EQ(host_pool->treeRefCount(task->descriptors()[0].target_blocks[0]), 1u);

    runner.releaseTaskResources(*task);
    EXPECT_EQ(device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);

    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
}

TEST(StoreTaskRunnerTest, LocalOnlyGroupDoesNotBlockCompleteGroupLowerTierStore) {
    auto full_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_policy.enable_prefix_reuse = true;
    auto hca_policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
    hca_policy.sliding_window_size = 128;
    const auto topology = makeTestTopology({makeTestGroupBase(full_policy), makeTestGroupBase(hca_policy)});
    auto full_device = makeTestDevicePool({{16, 0}}, 2, "store_full_device");
    auto hca_device = makeTestDevicePool({{16, 0}}, 2, "store_hca_device");
    auto full_host = block_transfer_engine_test::makeHostPool(16, 1);
    auto full_disk = block_tree_cache_test::makeDiskPool(16, 1, std::make_unique<MemoryDiskBlockIO>());
    GroupSetPtr full = makeTestGroupSet(0, topology, {"group0"}, {full_device}, full_host, full_disk);
    GroupSetPtr hca = std::make_shared<SWAGroupSet>(128,
                                                   topology->group("group1").seqSizePerBlock(),
                                                   std::vector<DeviceBlockPoolPtr>{hca_device},
                                                   nullptr,
                                                   nullptr);
    hca->initialize(1, topology, {"group1"}, {}, /*local_only=*/true);
    const std::vector<GroupSetPtr> group_sets{full, hca};
    StoreTaskRunner runner(group_sets);

    const MultiNodeBlocks full_source = allocateDeviceBlocksForTest(*full, 1);
    const MultiNodeBlocks hca_source = allocateDeviceBlocksForTest(*hca, 1);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = full_source[0];
    resources[0][1].device_blocks = hca_source[0];

    for (const Tier target : {Tier::HOST, Tier::DISK}) {
        StoreTaskRunner::Task task(target, CacheKeysType{100}, std::chrono::seconds(30));
        ASSERT_TRUE(runner.prepareTask(task, resources));
        ASSERT_EQ(task.descriptors().size(), 1u);
        EXPECT_EQ(task.descriptors()[0].group_set_id, 0u);
        EXPECT_EQ(task.descriptors()[0].target_tier, target);
        EXPECT_EQ(hca_device->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        EXPECT_EQ(full_device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
        runner.releaseTaskResources(task);
        EXPECT_EQ(full_device->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    }
    unreferenceDeviceBlocksForTest(*full, full_source);
    unreferenceDeviceBlocksForTest(*hca, hca_source);
}

TEST(StoreTaskRunnerTest, PrepareTaskRecordsPathIndexInTransferDescriptors) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const auto                                 group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool = makeTestDevicePool({{16, 0}}, 4, "store_task_runner_path_index");
    std::shared_ptr<HostBlockPool> host_pool   = block_transfer_engine_test::makeHostPool(16, 2);
    GroupSetPtr                    group_set = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 2);
    ASSERT_EQ(source_holder.size(), 2u);
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];
    resources[1][0].device_blocks = source_holder[1];

    StoreTaskRunner::Task task(Tier::HOST, CacheKeysType{100, 200}, std::chrono::seconds(30));
    ASSERT_TRUE(runner.prepareTask(task, resources));

    ASSERT_EQ(task.descriptors().size(), 2u);
    EXPECT_EQ(task.descriptors()[0].path_index, 0u);
    EXPECT_EQ(task.descriptors()[1].path_index, 1u);

    runner.releaseTaskResources(task);
    unreferenceDeviceBlocksForTest(*group_set, source_holder);
}

TEST(StoreTaskRunnerTest, ReleaseTaskResourcesDropsTemporaryHolds) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const auto                                 group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool          = makeTestDevicePool({{16, 0}}, 2, "store_task_runner_release");
    std::shared_ptr<HostBlockPool> host_pool            = block_transfer_engine_test::makeHostPool(16, 1);
    GroupSetPtr                    group_set = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{100}, std::chrono::seconds(30));
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

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        batches.push_back(descriptors);
        return std::make_shared<CompletedTransferAsyncContext>(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected copy failure"));
    }

    std::vector<std::vector<TransferDescriptor>> batches;
};

class PendingStoreTransferEngine final: public PerRankBlockTransferEngine {
public:
    PendingStoreTransferEngine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        batches.push_back(descriptors);
        auto context = std::make_shared<TransferBatchAsyncContext>();
        contexts.push_back(context);
        return context;
    }

    std::vector<std::vector<TransferDescriptor>>            batches;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
};

TEST(StoreTaskRunnerTest, DiskFailureWaitsForEverySubmittedBatch) {
    const auto                     group    = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL));
    const auto                     topology = makeTestTopology({group});
    const std::vector<GroupSetPtr> group_sets{makeTestGroupSet(0, topology, {"group0"}, {}),
                                              makeTestGroupSet(1, topology, {"group0"}, {}, nullptr, nullptr, true)};
    StoreTaskRunner                runner(group_sets);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};

    for (const bool mixed_parent : {false, true}) {
        SCOPED_TRACE(mixed_parent ? "mixed parent" : "standalone ordinary parent");
        auto                    engine = std::make_shared<PendingStoreTransferEngine>();
        BlockTransferDispatcher dispatcher(engine);
        auto task = std::make_shared<StoreTaskRunner::Task>(Tier::DISK, CacheKeysType{}, std::chrono::seconds(30));
        task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(0, {1}, 1));
        task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(0, {2}, 2));
        if (mixed_parent) {
            task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(1, {3}, 3));
        }
        std::optional<ErrorInfo> result;
        runner.runTransfer(
            task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
        EXPECT_EQ(engine->batches.size(), mixed_parent ? 3u : 2u);
        for (const auto& batch : engine->batches) {
            EXPECT_EQ(batch.size(), 1u);
        }
        EXPECT_FALSE(result.has_value());
        // A failure does not release the task while another batch is writing.
        if (mixed_parent && !engine->contexts.empty()) {
            engine->contexts.back()->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected copy failure"));
            EXPECT_FALSE(result.has_value());
        }
        for (const auto& context : engine->contexts) {
            context->complete(ErrorInfo::OkStatus());
        }
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->code(), mixed_parent ? ErrorCode::EXECUTION_EXCEPTION : ErrorCode::NONE_ERROR);
    }
}

TEST(StoreTaskRunnerTest, SplitHostAndDiskBatchesReportOneCrcErrorAfterAllCompletions) {
    for (const auto target : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(target));
        const std::vector<GroupSetPtr> groups;
        StoreTaskRunner                runner(groups);
        auto metrics_reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
        BlockTreeCacheMetricsReporter metrics(metrics_reporter);
        auto                          engine = std::make_shared<PendingStoreTransferEngine>();
        BlockTransferDispatcher       dispatcher(engine, nullptr, 1);
        auto task = std::make_shared<StoreTaskRunner::Task>(target, CacheKeysType{}, std::chrono::seconds(30));
        for (int block = 1; block <= 3; ++block) {
            task->transfer_task.addDescriptor(target == Tier::HOST ?
                                                  TransferDescriptor::deviceToHost(0, {block}, block) :
                                                  TransferDescriptor::deviceToDisk(0, {block}, block));
        }
        size_t callbacks = 0;
        runner.runTransfer(task, dispatcher, metrics, [&](ErrorInfo error) {
            EXPECT_FALSE(error.ok());
            ++callbacks;
        });
        if (engine->contexts.size() != 3u) {
            for (const auto& context : engine->contexts) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "finish failed test"));
            }
            FAIL() << "expected three batches, got " << engine->contexts.size();
        }
        auto* metric =
            metrics_reporter->getMetricsGroup<RtpLLMCacheTransferMetrics>()->memory_cache_copy_error_qps_metric;
        engine->contexts[1]->complete(
            {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC compute failed"), CacheCopyError::CRC_COMPUTE_FAILED});
        engine->contexts[0]->complete(
            {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RPC failed"), CacheCopyError::RPC_FAILED});
        EXPECT_EQ(callbacks, 0u);
        EXPECT_EQ(metric->metric_data_->Size(), 0u);
        engine->contexts[2]->complete(ErrorInfo::OkStatus());
        EXPECT_EQ(callbacks, 1u);
        EXPECT_EQ(metric->metric_data_->Size(), 1u);
        kmonitor::MetricsTags tags("copy_direction", "FROM_GPU");
        tags.AddTag("error_type", "CRC_COMPUTE_FAILED");
        auto* series = metric->DeclareMetric(&tags);
        ASSERT_NE(series, nullptr);
        kmonitor::MetricsRecord snapshot(nullptr, nullptr, 0);
        series->Snapshot(&snapshot, 1000);
        EXPECT_TRUE(metric->UndeclareMetric(series));
        ASSERT_EQ(snapshot.Values().size(), 1u);
        EXPECT_DOUBLE_EQ(std::stod(snapshot.Values().front()->Value()), 1);
    }
}

TEST(StoreTaskRunnerTest, RunTransferReturnsDispatcherFailure) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const auto                                 group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool = makeTestDevicePool({{16, 0}}, 2, "store_task_runner_transfer");
    std::shared_ptr<HostBlockPool> host_pool   = block_transfer_engine_test::makeHostPool(16, 1);
    GroupSetPtr                    group_set = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

    MultiNodeBlocks source_holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(source_holder.size(), 1u);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = source_holder[0];

    auto task = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{100}, std::chrono::seconds(30));
    ASSERT_TRUE(runner.prepareTask(*task, resources));

    auto engine = std::make_shared<ControlledPerRankBlockTransferEngine>(group_sets, TransferCopyAction::Fail);
    BlockTransferDispatcher       dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    std::optional<ErrorInfo>      result;
    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());

    runner.releaseTaskResources(*task);
    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
}

TEST(StoreTaskRunnerTest, TaskPhasePreventsDuplicateSettlementSubmission) {
    StoreTaskRunner::Task task(Tier::HOST, CacheKeysType{100}, std::chrono::seconds(30));
    EXPECT_TRUE(task.markTransferring());
    EXPECT_FALSE(task.markTransferring());
    EXPECT_TRUE(task.markFinished());
    EXPECT_FALSE(task.markFinished());

    StoreTaskRunner::Task timed_out_task(Tier::HOST, CacheKeysType{100}, std::chrono::seconds(30));
    EXPECT_TRUE(timed_out_task.markFinished());
    EXPECT_FALSE(timed_out_task.markTransferring());
}

TEST(StoreTaskRunnerTest, RepeatedRunTransferCannotSettleSameTaskTwice) {
    const std::vector<GroupSetPtr> group_sets;
    StoreTaskRunner                runner(group_sets);
    auto                           engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};

    auto   task = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{}, std::chrono::seconds(30));
    task->transfer_task = TransferTask({TransferDescriptor::deviceToHost(0, {1}, 1)}, std::chrono::seconds(30));
    size_t callback_count = 0;
    const auto callback   = [&](ErrorInfo error) {
        ++callback_count;
        EXPECT_FALSE(error.ok());
    };

    runner.runTransfer(task, dispatcher, metrics_reporter, callback);
    runner.runTransfer(task, dispatcher, metrics_reporter, callback);
    EXPECT_TRUE(task->markFinished());
    runner.runTransfer(task, dispatcher, metrics_reporter, callback);

    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(engine->batches.size(), 1u);
    EXPECT_EQ(task->phase.load(), StoreTaskRunner::Task::Phase::FINISHED);
}

TEST(StoreTaskRunnerTest, TransferSubmissionFollowsTargetTier) {
    const auto                     group    = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL));
    const auto                     topology = makeTestTopology({group});
    const std::vector<GroupSetPtr> group_sets{
        makeTestGroupSet(0, topology, {"group0"}, {makeTestDevicePool({{16, 0}}, 2, "store_task_runner_submission_0")}),
        makeTestGroupSet(
            1, topology, {"group0"}, {makeTestDevicePool({{16, 0}}, 2, "store_task_runner_submission_1")})};
    StoreTaskRunner               runner(group_sets);
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};

    auto host_task = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{}, std::chrono::seconds(30));
    host_task->transfer_task =
        TransferTask({TransferDescriptor::deviceToHost(0, {1}, 1), TransferDescriptor::deviceToHost(1, {2}, 2)},
                     std::chrono::seconds(30));
    auto                     host_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher  host_dispatcher(host_engine);
    std::optional<ErrorInfo> host_result;
    runner.runTransfer(
        host_task, host_dispatcher, metrics_reporter, [&](ErrorInfo error) { host_result.emplace(std::move(error)); });
    ASSERT_TRUE(host_result.has_value());
    EXPECT_FALSE(host_result->ok());
    ASSERT_EQ(host_engine->batches.size(), 2u);
    for (size_t batch_index = 0; batch_index < host_engine->batches.size(); ++batch_index) {
        ASSERT_EQ(host_engine->batches[batch_index].size(), 1u);
        EXPECT_EQ(host_engine->batches[batch_index].front().group_set_id, batch_index);
        EXPECT_EQ(host_engine->batches[batch_index].front().target_tier, Tier::HOST);
    }

    auto disk_task = std::make_shared<StoreTaskRunner::Task>(Tier::DISK, CacheKeysType{}, std::chrono::seconds(30));
    disk_task->transfer_task =
        TransferTask({TransferDescriptor::deviceToDisk(0, {1}, 1), TransferDescriptor::deviceToDisk(1, {2}, 2)},
                     std::chrono::seconds(30));
    auto                     disk_engine = std::make_shared<RecordingStoreTransferEngine>(group_sets);
    BlockTransferDispatcher  disk_dispatcher(disk_engine);
    std::optional<ErrorInfo> disk_result;
    runner.runTransfer(
        disk_task, disk_dispatcher, metrics_reporter, [&](ErrorInfo error) { disk_result.emplace(std::move(error)); });
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
    StoreTaskRunner                runner(group_sets);
    auto                           engine = std::make_shared<PendingStoreTransferEngine>();
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    BlockTreeTaskPool              outer_pool(1, 8, "AsyncStoreOuter");
    ASSERT_TRUE(outer_pool.start());

    auto first  = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{}, std::chrono::seconds(30));
    auto second = std::make_shared<StoreTaskRunner::Task>(Tier::HOST, CacheKeysType{}, std::chrono::seconds(30));
    first->transfer_task  = TransferTask({TransferDescriptor::deviceToHost(0, {1}, 1)}, std::chrono::seconds(30));
    second->transfer_task = TransferTask({TransferDescriptor::deviceToHost(0, {2}, 2)}, std::chrono::seconds(30));
    std::atomic<size_t> started{0};
    std::atomic<size_t> settled{0};
    const auto          submit_task = [&](const std::shared_ptr<StoreTaskRunner::Task>& task) {
        return outer_pool.submit(BlockTreeTaskClass::BACKGROUND, [&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo) {
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
