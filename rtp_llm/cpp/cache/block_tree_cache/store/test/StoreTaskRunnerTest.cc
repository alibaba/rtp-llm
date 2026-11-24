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

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

// Fail one allocation on the submission thread after accepting the first batch.
namespace {
thread_local bool fail_next_store_allocation = false;
}
extern "C" void* __real__Znwm(size_t);
extern "C" void* __wrap__Znwm(size_t size) {
    if (std::exchange(fail_next_store_allocation, false))
        throw std::bad_alloc();
    return __real__Znwm(size);
}

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
    GroupSetPtr                    group_set = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const std::vector<GroupSetPtr> group_sets{group_set};
    StoreTaskRunner                runner(group_sets);

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

TEST(StoreTaskRunnerTest, PrepareTaskRecordsPathIndexInTransferDescriptors) {
    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const auto                                 group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr             device_pool = makeTestDevicePool({{16, 0}}, 4, "store_task_runner_path_index");
    std::shared_ptr<HostBlockPool> host_pool   = block_transfer_engine_test::makeHostPool(16, 2);
    GroupSetPtr                    group_set   = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
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
        const auto descriptors = task.descriptors();
        batches.emplace_back(descriptors.begin(), descriptors.end());
        return std::make_shared<CompletedTransferAsyncContext>(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected copy failure"));
    }

    std::vector<std::vector<TransferDescriptor>> batches;
};

class PendingStoreTransferEngine final: public PerRankBlockTransferEngine {
public:
    PendingStoreTransferEngine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        const auto descriptors = task.descriptors();
        batches.emplace_back(descriptors.begin(), descriptors.end());
        auto context = std::make_shared<TransferBatchAsyncContext>();
        contexts.push_back(context);
        return context;
    }

    std::vector<std::vector<TransferDescriptor>>            batches;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
};

TEST(StoreTaskRunnerTest, PressureCountsSuccessfulBatchWithinFailedStoreWithoutMetrics) {
    auto                          env = FullSWAEnvironment::create();
    StoreTaskRunner               runner(env->groups, env->cache->tree());
    BlockTreeCacheMetricsReporter metrics{nullptr};
    auto                          engine = std::make_shared<PendingStoreTransferEngine>();
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*env->cache, engine);
    auto& dispatcher = *env->cache->transfer_dispatcher_;
    auto  task       = std::make_shared<StoreTaskRunner::Task>(Tier::DISK, CacheKeysType{}, std::chrono::seconds(30));
    task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(0, {1, 1}, 1));
    task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(0, {2, 2}, 2));
    size_t callbacks = 0;
    runner.runTransfer(task, dispatcher, metrics, [&](TransferResult outcome) {
        EXPECT_FALSE(outcome.ok());
        ++callbacks;
    });
    ASSERT_EQ(engine->contexts.size(), 2u);
    engine->contexts[0]->complete(ErrorInfo::OkStatus());
    engine->contexts[1]->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
    EXPECT_EQ(callbacks, 1u);
    const auto& id = env->cache->tree()->pressurePoolId(env->disk_pools[0].get());
    const auto  evidence =
        env->cache->tree()->pressureState().transfer({0, id, CacheTransferOperation::STORE, Tier::DEVICE, Tier::DISK});
    EXPECT_EQ(evidence.copied_blocks, 1u);
    ASSERT_GT(env->disk_pools[0]->strideBytes(), env->disk_pools[0]->payloadBytes());
    EXPECT_EQ(evidence.copied_bytes, env->disk_pools[0]->strideBytes());
    EXPECT_EQ(evidence.published_blocks, 0u);
    EXPECT_EQ(evidence.failed_operations, 1u);
}

TEST(StoreTaskRunnerTest, PressureFailuresAreClassifiedByAffectedGroup) {
    auto                          env = FullSWAEnvironment::create();
    StoreTaskRunner               runner(env->groups, env->cache->tree());
    BlockTreeCacheMetricsReporter metrics{nullptr};
    auto                          engine = std::make_shared<PendingStoreTransferEngine>();
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*env->cache, engine);
    auto& dispatcher = *env->cache->transfer_dispatcher_;
    auto  task       = std::make_shared<StoreTaskRunner::Task>(Tier::DISK, CacheKeysType{}, std::chrono::seconds(30));
    task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(0, {1, 1}, 1));
    task->transfer_task.addDescriptor(TransferDescriptor::deviceToDisk(1, {2}, 2));
    size_t callbacks = 0;
    runner.runTransfer(task, dispatcher, metrics, [&](TransferResult outcome) {
        EXPECT_FALSE(outcome.ok());
        ++callbacks;
    });
    ASSERT_EQ(engine->contexts.size(), 2u);
    engine->contexts[0]->complete(ErrorInfo::OkStatus());
    engine->contexts[1]->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
    EXPECT_EQ(callbacks, 1u);
    const auto& id = env->cache->tree()->pressurePoolId(env->disk_pools[0].get());
    const auto  evidence =
        env->cache->tree()->pressureState().transfer({0, id, CacheTransferOperation::STORE, Tier::DEVICE, Tier::DISK});
    EXPECT_EQ(evidence.copied_blocks, 1u);
    EXPECT_EQ(evidence.copied_bytes, env->disk_pools[0]->strideBytes());
    EXPECT_EQ(evidence.published_blocks, 0u);
    EXPECT_EQ(evidence.failed_operations, 0u);
    const auto& failed_id = env->cache->tree()->pressurePoolId(env->disk_pools[1].get());
    const auto  failed    = env->cache->tree()->pressureState().transfer(
        {1, failed_id, CacheTransferOperation::STORE, Tier::DEVICE, Tier::DISK});
    EXPECT_EQ(failed.copied_blocks, 0u);
    EXPECT_EQ(failed.failed_operations, 1u);
}

TEST(StoreTaskRunnerTest, LaterDiskSubmissionFaultRetainsEarlierPhysicalHoldsUntilDrain) {
    struct Engine: PerRankBlockTransferEngine {
        Engine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}
        std::shared_ptr<TransferAsyncContext> execute(TransferTask) override {
            auto result = std::make_shared<TransferBatchAsyncContext>();
            if (!first) {
                first                      = result;
                fail_next_store_allocation = true;
            }
            return result;
        }
        std::shared_ptr<TransferBatchAsyncContext> first;
    };
    auto                          env    = FullSWAEnvironment::create();
    auto                          engine = std::make_shared<Engine>();
    BlockTransferDispatcher       dispatcher(engine);
    StoreTaskRunner               runner(env->groups);
    BlockTreeCacheMetricsReporter metrics{nullptr};
    auto task = std::make_shared<StoreTaskRunner::Task>(Tier::DISK, CacheKeysType{100, 101}, std::chrono::seconds(15));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(env->groups.size()));
    auto                                       source = allocateDeviceBlocksForTest(*env->groups[0], 2);
    ASSERT_EQ(source.size(), 2u);
    for (size_t i = 0; i < 2; ++i)
        resources[i][0].device_blocks = source[i];
    ASSERT_TRUE(runner.prepareTask(*task, resources));
    unreferenceDeviceBlocksForTest(*env->groups[0], source);
    size_t callbacks = 0;
    runner.runTransfer(task, dispatcher, metrics, [&](TransferResult outcome) {
        EXPECT_FALSE(outcome.ok());
        ++callbacks;
        runner.releaseTaskResources(*task);
    });
    EXPECT_FALSE(std::exchange(fail_next_store_allocation, false));
    EXPECT_EQ(callbacks, 0u);
    EXPECT_EQ(env->disk_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), 2u);
    EXPECT_EQ(env->device_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), 2u);
    ASSERT_TRUE(engine->first);
    engine->first->complete(ErrorInfo::OkStatus());
    dispatcher.drainTransfers();
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(env->disk_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_EQ(env->device_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
}

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
        runner.runTransfer(task, dispatcher, metrics_reporter, [&](TransferResult outcome) {
            result.emplace(std::move(outcome.status));
        });
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
        runner.runTransfer(task, dispatcher, metrics, [&](TransferResult outcome) {
            EXPECT_FALSE(outcome.ok());
            ++callbacks;
        });
        if (engine->contexts.size() != 3u) {
            for (const auto& context : engine->contexts) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "finish failed test"));
            }
            FAIL() << "expected three batches, got " << engine->contexts.size();
        }
        auto* metric =
            metrics_reporter->getMetricsGroup<RtpLLMCacheTransferMetrics>()->host_cache_copy_error_qps_metric;
        engine->contexts[1]->complete(
            {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC mismatch"), CacheCopyError::CRC_MISMATCH});
        engine->contexts[0]->complete(
            {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RPC failed"), CacheCopyError::RPC_FAILED});
        EXPECT_EQ(callbacks, 0u);
        EXPECT_EQ(metric->metric_data_->Size(), 0u);
        engine->contexts[2]->complete(ErrorInfo::OkStatus());
        EXPECT_EQ(callbacks, 1u);
        EXPECT_EQ(metric->metric_data_->Size(), 1u);
        kmonitor::MetricsTags tags("copy_direction", "FROM_GPU");
        tags.AddTag("error_type", "CRC_MISMATCH");
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
    GroupSetPtr                    group_set   = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
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
    runner.runTransfer(
        task, dispatcher, metrics_reporter, [&](TransferResult outcome) { result.emplace(std::move(outcome.status)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());

    runner.releaseTaskResources(*task);
    group_set->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, source_holder[0]}}});
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
    runner.runTransfer(host_task, host_dispatcher, metrics_reporter, [&](TransferResult outcome) {
        host_result.emplace(std::move(outcome.status));
    });
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
    runner.runTransfer(disk_task, disk_dispatcher, metrics_reporter, [&](TransferResult outcome) {
        disk_result.emplace(std::move(outcome.status));
    });
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
            runner.runTransfer(task, dispatcher, metrics_reporter, [&](TransferResult) {
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
