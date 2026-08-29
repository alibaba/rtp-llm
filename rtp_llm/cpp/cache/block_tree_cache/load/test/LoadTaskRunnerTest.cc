#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <deque>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

GroupSetPtr makeTaskRunnerTestGroupSet(size_t group_set_id = 0) {
    using namespace block_transfer_engine_test;

    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const GroupBase                            group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr pool = makeTestDevicePool({{group.kv_block_stride_bytes, group.kv_scale_stride_bytes}},
                                                 /*usable_count=*/1,
                                                 "load_task_runner_" + std::to_string(group_set_id));
    auto host_pool          = makeHostPool(group.kv_block_stride_bytes, /*usable_count=*/2, /*enable_pinned=*/false);
    auto disk_pool          = makeDiskPool(group.kv_block_stride_bytes,
                                  /*usable_count=*/2,
                                  "/tmp",
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    return makeTestGroupSet(group_set_id, topology, {0}, {std::move(pool)}, std::move(host_pool), std::move(disk_pool));
}

class RecordingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingPerRankEngine(std::deque<bool> results):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), results_(std::move(results)) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches_.push_back(descriptors);
        const TransferDescriptor& first     = descriptors.front();
        const std::string         direction = first.source_tier == Tier::HOST ? "host_" : "disk_";
        const std::string         suffix    = direction + std::to_string(first.group_set_id);
        events_.push_back("submit_" + suffix);
        const bool result = results_.front();
        results_.pop_front();
        if (result) {
            return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
        }
        return std::make_shared<CompletedAsyncContext>(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "submission failed"));
    }

    std::vector<std::vector<TransferDescriptor>> batches_;
    std::deque<bool>                             results_;
    std::vector<std::string>                     events_;
};

class PendingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    PendingPerRankEngine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        std::lock_guard<std::mutex> lock(mutex_);
        batches_.push_back(descriptors);
        auto context = std::make_shared<TransferBatchAsyncContext>();
        contexts_.push_back(context);
        return context;
    }

    size_t contextCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return contexts_.size();
    }

    void completeAll() {
        std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            contexts = contexts_;
        }
        for (const auto& context : contexts) {
            context->complete(ErrorInfo::OkStatus());
        }
    }

    std::vector<std::vector<TransferDescriptor>>            batches_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;

private:
    mutable std::mutex mutex_;
};

TEST(LoadTaskRunnerTest, CreateTaskAllowsNoTransferDescriptors) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);

    TransferDescriptor joined_desc;
    joined_desc.group_set_id                                  = 0;
    joined_desc.source_tier                                   = Tier::HOST;
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context = coordinator->create({joined_desc}, {true}, 1);
    LoadTaskRunner::TaskPtr                 task    = runner.createTask(context);
    EXPECT_EQ(task, nullptr);
}

TEST(LoadTaskRunnerTest, CreateTaskSkipsDeviceDescriptors) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);

    TransferDescriptor device_desc;
    device_desc.group_set_id                                  = 0;
    device_desc.source_tier                                   = Tier::DEVICE;
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context = coordinator->create({device_desc}, {false}, 1);
    LoadTaskRunner::TaskPtr                 task    = runner.createTask(context);
    EXPECT_EQ(task, nullptr);
}

TEST(LoadTaskRunnerTest, HostBatchCompletesBeforeDiskBatchIsSubmitted) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter;
    auto                           task = std::make_shared<LoadTaskRunner::Task>();
    task->load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})};
    std::optional<ErrorInfo> result;

    runner.runTransfer(
        task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->ok());
    ASSERT_EQ(engine->batches_.size(), 2u);
    EXPECT_EQ(engine->batches_[0].front().source_tier, Tier::HOST);
    EXPECT_EQ(engine->batches_[1].front().source_tier, Tier::DISK);
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

TEST(LoadTaskRunnerTest, HostFailureSkipsDiskBatch) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{false});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter;
    auto                           task = std::make_shared<LoadTaskRunner::Task>();
    task->load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})};
    std::optional<ErrorInfo> result;

    runner.runTransfer(
        task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0"}));
}

TEST(LoadTaskRunnerTest, DiskFailureFailsLoadAfterHostSuccess) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, false});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter;
    auto                           task = std::make_shared<LoadTaskRunner::Task>();
    task->load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})};
    std::optional<ErrorInfo> result;

    runner.runTransfer(
        task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

TEST(LoadTaskRunnerTest, SplitsEachDirectionByGroupSetId) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet(0), makeTaskRunnerTestGroupSet(1)};
    LoadTaskRunner                 runner(group_sets);
    auto                    engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true, true, true});
    BlockTransferDispatcher dispatcher(engine, nullptr, 8, 8);
    BlockTreeCacheMetricsReporter metrics_reporter;
    auto                          task = std::make_shared<LoadTaskRunner::Task>();
    task->load_descs                   = {TransferDescriptor::hostToDevice(0, 1, {1}),
                                          TransferDescriptor::hostToDevice(0, 2, {2}),
                                          TransferDescriptor::hostToDevice(1, 3, {3}),
                                          TransferDescriptor::diskToDevice(0, 4, {4}),
                                          TransferDescriptor::diskToDevice(1, 5, {5}),
                                          TransferDescriptor::diskToDevice(1, 6, {6})};
    std::optional<ErrorInfo> result;

    runner.runTransfer(
        task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->ok());
    ASSERT_EQ(engine->batches_.size(), 4u);
    EXPECT_EQ(engine->batches_[0].size(), 2u);
    EXPECT_EQ(engine->batches_[1].size(), 1u);
    EXPECT_EQ(engine->batches_[2].size(), 1u);
    EXPECT_EQ(engine->batches_[3].size(), 2u);
    for (size_t batch_index = 0; batch_index < engine->batches_.size(); ++batch_index) {
        const size_t expected_group_set_id = batch_index % 2;
        for (const auto& descriptor : engine->batches_[batch_index]) {
            EXPECT_EQ(descriptor.group_set_id, expected_group_set_id);
        }
    }
    EXPECT_EQ(engine->events_,
              (std::vector<std::string>{"submit_host_0", "submit_host_1", "submit_disk_0", "submit_disk_1"}));
}

TEST(LoadTaskRunnerTest, PendingTransferDoesNotRetainOuterWorker) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    LoadTaskRunner                 runner(group_sets);
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter;
    BlockTreeTaskPool              outer_pool(1, 8, "AsyncLoadOuter");
    ASSERT_TRUE(outer_pool.start());

    auto first         = std::make_shared<LoadTaskRunner::Task>();
    auto second        = std::make_shared<LoadTaskRunner::Task>();
    first->load_descs  = {TransferDescriptor::hostToDevice(0, 1, {1})};
    second->load_descs = {TransferDescriptor::hostToDevice(0, 2, {2})};
    std::atomic<size_t> started{0};
    std::atomic<size_t> settled{0};
    const auto          submit_task = [&](const LoadTaskRunner::TaskPtr& task) {
        return outer_pool.submit([&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo) {
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
    ASSERT_EQ(engine->contexts_.size(), 2u);
    EXPECT_EQ(settled.load(), 0u);

    engine->contexts_[0]->complete(ErrorInfo::OkStatus());
    engine->contexts_[1]->complete(ErrorInfo::OkStatus());
    outer_pool.waitForIdle();
    EXPECT_EQ(settled.load(), 2u);
}

TEST(LoadTaskRunnerTest, HundredPendingTransfersAreNotCappedByFourOuterWorkers) {
    constexpr size_t               kBusinessCount = 100;
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    LoadTaskRunner                 runner(group_sets);
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter;
    BlockTreeTaskPool              outer_pool(/*thread_count=*/4, /*queue_size=*/128, "AsyncLoadOuter");
    ASSERT_TRUE(outer_pool.start());

    std::atomic<size_t> started{0};
    std::atomic<size_t> settled{0};
    for (size_t index = 0; index < kBusinessCount; ++index) {
        auto               task        = std::make_shared<LoadTaskRunner::Task>();
        const BlockIdxType block_index = static_cast<BlockIdxType>(index + 1);
        task->load_descs               = {TransferDescriptor::hostToDevice(0, block_index, {block_index})};
        ASSERT_TRUE(outer_pool.submit([&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100, [&](ErrorInfo) {
                EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); }));
            });
            started.fetch_add(1);
        }));
    }

    for (size_t attempt = 0; attempt < 200 && started.load() != kBusinessCount; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(started.load(), kBusinessCount);
    EXPECT_EQ(engine->contextCount(), kBusinessCount);
    EXPECT_EQ(settled.load(), 0u);

    engine->completeAll();
    outer_pool.waitForIdle();
    EXPECT_EQ(settled.load(), kBusinessCount);
}

}  // namespace
}  // namespace rtp_llm
