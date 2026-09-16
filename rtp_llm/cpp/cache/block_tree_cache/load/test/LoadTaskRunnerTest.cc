#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <deque>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
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
    DeviceBlockPoolPtr pool      = makeTestDevicePool({{group.kv_block_stride_bytes, group.kv_scale_stride_bytes}},
                                                 /*usable_count=*/1,
                                                 "load_task_runner_" + std::to_string(group_set_id));
    auto               host_pool = makeHostPool(group.kv_block_stride_bytes, /*usable_count=*/2);
    auto               disk_pool = makeDiskPool(group.kv_block_stride_bytes,
                                  /*usable_count=*/2,
                                  "/tmp",
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    return makeTestGroupSet(group_set_id, topology, {0}, {std::move(pool)}, std::move(host_pool), std::move(disk_pool));
}

LoadTaskRunner::TaskPtr makeLoadTask(std::vector<TransferDescriptor> descriptors) {
    std::vector<TransferDescriptor> host_to_device_descriptors;
    std::vector<TransferDescriptor> disk_to_device_descriptors;
    for (const TransferDescriptor& descriptor : descriptors) {
        if (descriptor.source_tier == Tier::HOST) {
            host_to_device_descriptors.push_back(descriptor);
        } else {
            disk_to_device_descriptors.push_back(descriptor);
        }
    }
    return std::make_shared<LoadTaskRunner::Task>(
        std::move(descriptors),
        TransferTask(std::move(host_to_device_descriptors), std::chrono::seconds(30)),
        TransferTask(std::move(disk_to_device_descriptors), std::chrono::seconds(30)));
}

class RecordingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingPerRankEngine(std::deque<bool> results):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), results_(std::move(results)) {}

    std::shared_ptr<AsyncContext> execute(TransferTask task) override {
        const auto& descriptors = task.descriptors();
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

    std::shared_ptr<AsyncContext> execute(TransferTask task) override {
        auto context  = std::make_shared<TransferBatchAsyncContext>();
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batches_.push_back(task.descriptors());
            contexts_.push_back(context);
            released = released_;
        }
        if (released) {
            context->complete(ErrorInfo::OkStatus());
        }
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
            // Release future submissions too: timeout cleanup must unblock queued
            // outer workers even if runTransfer regresses to waiting synchronously.
            released_ = true;
            contexts  = contexts_;
        }
        for (const auto& context : contexts) {
            context->complete(ErrorInfo::OkStatus());
        }
    }

    std::vector<std::vector<TransferDescriptor>>            batches_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;

private:
    mutable std::mutex mutex_;
    bool               released_{false};
};

TEST(LoadTaskRunnerTest, PendingEngineReleaseAllAlsoCompletesFutureSubmissions) {
    PendingPerRankEngine engine;
    auto first = engine.execute(TransferTask({TransferDescriptor::hostToDevice(0, 1, {1})}, std::chrono::seconds(30)));
    EXPECT_FALSE(first->done());
    engine.completeAll();
    EXPECT_TRUE(first->done());
    auto later = engine.execute(TransferTask({TransferDescriptor::hostToDevice(0, 2, {2})}, std::chrono::seconds(30)));
    EXPECT_TRUE(later->done());
    EXPECT_TRUE(later->success());
    EXPECT_NO_THROW(engine.completeAll());
}

TEST(LoadTaskRunnerTest, CreateTaskAllowsNoTransferDescriptors) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);

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
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);

    TransferDescriptor device_desc;
    device_desc.group_set_id                                  = 0;
    device_desc.source_tier                                   = Tier::DEVICE;
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context = coordinator->create({device_desc}, {false}, 1);
    LoadTaskRunner::TaskPtr                 task    = runner.createTask(context);
    EXPECT_EQ(task, nullptr);
}


TEST(LoadTaskRunnerTest, CreateTaskPartitionsHostAndDiskDescriptors) {
    GroupSetPtr                                   group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr>                group_sets{group};
    LoadTaskRunner                                runner(group_sets, 30'000, 30'000);
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context = coordinator->create(
        {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})}, {false, false}, 2);

    LoadTaskRunner::TaskPtr task = runner.createTask(context);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->load_descs.size(), 2u);
    ASSERT_EQ(task->host_to_device_task.descriptors().size(), 1u);
    EXPECT_EQ(task->host_to_device_task.descriptors().front().source_tier, Tier::HOST);
    ASSERT_EQ(task->disk_to_device_task.descriptors().size(), 1u);
    EXPECT_EQ(task->disk_to_device_task.descriptors().front().source_tier, Tier::DISK);
}

TEST(LoadTaskRunnerTest, CreateTaskAssignsIndependentHostAndDiskTimeouts) {
    GroupSetPtr                                   group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr>                group_sets{group};
    LoadTaskRunner                                runner(group_sets, 0, 30'000);
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context = coordinator->create(
        {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})}, {false, false}, 2);

    LoadTaskRunner::TaskPtr task = runner.createTask(context);

    ASSERT_NE(task, nullptr);
    EXPECT_FALSE(task->host_to_device_task.remainingTimeout().has_value());
    EXPECT_TRUE(task->disk_to_device_task.remainingTimeout().has_value());
}

TEST(LoadTaskRunnerTest, CreateTaskUsesHostTimeoutWithoutDiskDescriptors) {
    GroupSetPtr                                   group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr>                group_sets{group};
    LoadTaskRunner                                runner(group_sets, 30'000, 0);
    const std::shared_ptr<LoadContextCoordinator> coordinator = std::make_shared<LoadContextCoordinator>(
        LoadContextCoordinator::CommitCallback{}, LoadContextCoordinator::AbortCallback{});
    const std::shared_ptr<LoadAsyncContext> context =
        coordinator->create({TransferDescriptor::hostToDevice(0, 1, {1})}, {false}, 1);

    LoadTaskRunner::TaskPtr task = runner.createTask(context);

    ASSERT_NE(task, nullptr);
    EXPECT_TRUE(task->host_to_device_task.remainingTimeout().has_value());
    EXPECT_TRUE(task->disk_to_device_task.descriptors().empty());
}

TEST(LoadTaskRunnerTest, HostBatchCompletesBeforeDiskBatchIsSubmitted) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    auto                           task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    std::optional<ErrorInfo> result;

    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
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
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{false});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    auto                           task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    std::optional<ErrorInfo> result;

    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0"}));
}

TEST(LoadTaskRunnerTest, ExpiredBusinessTaskDoesNotReachTransferEngine) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    auto                           task = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1})});
    task->host_to_device_task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);
    std::optional<ErrorInfo> result;

    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(engine->batches_.empty());
}

TEST(LoadTaskRunnerTest, DiskFailureFailsLoadAfterHostSuccess) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, false});
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    auto                           task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    std::optional<ErrorInfo> result;

    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

TEST(LoadTaskRunnerTest, SplitsEachDirectionByGroupSetId) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet(0), makeTaskRunnerTestGroupSet(1)};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                    engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true, true, true});
    BlockTransferDispatcher dispatcher(engine, nullptr, 8);
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    auto                          task = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}),
                                                       TransferDescriptor::hostToDevice(0, 2, {2}),
                                                       TransferDescriptor::hostToDevice(1, 3, {3}),
                                                       TransferDescriptor::diskToDevice(0, 4, {4}),
                                                       TransferDescriptor::diskToDevice(1, 5, {5}),
                                                       TransferDescriptor::diskToDevice(1, 6, {6})});
    std::optional<ErrorInfo>      result;

    runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo error) { result.emplace(std::move(error)); });
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
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    BlockTreeTaskPool              outer_pool(1, 8, "AsyncLoadOuter");
    ASSERT_TRUE(outer_pool.start());

    auto                    first  = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1})});
    auto                    second = makeLoadTask({TransferDescriptor::hostToDevice(0, 2, {2})});
    std::atomic<size_t>     started{0};
    std::atomic<size_t>     settled{0};
    std::mutex              started_mutex;
    std::condition_variable started_cv;
    const auto              submit_task = [&](const LoadTaskRunner::TaskPtr& task) {
        return outer_pool.submit(BlockTreeTaskClass::LOAD, [&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo) {
                EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); }));
            });
            {
                std::lock_guard<std::mutex> lock(started_mutex);
                started.fetch_add(1);
            }
            started_cv.notify_all();
        });
    };

    EXPECT_TRUE(submit_task(first));
    EXPECT_TRUE(submit_task(second));
    {
        std::unique_lock<std::mutex> lock(started_mutex);
        EXPECT_TRUE(started_cv.wait_for(lock, std::chrono::seconds(5), [&] { return started.load() == 2; }));
    }
    // The bounded predicate above proves runTransfer returned. Never drain
    // outer workers before releasing mock transfers on the timeout path.
    EXPECT_EQ(started.load(), 2u);
    EXPECT_EQ(engine->contextCount(), 2u);
    EXPECT_EQ(settled.load(), 0u);

    engine->completeAll();
    outer_pool.waitForIdle();
    EXPECT_EQ(settled.load(), 2u);
}

TEST(LoadTaskRunnerTest, HundredPendingTransfersAreNotCappedByFourOuterWorkers) {
    constexpr size_t               kBusinessCount = 100;
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    LoadTaskRunner                 runner(group_sets, 30'000, 30'000);
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine);
    BlockTreeCacheMetricsReporter  metrics_reporter{nullptr};
    BlockTreeTaskPool              outer_pool(/*thread_count=*/4,
                                 /*queue_size=*/kBusinessCount + BlockTreeTaskPool::kLoadReservedSlots,
                                 "AsyncLoadOuter");
    ASSERT_TRUE(outer_pool.start());

    std::atomic<size_t>     started{0};
    std::atomic<size_t>     settled{0};
    std::mutex              started_mutex;
    std::condition_variable started_cv;
    for (size_t index = 0; index < kBusinessCount; ++index) {
        const BlockIdxType block_index = static_cast<BlockIdxType>(index + 1);
        auto               task = makeLoadTask({TransferDescriptor::hostToDevice(0, block_index, {block_index})});
        EXPECT_TRUE(outer_pool.submit(BlockTreeTaskClass::LOAD, [&, task] {
            runner.runTransfer(task, dispatcher, metrics_reporter, [&](ErrorInfo) {
                EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); }));
            });
            {
                std::lock_guard<std::mutex> lock(started_mutex);
                started.fetch_add(1);
            }
            started_cv.notify_all();
        }));
    }

    {
        std::unique_lock<std::mutex> lock(started_mutex);
        EXPECT_TRUE(
            started_cv.wait_for(lock, std::chrono::seconds(5), [&] { return started.load() == kBusinessCount; }));
    }
    // The bounded predicate above proves runTransfer returned. Never drain
    // outer workers before releasing mock transfers on the timeout path.
    EXPECT_EQ(started.load(), kBusinessCount);
    EXPECT_EQ(engine->contextCount(), kBusinessCount);
    EXPECT_EQ(settled.load(), 0u);

    engine->completeAll();
    outer_pool.waitForIdle();
    EXPECT_EQ(settled.load(), kBusinessCount);
}

}  // namespace
}  // namespace rtp_llm
