#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <deque>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::bufferSafeResult;

kmonitor::MetricsTags stageTags(const char* stage, const char* source, bool success) {
    kmonitor::MetricsTags tags("operation", "load");
    tags.AddTag("source_tier", source);
    tags.AddTag("target_tier", "DEVICE");
    tags.AddTag("scope", "workflow_callback");
    tags.AddTag("stage", stage);
    tags.AddTag("success", success ? "true" : "false");
    return tags;
}

double snapshotMetric(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags& tags) {
    const auto serving_tags = kmonTagsWithServiceStatus(&tags);
    auto* series = metric->DeclareMetric(&serving_tags);
    if (series == nullptr) {
        ADD_FAILURE() << "missing metric: " << tags.ToString();
        return -1;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    series->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(series));
    if (record.Values().size() != 1) {
        ADD_FAILURE() << "unexpected metric value count: " << record.Values().size();
        return -1;
    }
    return std::stod(record.Values().front()->Value());
}

GroupSetPtr makeTaskRunnerTestGroupSet(size_t group_set_id       = 0,
                                       bool   enable_crc         = false,
                                       size_t device_block_count = 1) {
    using namespace block_transfer_engine_test;

    auto policy                                             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                              = true;
    const auto                                 group_config = makeTestGroupBase(policy);
    const auto                                 kv_bytes     = group_config.group.kvBlockStrideBytes();
    const auto                                 scale_bytes  = group_config.group.kvScaleStrideBytes();
    const std::shared_ptr<const CacheTopology> topology     = makeTestTopology({group_config});
    DeviceBlockPoolPtr                         pool         = makeTestDevicePool({{kv_bytes, scale_bytes}},
                                                 device_block_count,
                                                 "load_task_runner_" + std::to_string(group_set_id));
    auto                                       host_pool    = makeHostPool(kv_bytes, /*usable_count=*/2);
    auto                                       disk_pool    = makeDiskPool(kv_bytes,
                                  /*usable_count=*/2,
                                  "/tmp",
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    return makeTestGroupSet(
        group_set_id, topology, {"group0"}, {std::move(pool)}, std::move(host_pool), std::move(disk_pool), enable_crc);
}

std::shared_ptr<LoadTask> makeLoadTask(std::vector<TransferDescriptor> descriptors) {
    auto       coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    const auto count       = descriptors.size();
    auto       context     = coordinator->create(std::move(descriptors), std::vector<bool>(count, false), {}, count);
    return LoadTask::create(std::move(context), 30'000, 30'000);
}

class RecordingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingPerRankEngine(std::deque<bool> results):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), results_(std::move(results)) {}

    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        batches_.emplace_back(descriptors.begin(), descriptors.end());
        const TransferDescriptor& first     = descriptors.front();
        const std::string         direction = first.source_tier == Tier::HOST ? "host_" : "disk_";
        const std::string         suffix    = direction + std::to_string(first.group_set_id);
        events_.push_back("submit_" + suffix);
        const bool result = results_.front();
        results_.pop_front();
        if (result) {
            return std::make_shared<CompletedTransferAsyncContext>(bufferSafeResult());
        }
        return std::make_shared<CompletedTransferAsyncContext>(
            bufferSafeResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "submission failed")));
    }

    std::vector<std::vector<TransferDescriptor>> batches_;
    std::deque<bool>                             results_;
    std::vector<std::string>                     events_;
};

class PendingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit PendingPerRankEngine(std::vector<GroupSetPtr> groups = {}):
        PerRankBlockTransferEngine(std::move(groups)) {}

    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task) override {
        auto context  = std::make_shared<TransferBatchAsyncContext>();
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batches_.emplace_back(task.descriptors().begin(), task.descriptors().end());
            contexts_.push_back(context);
            released = released_;
        }
        if (released) {
            context->complete(bufferSafeResult());
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
            // outer workers even if run regresses to waiting synchronously.
            released_ = true;
            contexts  = contexts_;
        }
        for (const auto& context : contexts) {
            context->complete(bufferSafeResult());
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

TEST(LoadTaskTest, ConstructorPartitionsHostAndDiskDescriptors) {
    const std::shared_ptr<LoadContextCoordinator> coordinator =
        std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    auto               joined = block_tree_cache_test::makeLoadOperationForTest();
    TransferDescriptor device_desc;
    device_desc.source_tier = Tier::DEVICE;
    const auto context      = coordinator->create({joined->descriptor(),
                                                   TransferDescriptor::hostToDevice(0, 1, {1}),
                                                   device_desc,
                                                   TransferDescriptor::diskToDevice(0, 2, {2})},
                                                  {true, false, false, false},
                                                  {joined},
                                             4,
                                             1,
                                             3);

    auto task = LoadTask::create(context, 30'000, 30'000);

    EXPECT_EQ(task->host_to_device_task.workflowId(), context->workflowId());
    EXPECT_EQ(task->disk_to_device_task.workflowId(), context->workflowId());
    EXPECT_EQ(task->host_to_device_task.parentTraceId(), context->workflowId());
    EXPECT_EQ(task->disk_to_device_task.parentTraceId(), context->workflowId());
    EXPECT_EQ(task->load_descs.size(), 2u);
    ASSERT_EQ(task->host_to_device_task.descriptors().size(), 1u);
    EXPECT_EQ(task->host_to_device_task.descriptors().front().source_tier, Tier::HOST);
    ASSERT_EQ(task->disk_to_device_task.descriptors().size(), 1u);
    EXPECT_EQ(task->disk_to_device_task.descriptors().front().source_tier, Tier::DISK);
}

TEST(LoadTaskTest, ConstructorAssignsIndependentHostAndDiskTimeouts) {
    const std::shared_ptr<LoadContextCoordinator> coordinator =
        std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    const std::shared_ptr<LoadAsyncContext> context =
        coordinator->create({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})},
                            {false, false},
                            {},
                            2);

    auto task = LoadTask::create(context, 0, 30'000);

    EXPECT_FALSE(task->host_to_device_task.remainingTimeout().has_value());
    EXPECT_TRUE(task->disk_to_device_task.remainingTimeout().has_value());
}

TEST(LoadTaskTest, ConstructorUsesHostTimeoutWithoutDiskDescriptors) {
    const std::shared_ptr<LoadContextCoordinator> coordinator =
        std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    const std::shared_ptr<LoadAsyncContext> context =
        coordinator->create({TransferDescriptor::hostToDevice(0, 1, {1})}, {false}, {}, 1);

    auto task = LoadTask::create(context, 30'000, 0);

    EXPECT_TRUE(task->host_to_device_task.remainingTimeout().has_value());
    EXPECT_TRUE(task->disk_to_device_task.descriptors().empty());
}

TEST(LoadTaskTest, CompletionTransferUsesRecordedDiskResult) {
    auto task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {2}), TransferDescriptor::diskToDevice(0, 3, {4})});
    EXPECT_EQ(task->completionTransfer().traceId(), task->host_to_device_task.traceId());
    task->disk_result = TransferResult(1);
    EXPECT_EQ(task->completionTransfer().traceId(), task->disk_to_device_task.traceId());
}

TEST(LoadTaskRunnerTest, PendingTransfersKeepTaskAliveAfterRunnerDestruction) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter  reporter{nullptr};
    BlockTreeTaskPool              pool(1, 8, "LoadTaskRunnerTest");
    auto                           task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    std::weak_ptr<LoadTask> weak_task   = task;
    size_t                  completions = 0;
    {
        LoadTaskRunner runner(group_sets, &dispatcher, pool, reporter);
        runner.run(task, [&]() {
            const auto running_task = weak_task.lock();
            ASSERT_NE(running_task, nullptr);
            EXPECT_TRUE(running_task->host_result.ok());
            EXPECT_TRUE(running_task->disk_result.ok());
            ++completions;
        });
    }
    task.reset();
    EXPECT_FALSE(weak_task.expired());
    EXPECT_EQ(completions, 0u);
    engine->completeAll();
    EXPECT_EQ(completions, 1u);
    EXPECT_TRUE(weak_task.expired());
}

TEST(LoadTaskRunnerTest, ConcurrentTasksKeepIndependentResults) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    auto                           raw = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    BlockTreeCacheMetricsReporter  reporter(raw, true, std::chrono::hours(1));
    BlockTreeTaskPool              pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                 runner(group_sets, &dispatcher, pool, reporter);
    auto                           first =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    auto second =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 3, {3}), TransferDescriptor::diskToDevice(0, 4, {4})});
    size_t                                 first_completions  = 0;
    size_t                                 second_completions = 0;
    block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
    runner.run(first, [&]() { ++first_completions; });
    runner.run(second, [&]() { ++second_completions; });
    ASSERT_EQ(engine->contextCount(), 4u);
    engine->contexts_[2]->complete(bufferSafeResult());
    engine->contexts_[0]->complete(bufferSafeResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "first host failed")));
    EXPECT_EQ(first_completions, 0u);
    EXPECT_EQ(second_completions, 0u);
    engine->contexts_[3]->complete(bufferSafeResult());
    EXPECT_EQ(first_completions, 0u);
    EXPECT_EQ(second_completions, 1u);
    engine->contexts_[1]->complete(bufferSafeResult());
    EXPECT_EQ(first_completions, 1u);
    EXPECT_EQ(second_completions, 1u);
    EXPECT_FALSE(first->host_result.ok());
    EXPECT_TRUE(first->disk_result.ok());
    EXPECT_TRUE(second->host_result.ok());
    EXPECT_TRUE(second->disk_result.ok());
}

TEST(LoadTaskRunnerTest, SettlementCallbackRetainsTaskWhileWaitingForJoinedLoad) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};
    auto                           engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher        dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter  reporter{nullptr};
    BlockTreeTaskPool              pool(1, 8, "LoadRunnerSubmit");
    LoadTaskRunner                 runner(group_sets, &dispatcher, pool, reporter);
    ASSERT_TRUE(pool.start());

    auto                                   joined = block_tree_cache_test::makeLoadOperationForTest();
    std::weak_ptr<LoadTask>                weak_task;
    std::atomic<size_t>                    settlements{0};
    block_tree_cache_detail::ScopeRollback drain([&] {
        engine->completeAll();
        joined->complete(false);
        pool.waitForIdle();
    });
    auto                                   coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const auto& context) {
            auto task = LoadTask::create(context, 30'000, 30'000);
            weak_task = task;
            context->setSettlementReadyCallback([&, task](const auto& ready) {
                EXPECT_EQ(task, weak_task.lock());
                EXPECT_EQ(ready, task->context);
                EXPECT_TRUE(ready->settle(ready->localTransfersSucceeded(), ready->hostTransfersSucceeded()));
                ++settlements;
            });
            return runner.submit(task);
        },
        [](auto&) {});
    auto context = coordinator->create({joined->descriptor(),
                                        TransferDescriptor::hostToDevice(0, 1, {1}),
                                        TransferDescriptor::diskToDevice(0, 2, {2})},
                                       {true, false, false},
                                       {joined},
                                       3);
    ASSERT_TRUE(context->commit());
    auto dispatched        = std::make_shared<std::promise<void>>();
    auto dispatched_future = dispatched->get_future();
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::LOAD, [dispatched] { dispatched->set_value(); }));
    ASSERT_EQ(dispatched_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(engine->contextCount(), 2u);
    EXPECT_FALSE(weak_task.expired());

    engine->completeAll();
    EXPECT_EQ(settlements.load(), 0u);
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(weak_task.expired());

    joined->complete(true);
    EXPECT_EQ(settlements.load(), 1u);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(weak_task.expired());
}

TEST(LoadTaskRunnerTest, InlineHostAndDiskCompletionSucceeds) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};

    auto                          engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true});
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    size_t completions = 0;

    runner.run(task, [&]() { ++completions; });
    ASSERT_EQ(completions, 1u);
    EXPECT_TRUE(task->host_result.ok());
    EXPECT_TRUE(task->disk_result.ok());
    ASSERT_EQ(engine->batches_.size(), 2u);
    EXPECT_EQ(engine->batches_[0].front().source_tier, Tier::HOST);
    EXPECT_EQ(engine->batches_[1].front().source_tier, Tier::DISK);
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

TEST(LoadTaskRunnerTest, InlineHostFailureStillSubmitsDiskBatch) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};

    auto                          engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{false, true});
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    size_t completions = 0;

    runner.run(task, [&]() { ++completions; });
    ASSERT_EQ(completions, 1u);
    EXPECT_FALSE(task->host_result.ok());
    EXPECT_TRUE(task->disk_result.ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

class ConcurrentLoadTaskRunnerTest: public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

TEST_P(ConcurrentLoadTaskRunnerTest, BothDirectionsSubmitAndDrainBeforeCompletion) {
    const auto [host_first, host_success, disk_success] = GetParam();
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    size_t                                 completions = 0;
    block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
    runner.run(task, [&]() { ++completions; });

    ASSERT_EQ(engine->contextCount(), 2u);
    EXPECT_EQ(completions, 0u);
    EXPECT_EQ(engine->batches_[0].front().source_tier, Tier::HOST);
    EXPECT_EQ(engine->batches_[1].front().source_tier, Tier::DISK);
    const auto finish = [&](size_t index) {
        const bool success = index == 0 ? host_success : disk_success;
        const auto status  = success ? ErrorInfo::OkStatus() : ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "copy failed");
        engine->contexts_[index]->complete(bufferSafeResult(status));
    };
    finish(host_first ? 0 : 1);
    EXPECT_EQ(completions, 0u);
    finish(host_first ? 1 : 0);
    ASSERT_EQ(completions, 1u);
    EXPECT_EQ(task->host_result.ok(), host_success);
    EXPECT_EQ(task->disk_result.ok(), disk_success);
    EXPECT_EQ(task->host_result.block_errors.size(), 1u);
    EXPECT_EQ(task->disk_result.block_errors.size(), 1u);
}

TEST_P(ConcurrentLoadTaskRunnerTest, PreservesStageMetricsAndReportsOnlyTheSuccessfulDirectionsWait) {
    block_tree_cache_test::ScopedServingMetrics serving_metrics;
    const auto [host_first, host_success, disk_success] = GetParam();
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    auto                          raw = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    BlockTreeCacheMetricsReporter reporter(raw, true, std::chrono::hours(1));
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    size_t                                 completions = 0;
    block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
    runner.run(task, [&]() {
        ++completions;
        EXPECT_EQ(task->host_result.ok(), host_success);
        EXPECT_EQ(task->disk_result.ok(), disk_success);
    });
    ASSERT_EQ(engine->contextCount(), 2u);
    const auto finish = [&](size_t index) {
        const bool success = index == 0 ? host_success : disk_success;
        const auto status  = success ? ErrorInfo::OkStatus() : ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "copy failed");
        engine->contexts_[index]->complete(bufferSafeResult(status));
    };
    finish(host_first ? 0 : 1);
    EXPECT_EQ(completions, 0u);
    reporter.flushStageMetrics();
    auto* metrics = raw->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    ASSERT_NE(metrics, nullptr);
    for (bool success : {false, true}) {
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait", "MIXED", success)), 0);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_disk2d", "HOST", success)),
                         0);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_h2d", "DISK", success)), 0);
    }
    const auto wait_begin = AsyncProfileSession::steadyNowUs();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto wait_duration = AsyncProfileSession::steadyNowUs() - wait_begin;
    finish(host_first ? 1 : 0);
    EXPECT_EQ(completions, 1u);
    reporter.flushStageMetrics();
    for (Tier tier : {Tier::HOST, Tier::DISK}) {
        const bool success = tier == Tier::HOST ? host_success : disk_success;
        for (const char* stage : {"callback_total", "save_result", "metrics_report"}) {
            EXPECT_DOUBLE_EQ(
                snapshotMetric(metrics->transfer_stage_qps_metric, stageTags(stage, tierName(tier), success)), 1);
            EXPECT_DOUBLE_EQ(
                snapshotMetric(metrics->transfer_stage_qps_metric, stageTags(stage, tierName(tier), !success)), 0);
        }
        EXPECT_DOUBLE_EQ(
            snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("next_stage_submit", tierName(tier), true)),
            1);
        kmonitor::MetricsTags tags("operation", "load");
        tags.AddTag("source_tier", tierName(tier));
        tags.AddTag("target_tier", "DEVICE");
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_qps_metric, tags), 1);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_failed_qps_metric, tags), success ? 0 : 1);
        const auto direction = BlockTreeCacheMetricsReporter::transferDirectionIndex(tier, Tier::DEVICE);
        EXPECT_EQ(reporter.transfer_in_flight_[0][direction].load(), 0);
    }
    for (bool success : {false, true}) {
        const bool overall_success = host_success && disk_success;
        for (const char* stage : {"wait", "context_complete"}) {
            EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags(stage, "MIXED", success)),
                             success == overall_success ? 1 : 0);
        }
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_disk2d", "HOST", success)),
                         host_first && host_success && success == disk_success ? 1 : 0);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_h2d", "DISK", success)),
                         !host_first && disk_success && success == host_success ? 1 : 0);
    }
    EXPECT_GE(snapshotMetric(metrics->transfer_stage_latency_us_metric,
                             stageTags("wait", "MIXED", host_success && disk_success)),
              wait_duration);
    if (host_first ? host_success : disk_success) {
        const auto tags = stageTags(host_first ? "wait_disk2d" : "wait_h2d",
                                    host_first ? "HOST" : "DISK",
                                    host_first ? disk_success : host_success);
        EXPECT_GE(snapshotMetric(metrics->transfer_stage_latency_us_metric, tags), wait_duration);
    }
}

INSTANTIATE_TEST_SUITE_P(CompletionOrdersAndFailures,
                         ConcurrentLoadTaskRunnerTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()));

TEST(LoadTaskRunnerTest, InlineTransfersReportWaitBeforeBusinessCompletion) {
    block_tree_cache_test::ScopedServingMetrics serving_metrics;
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    auto                          engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true});
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    auto                          raw = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    BlockTreeCacheMetricsReporter reporter(raw, true, std::chrono::hours(1));
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    auto session = AsyncProfileSession::start(128);
    ASSERT_NE(session, nullptr);
    block_tree_cache_detail::ScopeRollback stop_profile([&] { session->finish(); });
    int64_t                                callback_begin = 0;
    runner.run(task, [&]() {
        EXPECT_TRUE(task->host_result.ok());
        EXPECT_TRUE(task->disk_result.ok());
        callback_begin = AsyncProfileSession::nowUs();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    reporter.flushStageMetrics();
    auto* metrics = raw->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    ASSERT_NE(metrics, nullptr);
    for (const char* stage : {"wait", "context_complete"}) {
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags(stage, "MIXED", true)), 1);
    }
    EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_disk2d", "HOST", true)), 1);
    EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_h2d", "DISK", true)), 0);
    const auto events = session->finish();
    size_t     waits  = 0;
    for (const auto& event : events) {
        if (event.name.find(".wait(") != std::string::npos || event.name.find(".wait_disk2d(") != std::string::npos) {
            ++waits;
            EXPECT_EQ(event.name.find("incomplete=1"), std::string::npos);
            EXPECT_LE(event.end_us, callback_begin);
        }
    }
    EXPECT_EQ(waits, 2u);
}

TEST(LoadTaskRunnerTest, SingleDirectionDoesNotReportWaitingForAnAbsentTransfer) {
    block_tree_cache_test::ScopedServingMetrics serving_metrics;
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    for (Tier tier : {Tier::HOST, Tier::DISK}) {
        auto                    engine = std::make_shared<PendingPerRankEngine>();
        BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
        auto                    raw = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
        BlockTreeCacheMetricsReporter reporter(raw, true, std::chrono::hours(1));
        BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
        LoadTaskRunner                runner(group_sets, &dispatcher, pool, reporter);
        auto task = makeLoadTask({tier == Tier::HOST ? TransferDescriptor::hostToDevice(0, 1, {1}) :
                                                       TransferDescriptor::diskToDevice(0, 1, {1})});
        size_t                        completions = 0;
        runner.run(task, [&]() {
            ++completions;
            EXPECT_TRUE((tier == Tier::HOST ? task->host_result : task->disk_result).ok());
        });
        engine->completeAll();
        EXPECT_EQ(completions, 1u);
        reporter.flushStageMetrics();
        auto* metrics = raw->getMetricsGroup<RtpLLMCacheTransferMetrics>();
        ASSERT_NE(metrics, nullptr);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait", tierName(tier), true)),
                         1);
        EXPECT_DOUBLE_EQ(
            snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("context_complete", tierName(tier), true)), 1);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_disk2d", "HOST", true)), 0);
        EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_h2d", "DISK", true)), 0);
    }
}

TEST(LoadTaskRunnerTest, SimultaneousCompletionsReportExactlyOneDirectionalWait) {
    block_tree_cache_test::ScopedServingMetrics serving_metrics;
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    auto                          raw = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    BlockTreeCacheMetricsReporter reporter(raw, true, std::chrono::hours(1));
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    std::atomic<size_t>                    completions{0};
    block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
    runner.run(task, [&]() {
        EXPECT_TRUE(task->host_result.ok());
        EXPECT_TRUE(task->disk_result.ok());
        completions.fetch_add(1);
    });
    ASSERT_EQ(engine->contextCount(), 2u);
    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  ready  = 0;
    const auto              finish = [&](size_t index) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++ready;
            cv.notify_all();
            cv.wait(lock, [&] { return ready == 2; });
        }
        engine->contexts_[index]->complete(bufferSafeResult());
    };
    std::thread host(finish, 0);
    std::thread disk(finish, 1);
    host.join();
    disk.join();
    EXPECT_EQ(completions.load(), 1u);
    reporter.flushStageMetrics();
    auto* metrics = raw->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    ASSERT_NE(metrics, nullptr);
    EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait", "MIXED", true)), 1);
    EXPECT_DOUBLE_EQ(snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_disk2d", "HOST", true))
                         + snapshotMetric(metrics->transfer_stage_qps_metric, stageTags("wait_h2d", "DISK", true)),
                     1);
}

TEST(LoadTaskRunnerTest, ConcurrentFailuresPreserveUnknownCompletionAndCorruption) {
    for (bool host_unknown : {false, true}) {
        for (bool host_first : {false, true}) {
            const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet(0, false, 2)};
            // Unknown completion now retains physical targets in the coordinator.
            auto       device_pool = group_sets.front()->devicePools().front();
            const auto host_target = block_transfer_engine_test::poolMalloc(*device_pool);
            const auto disk_target = block_transfer_engine_test::poolMalloc(*device_pool);
            ASSERT_NE(host_target, NULL_BLOCK_IDX);
            ASSERT_NE(disk_target, NULL_BLOCK_IDX);

            auto                          engine = std::make_shared<PendingPerRankEngine>(group_sets);
            BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
            BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
            BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
            LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
            auto                          task = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {host_target}),
                                                               TransferDescriptor::diskToDevice(0, 2, {disk_target})});
            size_t                        completions = 0;
            block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
            runner.run(task, [&]() { ++completions; });
            ASSERT_EQ(engine->contextCount(), 2u);
            const auto finish = [&](size_t index) {
                const bool unknown = (index == 0) == host_unknown;
                TransferResult completion(
                    ErrorInfo(unknown ? ErrorCode::TRANSFER_COMPLETION_UNKNOWN : ErrorCode::EXECUTION_EXCEPTION,
                              unknown ? "completion unknown" : "CRC mismatch"),
                    unknown ? CacheCopyError::RPC_FAILED : CacheCopyError::CRC_MISMATCH);
                completion.buffer_safe = !unknown;
                engine->contexts_[index]->complete(std::move(completion));
            };
            finish(host_first ? 0 : 1);
            EXPECT_EQ(completions, 0u);
            finish(host_first ? 1 : 0);
            ASSERT_EQ(completions, 1u);
            EXPECT_EQ((host_unknown ? task->host_result : task->disk_result).status.code(),
                      ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
            EXPECT_EQ((host_unknown ? task->disk_result : task->host_result).status.code(),
                      ErrorCode::EXECUTION_EXCEPTION);
            EXPECT_NE((host_unknown ? task->disk_result : task->host_result).status.ToString().find("CRC mismatch"),
                      std::string::npos);
            EXPECT_NE(
                (host_unknown ? task->host_result : task->disk_result).status.ToString().find("completion unknown"),
                std::string::npos);
            EXPECT_EQ((host_unknown ? task->disk_result : task->host_result).blockError(0),
                      CacheCopyError::CRC_MISMATCH);
            EXPECT_TRUE(device_pool->isFrozen(host_unknown ? host_target : disk_target));
            EXPECT_FALSE(device_pool->isFrozen(host_unknown ? disk_target : host_target));
        }
    }
}

TEST(LoadTaskRunnerTest, ExpiredHostStillWaitsForDiskTransfer) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet()};

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    task->host_to_device_task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);
    size_t                                 completions = 0;
    block_tree_cache_detail::ScopeRollback drain([&] { engine->completeAll(); });
    runner.run(task, [&]() { ++completions; });
    ASSERT_EQ(engine->contextCount(), 1u);
    EXPECT_EQ(engine->batches_[0].front().source_tier, Tier::DISK);
    EXPECT_EQ(completions, 0u);
    engine->completeAll();
    ASSERT_EQ(completions, 1u);
    EXPECT_EQ(task->host_result.status.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(task->disk_result.ok());
}

TEST(LoadTaskRunnerTest, ExpiredBusinessTaskDoesNotReachTransferEngine) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};

    auto                          engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true});
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task  = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1})});
    task->host_to_device_task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);
    size_t completions                  = 0;

    runner.run(task, [&]() { ++completions; });

    ASSERT_EQ(completions, 1u);
    EXPECT_EQ(task->host_result.status.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(engine->batches_.empty());
}

TEST(LoadTaskRunnerTest, DiskFailureFailsLoadAfterHostSuccess) {
    GroupSetPtr                    group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};

    auto                          engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, false});
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task =
        makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, {2})});
    size_t completions = 0;

    runner.run(task, [&]() { ++completions; });
    ASSERT_EQ(completions, 1u);
    EXPECT_TRUE(task->host_result.ok());
    EXPECT_FALSE(task->disk_result.ok());
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "submit_disk_0"}));
}

TEST(LoadTaskRunnerTest, SplitsEachDirectionByGroupSetId) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet(0), makeTaskRunnerTestGroupSet(1)};

    auto                    engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true, true, true});
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             pool(1, 8, "LoadTaskRunnerTest");
    LoadTaskRunner                runner(group_sets, &dispatcher, pool, metrics_reporter);
    auto                          task = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1}),
                                                       TransferDescriptor::hostToDevice(0, 2, {2}),
                                                       TransferDescriptor::hostToDevice(1, 3, {3}),
                                                       TransferDescriptor::diskToDevice(0, 4, {4}),
                                                       TransferDescriptor::diskToDevice(1, 5, {5}),
                                                       TransferDescriptor::diskToDevice(1, 6, {6})});
    size_t                        completions = 0;

    runner.run(task, [&]() { ++completions; });
    ASSERT_EQ(completions, 1u);
    EXPECT_TRUE(task->host_result.ok());
    EXPECT_TRUE(task->disk_result.ok());
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

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             outer_pool(1, 8, "AsyncLoadOuter");
    LoadTaskRunner                runner(group_sets, &dispatcher, outer_pool, metrics_reporter);
    ASSERT_TRUE(outer_pool.start());

    auto                    first  = makeLoadTask({TransferDescriptor::hostToDevice(0, 1, {1})});
    auto                    second = makeLoadTask({TransferDescriptor::hostToDevice(0, 2, {2})});
    std::atomic<size_t>     started{0};
    std::atomic<size_t>     settled{0};
    std::mutex              started_mutex;
    std::condition_variable started_cv;
    const auto              submit_task = [&](const std::shared_ptr<LoadTask>& task) {
        return outer_pool.submit(BlockTreeTaskClass::LOAD, [&, task] {
            runner.run(task, [&]() { EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); })); });
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
    // The bounded predicate above proves run returned. Never drain
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

    auto                          engine = std::make_shared<PendingPerRankEngine>();
    BlockTransferDispatcher       dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    BlockTreeCacheMetricsReporter metrics_reporter{nullptr};
    BlockTreeTaskPool             outer_pool(/*thread_count=*/4,
                                 /*queue_size=*/kBusinessCount + BlockTreeTaskPool::kLoadReservedSlots,
                                 "AsyncLoadOuter");
    LoadTaskRunner                runner(group_sets, &dispatcher, outer_pool, metrics_reporter);
    ASSERT_TRUE(outer_pool.start());

    std::atomic<size_t>     started{0};
    std::atomic<size_t>     settled{0};
    std::mutex              started_mutex;
    std::condition_variable started_cv;
    for (size_t index = 0; index < kBusinessCount; ++index) {
        const BlockIdxType block_index = static_cast<BlockIdxType>(index + 1);
        auto               task = makeLoadTask({TransferDescriptor::hostToDevice(0, block_index, {block_index})});
        EXPECT_TRUE(outer_pool.submit(BlockTreeTaskClass::LOAD, [&, task] {
            runner.run(task, [&]() { EXPECT_TRUE(outer_pool.submitCompletion([&] { settled.fetch_add(1); })); });
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
    // The bounded predicate above proves run returned. Never drain
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
