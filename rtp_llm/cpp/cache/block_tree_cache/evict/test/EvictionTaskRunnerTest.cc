#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

using ScriptedTransferEngine = block_tree_cache_test::ScriptedPerRankBlockTransferEngine;

TEST(EvictionTimingSnapshotTest, CapturesCandidateMetadataAtSelectionTime) {
    CandidateMeta candidate_meta;
    candidate_meta.tier_enter_time_us  = 10;
    candidate_meta.insert_time_us      = 20;
    candidate_meta.last_access_time_us = 30;

    const int64_t                before = currentTimeUs();
    const EvictionTimingSnapshot timing(candidate_meta);
    const int64_t                after = currentTimeUs();

    EXPECT_EQ(timing.tier_enter_time_us, 10);
    EXPECT_EQ(timing.insert_time_us, 20);
    EXPECT_EQ(timing.last_access_time_us, 30);
    EXPECT_GE(timing.selected_time_us, before);
    EXPECT_LE(timing.selected_time_us, after);
}

GroupSetPtr makeRunnerTestGroupSet(size_t                       group_set_id,
                                   const std::string&           pool_name,
                                   BlockTreeDiskBlockPoolPtr    disk_pool) {
    using namespace block_transfer_engine_test;
    auto topology    = makeTestTopology({makeTestGroupBase()});
    auto device_pool = makeTestDevicePool({{16, 0}}, 4, pool_name + "_device");
    auto host_pool   = makeHostPool(16, 4, false);
    return makeTestGroupSet(group_set_id,
                            std::move(topology),
                            {0},
                            {std::move(device_pool)},
                            std::move(host_pool),
                            std::move(disk_pool));
}

BlockTreeDiskBlockPoolPtr makeRunnerTestDiskPool(const std::string& pool_name) {
    using namespace block_transfer_engine_test;
    return makeDiskPool(16,
                        4,
                        "/tmp",
                        std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK),
                        pool_name);
}

class BatchCountingTransferEngine: public ScriptedTransferEngine {
public:
    using ScriptedTransferEngine::ScriptedTransferEngine;

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches_.push_back(descriptors);
        if (on_submit_) {
            on_submit_();
        }
        ++batch_count_;
        return ScriptedTransferEngine::submit(descriptors);
    }

    size_t batchCount() const {
        return batch_count_;
    }

    const std::vector<std::vector<TransferDescriptor>>& batches() const {
        return batches_;
    }

    void setOnSubmit(std::function<void()> on_submit) {
        on_submit_ = std::move(on_submit);
    }

private:
    std::vector<std::vector<TransferDescriptor>> batches_;
    size_t                batch_count_{0};
    std::function<void()> on_submit_;
};

class DeferredTransferEngine final: public PerRankBlockTransferEngine {
public:
    explicit DeferredTransferEngine(const std::vector<GroupSetPtr>& groups): PerRankBlockTransferEngine(groups) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches_.push_back(descriptors);
        auto context = std::make_shared<TransferBatchAsyncContext>();
        contexts_.push_back(context);
        return context;
    }

    size_t batchCount() const {
        return batches_.size();
    }

    const std::vector<TransferDescriptor>& batch(size_t index) const {
        return batches_.at(index);
    }

    void complete(size_t index, bool success) {
        contexts_.at(index)->complete(success ? ErrorInfo::OkStatus() :
                                                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
    }

private:
    std::vector<std::vector<TransferDescriptor>>       batches_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;
};

class EvictionTaskRunnerTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto disk_pool = makeRunnerTestDiskPool("eviction_runner_shared_disk");
        group_sets_ = {makeRunnerTestGroupSet(0, "eviction_runner_0", disk_pool),
                       makeRunnerTestGroupSet(1, "eviction_runner_1", disk_pool),
                       makeRunnerTestGroupSet(2, "eviction_runner_2", disk_pool)};
        transfer_engine_     = std::make_shared<BatchCountingTransferEngine>(group_sets_, false);
        transfer_dispatcher_ = std::make_unique<BlockTransferDispatcher>(transfer_engine_);
    }

    EvictionTaskRunner makeRunner() {
        return EvictionTaskRunner(group_sets_, transfer_dispatcher_.get(), 0, 0);
    }

    EvictionTaskResult runImmediately(EvictionTask task) {
        auto                              runner = makeRunner();
        std::optional<EvictionTaskResult> result;
        runner.runTransfer(std::make_shared<EvictionTask>(std::move(task)),
                           metrics_reporter_,
                           [&](EvictionTaskResult completed) { result = std::move(completed); });
        if (!result.has_value()) {
            ADD_FAILURE() << "scripted transfer did not complete inline";
            return {};
        }
        return std::move(*result);
    }

    void enableMetrics() {
        kmonitor::MetricsTags tags;
        metrics_reporter_.setMetricsReporter(std::make_shared<kmonitor::MetricsReporter>("", "", tags));
    }

    int64_t evictionInFlight(Tier source_tier, Tier target_tier) const {
        const size_t operation_index = static_cast<size_t>(CacheTransferOperation::EVICT);
        const size_t direction_index =
            static_cast<size_t>(BlockTreeCacheMetricsReporter::transferDirectionIndex(source_tier, target_tier));
        return metrics_reporter_.transfer_in_flight_[operation_index][direction_index].load();
    }

    std::vector<GroupSetPtr>                     group_sets_;
    std::shared_ptr<BatchCountingTransferEngine> transfer_engine_;
    std::unique_ptr<BlockTransferDispatcher>     transfer_dispatcher_;
    BlockTreeCacheMetricsReporter                metrics_reporter_;
};

EvictionTask makeCopyTask() {
    EvictionTask task;
    task.primary_desc.group_set_id  = 0;
    task.primary_desc.source_tier   = Tier::DEVICE;
    task.primary_desc.target_tier   = Tier::HOST;
    task.primary_desc.source_blocks = {1, 2};
    task.primary_desc.target_blocks = {3};

    TransferDescriptor cascade_desc;
    cascade_desc.group_set_id  = 1;
    cascade_desc.source_tier   = Tier::HOST;
    cascade_desc.target_tier   = Tier::DISK;
    cascade_desc.source_blocks = {4};
    cascade_desc.target_blocks = {5};
    task.cascade_descs.push_back(std::move(cascade_desc));
    return task;
}

TEST_F(EvictionTaskRunnerTest, SubmitsPrimaryAndCascadesSequentiallyWithoutBlocking) {
    auto deferred_engine = std::make_shared<DeferredTransferEngine>(group_sets_);
    BlockTransferDispatcher dispatcher(deferred_engine);
    EvictionTaskRunner runner(group_sets_, &dispatcher, 0, 0);
    auto task = std::make_shared<EvictionTask>(makeCopyTask());
    task->cascade_descs.push_back(TransferDescriptor::hostToDisk(2, 6, 7));
    std::optional<EvictionTaskResult> result;
    size_t terminal_count = 0;

    runner.runTransfer(task, metrics_reporter_, [&](EvictionTaskResult completed) {
        ++terminal_count;
        result = std::move(completed);
    });

    ASSERT_EQ(deferred_engine->batchCount(), 1u);
    ASSERT_EQ(deferred_engine->batch(0).size(), 1u);
    EXPECT_EQ(deferred_engine->batch(0).front().group_set_id, task->primary_desc.group_set_id);
    EXPECT_FALSE(result.has_value());

    deferred_engine->complete(0, true);
    ASSERT_EQ(deferred_engine->batchCount(), 2u);
    ASSERT_EQ(deferred_engine->batch(1).size(), 1u);
    EXPECT_EQ(deferred_engine->batch(1).front().group_set_id, task->cascade_descs[0].group_set_id);
    EXPECT_FALSE(result.has_value());

    deferred_engine->complete(1, false);
    ASSERT_EQ(deferred_engine->batchCount(), 3u);
    ASSERT_EQ(deferred_engine->batch(2).size(), 1u);
    EXPECT_EQ(deferred_engine->batch(2).front().group_set_id, task->cascade_descs[1].group_set_id);
    EXPECT_FALSE(result.has_value());

    deferred_engine->complete(2, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->primary_success);
    EXPECT_EQ(result->cascade_success, (std::vector<bool>{false, true}));
    EXPECT_EQ(terminal_count, 1u);
}

TEST_F(EvictionTaskRunnerTest, SubmitsSingleDescriptorVectorsInOrder) {
    auto task   = makeCopyTask();
    task.cascade_descs.push_back(TransferDescriptor::hostToDisk(2, 6, 7));

    const auto task_result = runImmediately(std::move(task));
    const auto descriptors = transfer_engine_->descriptors();

    EXPECT_TRUE(task_result.primary_success);
    EXPECT_EQ(task_result.cascade_success, (std::vector<bool>{true, true}));
    EXPECT_EQ(transfer_engine_->batchCount(), 3u);
    ASSERT_EQ(transfer_engine_->batches().size(), 3u);
    for (const auto& batch : transfer_engine_->batches()) {
        EXPECT_EQ(batch.size(), 1u);
    }
    ASSERT_EQ(descriptors.size(), 3u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].source_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[2].source_tier, Tier::HOST);
    EXPECT_EQ(descriptors[2].target_tier, Tier::DISK);
}

TEST_F(EvictionTaskRunnerTest, RunTransferOwnsTransferMetricsLifetime) {
    enableMetrics();
    auto deferred_engine = std::make_shared<DeferredTransferEngine>(group_sets_);
    BlockTransferDispatcher dispatcher(deferred_engine);
    EvictionTaskRunner runner(group_sets_, &dispatcher, 0, 0);
    auto task = std::make_shared<EvictionTask>(makeCopyTask());
    task->cascade_descs.clear();
    std::optional<EvictionTaskResult> result;

    runner.runTransfer(task, metrics_reporter_, [&](EvictionTaskResult completed) {
        result = std::move(completed);
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(evictionInFlight(Tier::DEVICE, Tier::HOST), 1);
    deferred_engine->complete(0, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->primary_success);
    EXPECT_EQ(evictionInFlight(Tier::DEVICE, Tier::HOST), 0);
}

TEST_F(EvictionTaskRunnerTest, BatchFailureFinishesTransferMetrics) {
    enableMetrics();
    transfer_engine_->enqueue(false);
    bool observed_in_flight = false;
    transfer_engine_->setOnSubmit([&]() { observed_in_flight = evictionInFlight(Tier::DEVICE, Tier::HOST) == 1; });

    const EvictionTaskResult result = runImmediately(makeCopyTask());

    EXPECT_FALSE(result.primary_success);
    EXPECT_TRUE(observed_in_flight);
    EXPECT_EQ(evictionInFlight(Tier::DEVICE, Tier::HOST), 0);
}

TEST_F(EvictionTaskRunnerTest, MalformedDescriptorFinishesTransferMetrics) {
    enableMetrics();
    auto task   = makeCopyTask();
    task.primary_desc.source_blocks.clear();

    const EvictionTaskResult result = runImmediately(std::move(task));

    EXPECT_FALSE(result.primary_success);
    EXPECT_EQ(transfer_engine_->batchCount(), 0u);
    EXPECT_EQ(evictionInFlight(Tier::DEVICE, Tier::HOST), 0);
}

TEST_F(EvictionTaskRunnerTest, SubmitExceptionFinishesTransferMetrics) {
    enableMetrics();
    bool observed_in_flight = false;
    transfer_engine_->setOnSubmit([&]() {
        observed_in_flight = evictionInFlight(Tier::DEVICE, Tier::HOST) == 1;
        throw std::runtime_error("injected submit failure");
    });

    EXPECT_NO_THROW({
        const EvictionTaskResult result = runImmediately(makeCopyTask());
        EXPECT_FALSE(result.primary_success);
    });
    EXPECT_TRUE(observed_in_flight);
    EXPECT_EQ(evictionInFlight(Tier::DEVICE, Tier::HOST), 0);
}

TEST_F(EvictionTaskRunnerTest, CascadeFailureDoesNotSkipLaterCascade) {
    transfer_engine_->enqueue(true);
    transfer_engine_->enqueue(false);
    transfer_engine_->enqueue(true);
    auto task = makeCopyTask();
    task.cascade_descs.push_back(TransferDescriptor::hostToDisk(2, 6, 7));

    const auto task_result = runImmediately(std::move(task));

    EXPECT_TRUE(task_result.primary_success);
    EXPECT_EQ(task_result.cascade_success, (std::vector<bool>{false, true}));
    EXPECT_EQ(transfer_engine_->batchCount(), 3u);
    EXPECT_EQ(transfer_engine_->descriptors().size(), 3u);
}

TEST_F(EvictionTaskRunnerTest, PrimaryFailureSkipsCascades) {
    transfer_engine_->enqueue(false);
    const auto task_result = runImmediately(makeCopyTask());

    EXPECT_FALSE(task_result.primary_success);
    EXPECT_EQ(task_result.cascade_success, (std::vector<bool>{false}));
    EXPECT_EQ(transfer_engine_->batchCount(), 1u);
}

TEST_F(EvictionTaskRunnerTest, RunTransferRejectsMalformedDescriptors) {
    const auto expect_rejected = [this](void (*mutate)(EvictionTask&)) {
        auto task = makeCopyTask();
        mutate(task);
        const EvictionTaskResult result = runImmediately(std::move(task));
        EXPECT_FALSE(result.primary_success);
        EXPECT_EQ(result.cascade_success, (std::vector<bool>{false}));
    };

    expect_rejected([](auto& task) { task.primary_desc.source_blocks = {}; });
    expect_rejected([](auto& task) { task.primary_desc.source_blocks = {1, NULL_BLOCK_IDX}; });
    expect_rejected([](auto& task) { task.primary_desc.target_blocks = {}; });
    expect_rejected([](auto& task) { task.primary_desc.target_blocks = {3, 4}; });
    expect_rejected([](auto& task) { task.primary_desc.source_tier = Tier::DISK; });
    expect_rejected([](auto& task) { task.cascade_descs[0].source_blocks = {4, 5}; });
    expect_rejected([](auto& task) { task.cascade_descs[0].target_blocks = {NULL_BLOCK_IDX}; });
}

TEST_F(EvictionTaskRunnerTest, DiskInvolvementSelectsDiskTransferTimeout) {
    auto task = makeCopyTask();
    task.cascade_descs.clear();
    EXPECT_EQ(EvictionTaskRunner::selectTransferTimeoutMs(task, 8000, 3000), 8000);

    task.primary_desc.target_tier = Tier::DISK;
    EXPECT_EQ(EvictionTaskRunner::selectTransferTimeoutMs(task, 8000, 3000), 3000);

    task.primary_desc.target_tier = Tier::HOST;
    task.cascade_descs.push_back(TransferDescriptor::hostToDisk(1, 4, 5));
    EXPECT_EQ(EvictionTaskRunner::selectTransferTimeoutMs(task, 8000, 3000), 3000);
}

TEST_F(EvictionTaskRunnerTest, RunTransferSupportsDeviceToDisk) {
    EvictionTask task;
    task.primary_desc.group_set_id  = 0;
    task.primary_desc.source_tier   = Tier::DEVICE;
    task.primary_desc.target_tier   = Tier::DISK;
    task.primary_desc.source_blocks = {1, 2};
    task.primary_desc.target_blocks = {3};

    const auto task_result = runImmediately(std::move(task));
    ASSERT_TRUE(task_result.primary_success);
    const auto descriptors = transfer_engine_->descriptors();
    ASSERT_EQ(descriptors.size(), 1u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[0].blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].singleBlockAt(Tier::DISK), 3);
}

}  // namespace
}  // namespace rtp_llm
