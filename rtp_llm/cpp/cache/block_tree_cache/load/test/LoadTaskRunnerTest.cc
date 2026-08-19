#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
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
    auto host_pool = makeHostPool(group.kv_block_stride_bytes, /*usable_count=*/2, /*enable_pinned=*/false);
    auto disk_pool = makeDiskPool(group.kv_block_stride_bytes,
                                  /*usable_count=*/2,
                                  "/tmp",
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    return makeTestGroupSet(
        group_set_id, topology, {0}, {std::move(pool)}, std::move(host_pool), std::move(disk_pool));
}

class SubmissionOrderContext final: public AsyncContext {
public:
    SubmissionOrderContext(std::vector<std::string>& events, std::string wait_event, bool succeed):
        events_(events), wait_event_(std::move(wait_event)), succeed_(succeed) {}

    void waitDone() override {
        events_.push_back(wait_event_);
        done_ = true;
    }
    bool done() const override {
        return done_;
    }
    bool success() const override {
        return done_ && succeed_;
    }

private:
    std::vector<std::string>& events_;
    std::string               wait_event_;
    bool                      succeed_;
    bool                      done_{false};
};

class RecordingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingPerRankEngine(std::deque<bool> results):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), results_(std::move(results)) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches_.push_back(descriptors);
        const TransferDescriptor& first = descriptors.front();
        const std::string direction = first.source_tier == Tier::HOST ? "host_" : "disk_";
        const std::string suffix = direction + std::to_string(first.group_set_id);
        events_.push_back("submit_" + suffix);
        const bool result = results_.front();
        results_.pop_front();
        return std::make_shared<SubmissionOrderContext>(events_, "wait_" + suffix, result);
    }

    std::vector<std::vector<TransferDescriptor>> batches_;
    std::deque<bool>                             results_;
    std::vector<std::string>                     events_;
};

TEST(LoadTaskRunnerTest, CreateTaskAllowsNoTransferDescriptors) {
    GroupSetPtr    group = makeTaskRunnerTestGroupSet();
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
    GroupSetPtr    group = makeTaskRunnerTestGroupSet();
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
    GroupSetPtr group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner runner(group_sets);
    auto engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true});
    BlockTransferDispatcher dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    LoadTaskRunner::Task task;
    task.load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}),
                       TransferDescriptor::diskToDevice(0, 2, {2})};

    EXPECT_TRUE(runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100));
    ASSERT_EQ(engine->batches_.size(), 2u);
    EXPECT_EQ(engine->batches_[0].front().source_tier, Tier::HOST);
    EXPECT_EQ(engine->batches_[1].front().source_tier, Tier::DISK);
    EXPECT_EQ(engine->events_,
              (std::vector<std::string>{"submit_host_0", "wait_host_0", "submit_disk_0", "wait_disk_0"}));
}

TEST(LoadTaskRunnerTest, HostFailureSkipsDiskBatch) {
    GroupSetPtr group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner runner(group_sets);
    auto engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{false});
    BlockTransferDispatcher dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    LoadTaskRunner::Task task;
    task.load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}),
                       TransferDescriptor::diskToDevice(0, 2, {2})};

    EXPECT_FALSE(runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100));
    EXPECT_EQ(engine->events_, (std::vector<std::string>{"submit_host_0", "wait_host_0"}));
}

TEST(LoadTaskRunnerTest, DiskFailureFailsLoadAfterHostSuccess) {
    GroupSetPtr group = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner runner(group_sets);
    auto engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, false});
    BlockTransferDispatcher dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    LoadTaskRunner::Task task;
    task.load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}),
                       TransferDescriptor::diskToDevice(0, 2, {2})};

    EXPECT_FALSE(runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100));
    EXPECT_EQ(engine->events_,
              (std::vector<std::string>{"submit_host_0", "wait_host_0", "submit_disk_0", "wait_disk_0"}));
}

TEST(LoadTaskRunnerTest, SplitsEachDirectionByGroupSetId) {
    const std::vector<GroupSetPtr> group_sets{makeTaskRunnerTestGroupSet(0), makeTaskRunnerTestGroupSet(1)};
    LoadTaskRunner runner(group_sets);
    auto engine = std::make_shared<RecordingPerRankEngine>(std::deque<bool>{true, true, true, true});
    BlockTransferDispatcher dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    LoadTaskRunner::Task task;
    task.load_descs = {TransferDescriptor::hostToDevice(0, 1, {1}),
                       TransferDescriptor::hostToDevice(0, 2, {2}),
                       TransferDescriptor::hostToDevice(1, 3, {3}),
                       TransferDescriptor::diskToDevice(0, 4, {4}),
                       TransferDescriptor::diskToDevice(1, 5, {5}),
                       TransferDescriptor::diskToDevice(1, 6, {6})};

    ASSERT_TRUE(runner.runTransfer(task, dispatcher, metrics_reporter, 100, 100));
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
              (std::vector<std::string>{"submit_host_0",
                                        "submit_host_1",
                                        "wait_host_0",
                                        "wait_host_1",
                                        "submit_disk_0",
                                        "submit_disk_1",
                                        "wait_disk_0",
                                        "wait_disk_1"}));
}

}  // namespace
}  // namespace rtp_llm
