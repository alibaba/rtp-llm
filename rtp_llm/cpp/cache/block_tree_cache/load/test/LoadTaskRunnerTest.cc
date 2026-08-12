#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

GroupSetPtr makeTaskRunnerTestGroupSet() {
    using namespace block_transfer_engine_test;

    auto policy                                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse                          = true;
    const GroupBase                            group    = makeTestGroupBase(policy);
    const std::shared_ptr<const CacheTopology> topology = makeTestTopology({group});
    DeviceBlockPoolPtr pool = makeTestDevicePool({{group.kv_block_stride_bytes, group.kv_scale_stride_bytes}},
                                                 /*usable_count=*/1,
                                                 "load_task_runner");
    auto host_pool = makeHostPool(group.kv_block_stride_bytes, /*usable_count=*/2, /*enable_pinned=*/false);
    auto disk_pool = makeDiskPool(group.kv_block_stride_bytes,
                                  /*usable_count=*/2,
                                  "/tmp",
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    return makeTestGroupSet(0, topology, {0}, {std::move(pool)}, std::move(host_pool), std::move(disk_pool));
}

class SubmissionOrderContext final: public AsyncContext {
public:
    SubmissionOrderContext(std::atomic<size_t>& submit_count,
                           bool succeed,
                           std::shared_ptr<std::atomic<bool>> all_submitted_before_wait):
        submit_count_(submit_count),
        succeed_(succeed),
        all_submitted_before_wait_(std::move(all_submitted_before_wait)) {}

    void waitDone() override {
        all_submitted_before_wait_->store(submit_count_.load() == 2);
        done_ = true;
    }
    bool done() const override {
        return done_;
    }
    bool success() const override {
        return done_ && succeed_;
    }

private:
    std::atomic<size_t>& submit_count_;
    bool                 succeed_;
    std::shared_ptr<std::atomic<bool>> all_submitted_before_wait_;
    bool                 done_{false};
};

class RecordingPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit RecordingPerRankEngine(std::deque<bool> results):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), results_(std::move(results)) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        batches_.push_back(descriptors);
        ++submit_count_;
        const bool result = results_.front();
        results_.pop_front();
        waits_after_all_submits_.push_back(std::make_shared<std::atomic<bool>>(false));
        return std::make_shared<SubmissionOrderContext>(
            submit_count_, result, waits_after_all_submits_.back());
    }

    std::vector<std::vector<TransferDescriptor>> batches_;
    std::deque<bool>                             results_;
    std::deque<std::shared_ptr<std::atomic<bool>>> waits_after_all_submits_;
    std::atomic<size_t>                          submit_count_{0};
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

TEST(LoadTaskRunnerTest, HostAndDiskBatchesAreSubmittedBeforeUnifiedWait) {
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
    EXPECT_TRUE(engine->waits_after_all_submits_[0]->load());
    EXPECT_TRUE(engine->waits_after_all_submits_[1]->load());
}

TEST(LoadTaskRunnerTest, AnyDirectionFailureFailsTheLoadBatch) {
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
}

}  // namespace
}  // namespace rtp_llm
