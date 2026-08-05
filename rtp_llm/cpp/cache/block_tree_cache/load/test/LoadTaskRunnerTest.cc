#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

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
    return makeTestGroupSet(0, topology, {0}, {std::move(pool)});
}

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

class CountingTransferEngine: public PerRankBlockTransferEngine {
public:
    explicit CountingTransferEngine(std::vector<GroupSetPtr> groups): PerRankBlockTransferEngine(std::move(groups)) {}

    std::shared_ptr<AsyncContext> submit(const TransferDescriptor& desc) override {
        ++submit_count;
        return PerRankBlockTransferEngine::submit(desc);
    }

    size_t submit_count{0};
};

TEST(LoadTaskRunnerTest, MissingSourcePoolIsRejectedByTransferEngine) {
    GroupSetPtr                   group  = makeTaskRunnerTestGroupSet();
    const std::vector<GroupSetPtr> group_sets{group};
    LoadTaskRunner                 runner(group_sets);
    auto                          engine = std::make_shared<CountingTransferEngine>(std::vector<GroupSetPtr>{group});
    BlockTransferDispatcher       dispatcher(engine);
    BlockTreeCacheMetricsReporter metrics_reporter;

    LoadTaskRunner::Task task;
    task.load_descs.push_back(TransferDescriptor::hostToDevice(0, 1, {1}));

    EXPECT_FALSE(runner.runTransfer(
        task, dispatcher, metrics_reporter, /*disk_timeout_ms=*/2000, /*host_timeout_ms=*/1000));
    EXPECT_EQ(engine->submit_count, 1u);
}

}  // namespace
}  // namespace rtp_llm
