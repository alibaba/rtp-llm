#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTaskRunner.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

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

TEST(LoadTaskRunnerTest, CreateTaskAllowsNoTransferItems) {
    LoadTaskRunner runner;
    GroupSetPtr    group = makeTaskRunnerTestGroupSet();

    LoadTicket::PendingLoadItem joined_item;
    joined_item.group_set_id                        = 0;
    joined_item.source_tier                         = Tier::HOST;
    joined_item.joined_load                         = true;
    const std::shared_ptr<LoadAsyncContext> context = std::make_shared<LoadAsyncContext>(1);
    LoadTaskRunner::TaskPtr                 task    = std::make_shared<LoadTaskRunner::Task>();

    ASSERT_TRUE(runner.createTask({joined_item}, {group}, context, task));
    EXPECT_EQ(task, nullptr);
}

}  // namespace
}  // namespace rtp_llm
