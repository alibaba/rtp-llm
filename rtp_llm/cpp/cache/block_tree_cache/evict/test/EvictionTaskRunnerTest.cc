#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {
namespace {

using ScriptedTransferEngine = block_tree_cache_test::ScriptedPerRankBlockTransferEngine;

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
        ++batch_count_;
        return ScriptedTransferEngine::submit(descriptors);
    }

    size_t batchCount() const {
        return batch_count_;
    }

private:
    size_t batch_count_{0};
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
        return EvictionTaskRunner(group_sets_,
                                  transfer_dispatcher_.get(),
                                  nullptr,
                                  metrics_reporter_,
                                  mutex_,
                                  0,
                                  0,
                                  [](Tier) { return true; },
                                  [](bool, bool) {});
    }

    std::vector<GroupSetPtr>                 group_sets_;
    std::shared_ptr<BatchCountingTransferEngine> transfer_engine_;
    std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher_;
    BlockTreeCacheMetricsReporter            metrics_reporter_;
    std::mutex                               mutex_;
};

BlockTreeEvictor::EvictionPlan makeCopyPlan() {
    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc.group_set_id  = 0;
    plan.primary_desc.source_tier   = Tier::DEVICE;
    plan.primary_desc.target_tier   = Tier::HOST;
    plan.primary_desc.source_blocks = {1, 2};
    plan.primary_desc.target_blocks = {3};

    TransferDescriptor cascade_desc;
    cascade_desc.group_set_id  = 1;
    cascade_desc.source_tier   = Tier::HOST;
    cascade_desc.target_tier   = Tier::DISK;
    cascade_desc.source_blocks = {4};
    cascade_desc.target_blocks = {5};
    plan.cascade_descs.push_back(std::move(cascade_desc));
    return plan;
}

TEST_F(EvictionTaskRunnerTest, PerformCopySubmitsPlanAsDirectionBatches) {
    auto runner = makeRunner();
    auto plan   = makeCopyPlan();
    plan.cascade_descs.push_back(TransferDescriptor::hostToDisk(2, 6, 7));

    const auto results = runner.performCopy(plan);
    const auto descriptors = transfer_engine_->descriptors();

    EXPECT_TRUE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{true, true}));
    EXPECT_EQ(transfer_engine_->batchCount(), 2u);
    ASSERT_EQ(descriptors.size(), 3u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].source_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[2].source_tier, Tier::HOST);
    EXPECT_EQ(descriptors[2].target_tier, Tier::DISK);
}

TEST_F(EvictionTaskRunnerTest, HostDiskBatchesAreSplitByDiskPool) {
    group_sets_ = {makeRunnerTestGroupSet(0, "eviction_runner_0", makeRunnerTestDiskPool("eviction_runner_0_disk")),
                   makeRunnerTestGroupSet(1, "eviction_runner_1", makeRunnerTestDiskPool("eviction_runner_1_disk"))};
    auto runner = makeRunner();
    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc = TransferDescriptor::hostToDisk(0, 1, 2);
    plan.cascade_descs = {TransferDescriptor::hostToDisk(0, 3, 4),
                          TransferDescriptor::hostToDisk(1, 5, 6)};

    const auto results = runner.performCopy(plan);

    EXPECT_TRUE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{true, true}));
    EXPECT_EQ(transfer_engine_->batchCount(), 2u);
    EXPECT_EQ(transfer_engine_->descriptors().size(), 3u);
}

TEST_F(EvictionTaskRunnerTest, TransferBatchPartitionGroupsByDirectionAndDiskPool) {
    group_sets_ = {makeRunnerTestGroupSet(0, "eviction_runner_0", makeRunnerTestDiskPool("eviction_runner_0_disk")),
                   makeRunnerTestGroupSet(1, "eviction_runner_1", makeRunnerTestDiskPool("eviction_runner_1_disk"))};
    auto runner = makeRunner();
    std::vector<TransferDescriptor> descriptors = {
        TransferDescriptor::deviceToHost(0, {1}, 2),
        TransferDescriptor::hostToDisk(0, 3, 4),
        TransferDescriptor::hostToDisk(0, 5, 6),
        TransferDescriptor::hostToDisk(1, 7, 8),
    };

    const auto batches = runner.partitionTransferBatch(descriptors);

    ASSERT_EQ(batches.size(), 3u);
    EXPECT_EQ(batches[0].size(), 1u);
    EXPECT_EQ(batches[1].size(), 2u);
    EXPECT_EQ(batches[2].size(), 1u);
}

TEST_F(EvictionTaskRunnerTest, DirectionBatchFailureFailsTheWholePlan) {
    transfer_engine_->enqueue(false);
    auto runner = makeRunner();

    const auto results = runner.performCopy(makeCopyPlan());

    EXPECT_FALSE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false}));
    EXPECT_EQ(transfer_engine_->batchCount(), 2u);
}

TEST_F(EvictionTaskRunnerTest, BuildTransferBatchIncludesPrimaryAndCascades) {
    std::vector<TransferDescriptor> descriptors;

    ASSERT_TRUE(EvictionTaskRunner::buildTransferBatch(makeCopyPlan(), descriptors));

    ASSERT_EQ(descriptors.size(), 2u);
    EXPECT_EQ(descriptors[0].blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].singleBlockAt(Tier::HOST), 3);
    EXPECT_EQ(descriptors[1].singleBlockAt(Tier::HOST), 4);
    EXPECT_EQ(descriptors[1].singleBlockAt(Tier::DISK), 5);
}

TEST_F(EvictionTaskRunnerTest, BuildTransferBatchRejectsMalformedDescriptors) {
    const auto expect_rejected = [](void (*mutate)(BlockTreeEvictor::EvictionPlan&)) {
        auto plan = makeCopyPlan();
        mutate(plan);
        std::vector<TransferDescriptor> descriptors;
        EXPECT_FALSE(EvictionTaskRunner::buildTransferBatch(plan, descriptors));
        EXPECT_TRUE(descriptors.empty());
    };

    expect_rejected([](auto& plan) { plan.primary_desc.source_blocks = {}; });
    expect_rejected([](auto& plan) { plan.primary_desc.source_blocks = {1, NULL_BLOCK_IDX}; });
    expect_rejected([](auto& plan) { plan.primary_desc.target_blocks = {}; });
    expect_rejected([](auto& plan) { plan.primary_desc.target_blocks = {3, 4}; });
    expect_rejected([](auto& plan) { plan.primary_desc.source_tier = Tier::DISK; });
    expect_rejected([](auto& plan) { plan.cascade_descs[0].source_blocks = {4, 5}; });
    expect_rejected([](auto& plan) { plan.cascade_descs[0].target_blocks = {NULL_BLOCK_IDX}; });
}

TEST_F(EvictionTaskRunnerTest, DiskInvolvedPlanUsesDiskTimeoutEvenWhenSmaller) {
    auto plan = makeCopyPlan();
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 3000);

    plan.primary_desc.target_tier = Tier::DISK;
    plan.cascade_descs.clear();
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 3000);

    plan.primary_desc.target_tier = Tier::HOST;
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 8000);
}

TEST_F(EvictionTaskRunnerTest, DeviceEvictionBypassesDisabledHost) {
    auto runner = makeRunner();
    runner.is_tier_enabled_ = [](Tier tier) { return tier == Tier::DISK; };

    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc.group_set_id  = 0;
    plan.primary_desc.source_tier   = Tier::DEVICE;
    plan.primary_desc.target_tier   = runner.normalizeTargetTier(Tier::DEVICE);
    plan.primary_desc.source_blocks = {1, 2};
    plan.primary_desc.target_blocks = {3};

    const auto results = runner.performCopy(plan);
    ASSERT_TRUE(results.primary_success);
    const auto descriptors = transfer_engine_->descriptors();
    ASSERT_EQ(descriptors.size(), 1u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[0].blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].singleBlockAt(Tier::DISK), 3);
}

TEST_F(EvictionTaskRunnerTest, TargetNormalizationUsesNearestEnabledTier) {
    auto runner = makeRunner();

    runner.is_tier_enabled_ = [](Tier tier) { return tier == Tier::HOST || tier == Tier::DISK; };
    EXPECT_EQ(runner.normalizeTargetTier(Tier::DEVICE), Tier::HOST);
    EXPECT_EQ(runner.normalizeTargetTier(Tier::HOST), Tier::DISK);

    runner.is_tier_enabled_ = [](Tier tier) { return tier == Tier::DISK; };
    EXPECT_EQ(runner.normalizeTargetTier(Tier::DEVICE), Tier::DISK);

    runner.is_tier_enabled_ = [](Tier) { return false; };
    EXPECT_EQ(runner.normalizeTargetTier(Tier::DEVICE), Tier::NONE);
    EXPECT_EQ(runner.normalizeTargetTier(Tier::HOST), Tier::NONE);
    EXPECT_EQ(runner.normalizeTargetTier(Tier::DISK), Tier::NONE);
}

}  // namespace
}  // namespace rtp_llm
