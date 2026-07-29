#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

namespace rtp_llm {
namespace {

BlockTreeEvictor::EvictionPlan makeCopyPlan() {
    BlockTreeEvictor::EvictionPlan plan;
    plan.primary.group_set_id  = 0;
    plan.primary.source_tier   = Tier::DEVICE;
    plan.primary.target_tier   = Tier::HOST;
    plan.primary.source_blocks = {1, 2};
    plan.primary.target_blocks = {3};

    EvictionMove cascade;
    cascade.group_set_id  = 1;
    cascade.source_tier   = Tier::HOST;
    cascade.target_tier   = Tier::DISK;
    cascade.source_blocks = {4};
    cascade.target_blocks = {5};
    plan.cascade_moves.push_back(std::move(cascade));
    return plan;
}

TEST(EvictionTaskRunnerTest, PerformCopyExecutesPrimaryAndCascadesInOrder) {
    std::vector<TransferDescriptor> descriptors;
    EvictionTaskRunner              runner([&descriptors](const TransferDescriptor& descriptor) {
        descriptors.push_back(descriptor);
        return true;
    });

    const auto results = runner.performCopy(makeCopyPlan());

    EXPECT_TRUE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{true}));
    ASSERT_EQ(descriptors.size(), 2u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].source_tier, Tier::HOST);
    EXPECT_EQ(descriptors[1].target_tier, Tier::DISK);
}

TEST(EvictionTaskRunnerTest, PrimaryFailureSuppressesCascadeCopy) {
    size_t             execution_count = 0;
    EvictionTaskRunner runner([&execution_count](const TransferDescriptor&) {
        ++execution_count;
        return false;
    });

    const auto results = runner.performCopy(makeCopyPlan());

    EXPECT_FALSE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false}));
    EXPECT_EQ(execution_count, 1u);
}

TEST(EvictionTaskRunnerTest, BuildTransferBatchIncludesPrimaryAndCascades) {
    std::vector<TransferDescriptor> descriptors;

    ASSERT_TRUE(EvictionTaskRunner::buildTransferBatch(makeCopyPlan(), descriptors));

    ASSERT_EQ(descriptors.size(), 2u);
    EXPECT_EQ(descriptors[0].device_blocks, (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].host_block, 3);
    EXPECT_EQ(descriptors[1].host_block, 4);
    EXPECT_EQ(descriptors[1].disk_block, 5);
}

TEST(EvictionTaskRunnerTest, DiskInvolvedPlanUsesDiskTimeoutEvenWhenSmaller) {
    auto plan = makeCopyPlan();
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 3000);

    plan.primary.target_tier = Tier::DISK;
    plan.cascade_moves.clear();
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 3000);

    plan.primary.target_tier = Tier::HOST;
    EXPECT_EQ(EvictionTaskRunner::transferTimeoutMs(plan, 8000, 3000), 8000);
}

TEST(EvictionTaskRunnerTest, DeviceEvictionBypassesDisabledHost) {
    std::vector<TransferDescriptor> descriptors;
    EvictionTaskRunner              runner([&descriptors](const TransferDescriptor& descriptor) {
        descriptors.push_back(descriptor);
        return true;
    });
    runner.is_tier_enabled_ = [](Tier tier) { return tier == Tier::DISK; };

    BlockTreeEvictor::EvictionPlan plan;
    plan.primary.group_set_id  = 0;
    plan.primary.source_tier   = Tier::DEVICE;
    plan.primary.target_tier   = runner.normalizeTargetTier(Tier::DEVICE);
    plan.primary.source_blocks = {1, 2};
    plan.primary.target_blocks = {3};

    const auto results = runner.performCopy(plan);
    ASSERT_TRUE(results.primary_success);
    ASSERT_EQ(descriptors.size(), 1u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[0].device_blocks, (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].disk_block, 3);
}

TEST(EvictionTaskRunnerTest, TargetNormalizationUsesNearestEnabledTier) {
    EvictionTaskRunner runner([](const TransferDescriptor&) { return true; });

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
