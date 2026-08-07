#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"

namespace rtp_llm {
namespace {

class EvictionTaskRunnerTest: public ::testing::Test {
protected:
    EvictionTaskRunner makeRunner(EvictionTaskRunner::ExecuteTransferFn execute_transfer) {
        return EvictionTaskRunner(std::move(execute_transfer),
                                  group_sets_,
                                  nullptr,
                                  nullptr,
                                  metrics_reporter_,
                                  mutex_,
                                  0,
                                  0,
                                  [](Tier) { return true; },
                                  [](bool, bool) {},
                                  [](CacheKeyType, size_t) {});
    }

    std::vector<GroupSetPtr>      group_sets_;
    BlockTreeCacheMetricsReporter metrics_reporter_;
    std::mutex                    mutex_;
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

TEST_F(EvictionTaskRunnerTest, PerformCopyExecutesPrimaryAndCascadesInOrder) {
    std::vector<TransferDescriptor> descriptors;
    auto runner = makeRunner([&descriptors](const TransferDescriptor& descriptor) {
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

TEST_F(EvictionTaskRunnerTest, PrimaryFailureSuppressesCascadeCopy) {
    size_t execution_count = 0;
    auto runner = makeRunner([&execution_count](const TransferDescriptor&) {
        ++execution_count;
        return false;
    });

    const auto results = runner.performCopy(makeCopyPlan());

    EXPECT_FALSE(results.primary_success);
    EXPECT_EQ(results.cascade_success, (std::vector<bool>{false}));
    EXPECT_EQ(execution_count, 1u);
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
    std::vector<TransferDescriptor> descriptors;
    auto runner = makeRunner([&descriptors](const TransferDescriptor& descriptor) {
        descriptors.push_back(descriptor);
        return true;
    });
    runner.is_tier_enabled_ = [](Tier tier) { return tier == Tier::DISK; };

    BlockTreeEvictor::EvictionPlan plan;
    plan.primary_desc.group_set_id  = 0;
    plan.primary_desc.source_tier   = Tier::DEVICE;
    plan.primary_desc.target_tier   = runner.normalizeTargetTier(Tier::DEVICE);
    plan.primary_desc.source_blocks = {1, 2};
    plan.primary_desc.target_blocks = {3};

    const auto results = runner.performCopy(plan);
    ASSERT_TRUE(results.primary_success);
    ASSERT_EQ(descriptors.size(), 1u);
    EXPECT_EQ(descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(descriptors[0].target_tier, Tier::DISK);
    EXPECT_EQ(descriptors[0].blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{1, 2}));
    EXPECT_EQ(descriptors[0].singleBlockAt(Tier::DISK), 3);
}

TEST_F(EvictionTaskRunnerTest, TargetNormalizationUsesNearestEnabledTier) {
    auto runner = makeRunner([](const TransferDescriptor&) { return true; });

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
