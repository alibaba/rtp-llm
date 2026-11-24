#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/MockCacheHitController.h"

namespace rtp_llm {
namespace {
bool anyPrefix(size_t) {
    return true;
}

TEST(MockCacheHitControllerTest, RemoteDebtBeforeDiskMemory) {
    MockCacheHitController controller(0.25, 0.25, 0.5);
    const auto             first = controller.plan(3, anyPrefix);
    EXPECT_EQ(first.planned_remote_blocks, 1u);
    const auto second = controller.plan(1, anyPrefix);
    EXPECT_EQ(second.planned_remote_blocks, 1u);
    EXPECT_EQ(second.planned_disk_blocks, 0u);
    EXPECT_EQ(second.planned_memory_blocks, 0u);
    const auto third = controller.plan(1, anyPrefix);
    EXPECT_EQ(third.planned_disk_blocks, 1u);
    EXPECT_EQ(third.planned_memory_blocks, 0u);
    const auto fourth = controller.plan(1, anyPrefix);
    EXPECT_EQ(fourth.planned_remote_blocks, 1u);
    EXPECT_EQ(controller.plan(1, anyPrefix).planned_memory_blocks, 1u);
    for (size_t i = 7; i < 400; ++i) {
        const auto d = controller.plan(1, anyPrefix);
        EXPECT_EQ(d.gpuPrefixBlocks() + d.planned_memory_blocks + d.planned_disk_blocks + d.planned_remote_blocks, 1u);
    }
    const auto state = controller.snapshot();
    EXPECT_EQ(state.cumulative_eligible_blocks, 400u);
    EXPECT_EQ(state.cumulative_planned_remote_blocks, 200u);
    // The first admission left two GPU blocks; later singleton admissions
    // cannot retroactively recover them, leaving one block of debt in each lower tier.
    EXPECT_EQ(state.cumulative_planned_disk_blocks, 99u);
    EXPECT_EQ(state.cumulative_planned_memory_blocks, 99u);
}

TEST(MockCacheHitControllerTest, ThreeTierCumulativeTargetsAtBlockBoundaries) {
    MockCacheHitController controller(0.25, 0.25, 0.5);
    for (size_t i = 0; i < 100; ++i) {
        const auto d = controller.plan(4, anyPrefix);
        EXPECT_EQ(d.planned_remote_blocks, 2u);
        EXPECT_EQ(d.planned_disk_blocks, 1u);
        EXPECT_EQ(d.planned_memory_blocks, 1u);
        EXPECT_EQ(d.gpuPrefixBlocks(), 0u);
    }
    const auto state = controller.snapshot();
    EXPECT_EQ(state.cumulative_eligible_blocks, 400u);
    EXPECT_EQ(state.cumulative_planned_remote_blocks, 200u);
    EXPECT_EQ(state.cumulative_planned_disk_blocks, 100u);
    EXPECT_EQ(state.cumulative_planned_memory_blocks, 100u);
}

TEST(MockCacheHitControllerTest, ThreeTierSumRejected) {
    EXPECT_THROW((MockCacheHitController{0.25, 0.5, 0.5}), std::invalid_argument);
    for (double ratio :
         {-0.1, 1.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW((MockCacheHitController{0, 0, ratio}), std::invalid_argument);
    }
    MockCacheHitController rounded(0.0000005, 0.4999995, 0.5);
    const auto             d = rounded.plan(1'000'000, anyPrefix);
    EXPECT_EQ(d.planned_remote_blocks, 500'000u);
    EXPECT_EQ(d.planned_disk_blocks, 500'000u);
    EXPECT_EQ(d.planned_memory_blocks, 0u);
}

TEST(MockCacheHitControllerTest, RemoteOnlyInfeasibleGpuPrefixExpandsRemote) {
    for (const double ratio : {0.0, 0.5, 1.0}) {
        MockCacheHitController controller(0, 0, ratio);
        const auto             d = controller.plan(8, anyPrefix);
        EXPECT_EQ(d.planned_remote_blocks, static_cast<size_t>(8 * ratio));
        EXPECT_EQ(d.gpuPrefixBlocks(), 8 - d.planned_remote_blocks);
    }
    MockCacheHitController controller(0, 0, 0.5);
    const auto             d = controller.plan(8, [](size_t gpu) { return gpu == 0; });
    EXPECT_EQ(d.planned_remote_blocks, 8u);
    EXPECT_EQ(d.planned_memory_blocks + d.planned_disk_blocks, 0u);
    EXPECT_EQ(controller.plan(8, anyPrefix).planned_remote_blocks, 0u);
}

TEST(MockCacheHitControllerTest, EffectiveZeroDoesNotTakeExpansion) {
    MockCacheHitController remote(0.0000004, 0.0000004, 0.5);
    EXPECT_EQ(remote.plan(8, [](size_t n) { return n == 0; }).planned_remote_blocks, 8u);
    MockCacheHitController disk(0.0000004, 0.5, 0.0000004);
    const auto             d = disk.plan(8, [](size_t n) { return n == 0; });
    EXPECT_EQ(d.planned_disk_blocks, 8u);
    EXPECT_EQ(d.planned_remote_blocks, 0u);
    MockCacheHitController memory(0.25, 0.25, 0.25);
    const auto             m = memory.plan(8, [](size_t n) { return n == 0; });
    EXPECT_EQ(m.planned_memory_blocks, 4u);
    EXPECT_EQ(m.planned_disk_blocks, 2u);
    EXPECT_EQ(m.planned_remote_blocks, 2u);
    MockCacheHitController disabled(0.0000004, 0.0000004, 0.0000004);
    EXPECT_FALSE(disabled.hasActiveRatios());
    EXPECT_NO_THROW(disabled.plan(8, [](size_t) -> bool { throw std::runtime_error("disabled"); }));
}

TEST(MockCacheHitControllerTest, RemoteFailureAndOverflowRollback) {
    MockCacheHitController controller(0.25, 0.25, 0.5);
    EXPECT_THROW(controller.plan(4, [](size_t) { return false; }), std::invalid_argument);
    EXPECT_THROW(controller.plan(4, [](size_t) -> bool { throw std::runtime_error("validator"); }), std::runtime_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_remote_blocks, 0u);
    const auto maximum = std::numeric_limits<uint64_t>::max();
    const auto d       = controller.plan(maximum, anyPrefix);
    EXPECT_EQ(d.admissionOffsetBlocks(), 0u);
    EXPECT_EQ(d.planned_remote_blocks, maximum / 2);
    EXPECT_THROW(controller.plan(1, anyPrefix), std::overflow_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, maximum);
    EXPECT_EQ(controller.snapshot().cumulative_planned_remote_blocks, maximum / 2);
}

TEST(MockCacheHitControllerTest, LegacyAggregateInitializersKeepTheirMeaning) {
    const MockCacheHitDecision d{8, 2, 3, 18, 4, 6};
    EXPECT_EQ(d.cumulative_eligible_blocks, 18u);
    EXPECT_EQ(d.cumulative_planned_memory_blocks, 4u);
    EXPECT_EQ(d.cumulative_planned_disk_blocks, 6u);
    EXPECT_EQ(d.planned_remote_blocks, 0u);
    EXPECT_EQ(d.gpuPrefixBlocks(), 3u);
    EXPECT_EQ(d.admissionOffsetBlocks(), 10u);
    const MockCacheHitSnapshot s{18, 4, 6};
    EXPECT_EQ(s.cumulative_planned_disk_blocks, 6u);
    EXPECT_EQ(s.cumulative_planned_remote_blocks, 0u);
}

TEST(MockCacheHitControllerTest, MixedCumulativeTargetsForOneHundredThreeBlockAdmissions) {
    MockCacheHitController controller(0.25, 0.5);
    for (size_t i = 0; i < 100; ++i) {
        const auto decision = controller.plan(3, anyPrefix);
        EXPECT_LE(decision.planned_memory_blocks + decision.planned_disk_blocks, 3u);
        EXPECT_EQ(decision.gpuPrefixBlocks() + decision.planned_memory_blocks + decision.planned_disk_blocks, 3u);
    }
    const auto state = controller.snapshot();
    EXPECT_EQ(state.cumulative_eligible_blocks, 300u);
    EXPECT_EQ(state.cumulative_planned_memory_blocks, 75u);
    EXPECT_EQ(state.cumulative_planned_disk_blocks, 150u);
}

TEST(MockCacheHitControllerTest, DiskOnlyZeroOneAndBlockRounding) {
    for (const auto& [ratio, expected] : {std::pair<double, size_t>{0.0, 0}, {0.333333, 333'333}, {1.0, 1'000'000}}) {
        MockCacheHitController controller(0, ratio);
        const auto             decision = controller.plan(1'000'000, anyPrefix);
        EXPECT_EQ(decision.planned_disk_blocks, expected);
        EXPECT_EQ(decision.planned_memory_blocks, 0u);
    }
    MockCacheHitController rounded(0.3333336, 0.3333334);
    const auto             decision = rounded.plan(1'000'000, anyPrefix);
    EXPECT_EQ(decision.planned_memory_blocks, 333'334u);
    EXPECT_EQ(decision.planned_disk_blocks, 333'333u);
}

TEST(MockCacheHitControllerTest, QuantizedZeroNeverForcesReadsOrExpandsIntoThatTier) {
    MockCacheHitController disabled(0.0000004, 0.0000004);
    EXPECT_FALSE(disabled.hasActiveRatios());
    EXPECT_NO_THROW(disabled.plan(8, [](size_t) -> bool { throw std::runtime_error("ordinary routing required"); }));
    MockCacheHitController disk_only(0.0000004, 0.5);
    EXPECT_TRUE(disk_only.hasActiveRatios());
    const auto disk = disk_only.plan(8, [](size_t gpu) { return gpu == 0; });
    EXPECT_EQ(disk.planned_memory_blocks, 0u);
    EXPECT_EQ(disk.planned_disk_blocks, 8u);
    MockCacheHitController memory_only(0.5, 0.0000004);
    const auto             memory = memory_only.plan(8, [](size_t gpu) { return gpu == 0; });
    EXPECT_EQ(memory.planned_memory_blocks, 8u);
    EXPECT_EQ(memory.planned_disk_blocks, 0u);
}

TEST(MockCacheHitControllerTest, ComplementaryHalfMillionthsCannotRoundTotalAboveOne) {
    // Each ratio rounds upward independently: 1 ppm + 1,000,000 ppm.
    // Disk priority clamps effective memory to zero, preserving a unit total.
    MockCacheHitController controller(0.0000005, 0.9999995);
    const auto             decision = controller.plan(1'000'000, anyPrefix);
    EXPECT_EQ(decision.planned_memory_blocks, 0u);
    EXPECT_EQ(decision.planned_disk_blocks, 1'000'000u);
}

TEST(MockCacheHitControllerTest, SimultaneousSingleBlockDeficitsPrioritizeDiskAndCarryMemory) {
    MockCacheHitController controller(0.5, 0.5);
    EXPECT_EQ(controller.plan(1, anyPrefix).planned_disk_blocks, 0u);
    const auto second = controller.plan(1, anyPrefix);
    EXPECT_EQ(second.planned_disk_blocks, 1u);
    EXPECT_EQ(second.planned_memory_blocks, 0u);
    const auto third = controller.plan(1, anyPrefix);
    EXPECT_EQ(third.planned_disk_blocks, 0u);
    EXPECT_EQ(third.planned_memory_blocks, 1u);
    controller.plan(1, anyPrefix);
    controller.plan(1, anyPrefix);
    EXPECT_EQ(controller.snapshot().cumulative_planned_disk_blocks, 2u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_memory_blocks, 2u);
}

TEST(MockCacheHitControllerTest, RejectsInvalidRatiosAndQuantizedSumOverflow) {
    for (const double ratio : {-0.1,
                               1.1,
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW((MockCacheHitController{ratio, 0}), std::invalid_argument);
        EXPECT_THROW((MockCacheHitController{0, ratio}), std::invalid_argument);
    }
    EXPECT_THROW((MockCacheHitController{0.6, 0.5}), std::invalid_argument);
    MockCacheHitController rounded(0.5000005, 0.4999995);
    const auto             decision = rounded.plan(1'000'000, anyPrefix);
    EXPECT_LE(decision.planned_memory_blocks + decision.planned_disk_blocks, 1'000'000u);
}

TEST(MockCacheHitControllerTest, InfeasibleGpuPrefixExpandsMemoryAndCarriesOvershoot) {
    MockCacheHitController controller(0.25, 0.5);
    const auto             first = controller.plan(8, [](size_t gpu) { return gpu == 0; });
    EXPECT_EQ(first.planned_disk_blocks, 4u);
    EXPECT_EQ(first.planned_memory_blocks, 4u);
    EXPECT_EQ(first.gpuPrefixBlocks(), 0u);
    const auto second = controller.plan(8, anyPrefix);
    EXPECT_EQ(second.planned_disk_blocks, 4u);
    EXPECT_EQ(second.planned_memory_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_memory_blocks, 4u);
}

TEST(MockCacheHitControllerTest, DiskOnlyExpansionNeverUsesZeroMemoryTier) {
    MockCacheHitController controller(0, 0.5);
    const auto             decision = controller.plan(8, [](size_t gpu) { return gpu == 0; });
    EXPECT_EQ(decision.planned_disk_blocks, 8u);
    EXPECT_EQ(decision.planned_memory_blocks, 0u);
    EXPECT_EQ(controller.plan(8, anyPrefix).planned_disk_blocks, 0u);
}

TEST(MockCacheHitControllerTest, AllZeroSkipsForcedReadsAndColdPrefixIsEmpty) {
    MockCacheHitController disabled(0, 0);
    const auto natural = disabled.plan(8, [](size_t) -> bool { throw std::runtime_error("must not force sources"); });
    EXPECT_EQ(natural.planned_memory_blocks + natural.planned_disk_blocks, 0u);
    MockCacheHitController controller(0.25, 0.5);
    const auto             cold = controller.plan(0, anyPrefix);
    EXPECT_EQ(cold.eligible_blocks, 0u);
    EXPECT_EQ(cold.planned_memory_blocks + cold.planned_disk_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 0u);
}

TEST(MockCacheHitControllerTest, FailedFeasibilityAndThrowingValidatorDoNotCharge) {
    MockCacheHitController controller(0.25, 0.5);
    EXPECT_THROW(controller.plan(4, [](size_t) { return false; }), std::invalid_argument);
    EXPECT_THROW(controller.plan(4, [](size_t) -> bool { throw std::runtime_error("failed"); }), std::runtime_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_memory_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_disk_blocks, 0u);
    const auto decision = controller.plan(4, anyPrefix);
    EXPECT_EQ(decision.planned_memory_blocks, 1u);
    EXPECT_EQ(decision.planned_disk_blocks, 2u);
}

TEST(MockCacheHitControllerTest, WideArithmeticAndOverflowPreserveCounters) {
    MockCacheHitController controller(0.25, 0.5);
    const auto             maximum  = std::numeric_limits<uint64_t>::max();
    const auto             decision = controller.plan(maximum, anyPrefix);
    EXPECT_EQ(decision.planned_disk_blocks, maximum / 2);
    EXPECT_EQ(decision.planned_memory_blocks, maximum / 4);
    EXPECT_THROW(controller.plan(1, anyPrefix), std::overflow_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, maximum);
    EXPECT_EQ(controller.plan(0, anyPrefix).eligible_blocks, 0u);
}
}  // namespace
}  // namespace rtp_llm
