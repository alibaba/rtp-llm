#include "rtp_llm/cpp/cache/MlaHostCacheBudget.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(MlaHostCacheBudgetTest, PrefillChargesIndexerAndDraftAndCapsStagingWindow) {
    LayeredMlaPrefillBudget input;
    input.device_bytes               = 460;
    input.host_bytes                 = 200;
    input.all_device_bytes_per_block = 60;  // Four MLA layers plus 20 bytes of device-only data.
    input.layer_bytes_per_block      = 10;
    input.main_layers                = 4;
    input.max_logical_blocks         = 100;
    input.max_staging_blocks         = 2;
    input.requested_staging_layers   = 4;

    const auto plan = planLayeredMlaPrefill(input);
    EXPECT_EQ(plan.logical_blocks, 10);
    EXPECT_EQ(plan.prefill_hbm_layers, 2);
    EXPECT_EQ(plan.prefill_staging_blocks, 2);
    EXPECT_EQ(plan.prefill_staging_layers, 2);
}

TEST(MlaHostCacheBudgetTest, PrefillCanStageEveryLogicalPageWhenNotCappedByAdmission) {
    LayeredMlaPrefillBudget input{440, 1000, 60, 10, 4, 100, 100, 2};
    const auto              plan = planLayeredMlaPrefill(input);
    ASSERT_TRUE(plan.layeredPrefill());
    EXPECT_EQ(plan.prefill_staging_blocks, plan.logical_blocks);
    EXPECT_EQ(plan.logical_blocks, 8);
}

TEST(MlaHostCacheBudgetTest, PrefillTiePrefersDeviceLayers) {
    LayeredMlaPrefillBudget input{120, 1000, 30, 10, 3, 4, 1, 2};
    const auto              plan = planLayeredMlaPrefill(input);
    EXPECT_EQ(plan.logical_blocks, 4);
    EXPECT_EQ(plan.prefill_hbm_layers, 2);
    EXPECT_EQ(plan.prefill_staging_layers, 1);
}

TEST(MlaHostCacheBudgetTest, PrefillRejectsHostCapacityWithoutAUsableBlock) {
    LayeredMlaPrefillBudget input{1024, 10, 30, 10, 3, 100, 10, 2};
    EXPECT_FALSE(planLayeredMlaPrefill(input).layeredPrefill());
    input.host_bytes            = 1000;
    input.layer_bytes_per_block = std::numeric_limits<size_t>::max();
    EXPECT_FALSE(planLayeredMlaPrefill(input).layeredPrefill());
}

TEST(MlaHostCacheBudgetTest, AutomaticPrefillPreservesOrdinaryCapacityWhenHostBudgetIsSmall) {
    const auto plan = planLayeredMlaPrefill({6000, 200, 60, 10, 4, 100, 100, 4});
    ASSERT_TRUE(plan.layeredPrefill());
    ASSERT_EQ(plan.logical_blocks, 20);
    const int32_t candidates[] = {
        100, static_cast<int32_t>(plan.logical_blocks), static_cast<int32_t>(plan.prefill_staging_layers)};
    const auto capacities = minimumMlaPrefillCapacities(candidates, 1);
    EXPECT_EQ(capacities.ordinary_blocks, 100);
    EXPECT_FALSE(capacities.useLayered());
}

TEST(MlaHostCacheBudgetTest, AutomaticPrefillChoosesUsingBothStageMinimums) {
    const int32_t larger_stage_capacity[] = {100, 60, 4, 50, 70, 1, 80, 65, 3};
    const auto    capacities              = minimumMlaPrefillCapacities(larger_stage_capacity, 3);
    EXPECT_EQ(capacities.ordinary_blocks, 50);
    EXPECT_EQ(capacities.layered_blocks, 60);
    EXPECT_EQ(capacities.staging_layers, 1);
    EXPECT_TRUE(capacities.useLayered());

    const int32_t tied_capacity[] = {100, 50, 4, 50, 70, 1};
    EXPECT_FALSE(minimumMlaPrefillCapacities(tied_capacity, 2).useLayered());
    const int32_t unsupported_rank[] = {100, 0, 0, 50, 70, 4};
    EXPECT_FALSE(minimumMlaPrefillCapacities(unsupported_rank, 2).useLayered());

    const int32_t no_ordinary_capacity[] = {100, 60, 4, 0, 70, 1};
    const auto    layered_only           = minimumMlaPrefillCapacities(no_ordinary_capacity, 2);
    EXPECT_EQ(layered_only.ordinary_blocks, 0);
    EXPECT_EQ(layered_only.layered_blocks, 60);
    EXPECT_EQ(layered_only.staging_layers, 1);
    EXPECT_TRUE(layered_only.useLayered());
}

TEST(MlaHostCacheBudgetTest, ExplicitLayeredPrefillSharesMinimumStagingDepth) {
    const int32_t explicit_layered[] = {60, 0, 4, 50, 0, 1, 70, 0, 3};
    const auto    capacities         = minimumMlaPrefillCapacities(explicit_layered, 3);
    EXPECT_EQ(capacities.ordinary_blocks, 50);
    EXPECT_EQ(capacities.staging_layers, 1);
    EXPECT_FALSE(capacities.useLayered());
}

TEST(MlaHostCacheBudgetTest, DecodeScratchCoversLargerDraftIndexerAndRejectsOverflow) {
    EXPECT_EQ(pinnedMlaRequiredTokenSlots(2048, 4096, 6, 4), 98'304);
    EXPECT_EQ(pinnedMlaRequiredTokenSlots(4096, 2048, 6, 4), 98'304);
    EXPECT_EQ(pinnedMlaRequiredTokenSlots(2048, 0, 1, 4), 8192);
    EXPECT_EQ(pinnedMlaRequiredTokenSlots(std::numeric_limits<size_t>::max(), 0, 2, 1), 0);
    EXPECT_EQ(pinnedMlaRequiredTokenSlots(2, 0, 1, std::numeric_limits<size_t>::max()), 0);
}

TEST(MlaHostCacheBudgetTest, DecodeReservesScratchAndChargesLogicalIndexerCapacity) {
    PinnedMlaDecodeBudget input{1000, 300, 100, 10, 300, 3, 100, 100, 4};
    const auto            plan = planPinnedMlaDecode(input);
    EXPECT_EQ(plan.logical_blocks, 9);
    EXPECT_EQ(plan.decode_host_blocks, 3);
    EXPECT_EQ(plan.decode_resident_blocks, 3);
    EXPECT_EQ(plan.decodeDeviceBlocks(plan.logical_blocks), 6);
    EXPECT_EQ(plan.decodeDeviceBlocks(4), 2);
    EXPECT_EQ(plan.decode_hbm_share_denominator, 4);
}

TEST(MlaHostCacheBudgetTest, DecodeChargesSystemPromptStagingBeforeResidentCache) {
    PinnedMlaDecodeBudget input{1000, 300, 100, 10, 300, 3, 100, 100, 4};
    input.context_workspace_bytes = 110;
    const auto plan               = planPinnedMlaDecode(input);
    ASSERT_TRUE(plan.pinnedDecode());
    EXPECT_EQ(plan.logical_blocks, 8);
    EXPECT_EQ(plan.decodeDeviceBlocks(plan.logical_blocks), 5);
    input.context_workspace_bytes = 481;  // Leaves only 219 bytes for backing.
    EXPECT_FALSE(planPinnedMlaDecode(input).pinnedDecode());
}

TEST(MlaHostCacheBudgetTest, DecodeHonorsBothLogicalAndPhysicalLimits) {
    PinnedMlaDecodeBudget input{10'000, 10'000, 100, 10, 300, 3, 10, 4, 0};
    const auto            plan = planPinnedMlaDecode(input);
    EXPECT_EQ(plan.decodeDeviceBlocks(plan.logical_blocks), 2);
    EXPECT_EQ(plan.logical_blocks, 10);
    input.host_bytes = 200;
    EXPECT_EQ(planPinnedMlaDecode(input).decodeDeviceBlocks(6), 4);

    input.device_bytes = 519;  // Just short of the scratch and two HBM blocks.
    EXPECT_FALSE(planPinnedMlaDecode(input).pinnedDecode());
}

}  // namespace
}  // namespace rtp_llm
