#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(KVCacheTransferPlannerTest, FullCpPageRrUsesRankOwnedOffsets) {
    const auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/8,
                                               /*reuse_block_size=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::FULL,
                                               KVCacheRegionName::DEFAULT,
                                               /*cp_rank=*/3,
                                               /*cp_size=*/8);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 0);
}

TEST(KVCacheTransferPlannerTest, Dsv4FixedCpPageRrUsesVirtualOffsetsAndLastRankKeys) {
    const auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/8,
                                               /*reuse_block_size=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::SWA,
                                               KVCacheRegionName::SWA_KV,
                                               /*cp_rank=*/3,
                                               /*cp_size=*/8);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].key_index, 7);
    EXPECT_EQ(plan[0].offset_index, 0);
}

TEST(KVCacheTransferPlannerTest, Dsv4FixedCpPageRrKeepsLastTwoVirtualSwaBlocks) {
    const auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/16,
                                               /*reuse_block_size=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::SWA,
                                               KVCacheRegionName::INDEXER_STATE,
                                               /*cp_rank=*/0,
                                               /*cp_size=*/8);

    ASSERT_EQ(plan.size(), 2);
    EXPECT_EQ(plan[0].key_index, 7);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].key_index, 15);
    EXPECT_EQ(plan[1].offset_index, 1);
}

TEST(KVCacheTransferPlannerTest, RawSwaKeepsRawLastTwoBlocks) {
    const auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/8,
                                               /*reuse_block_size=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::SWA,
                                               KVCacheRegionName::DEFAULT,
                                               /*cp_rank=*/0,
                                               /*cp_size=*/8);

    ASSERT_EQ(plan.size(), 2);
    EXPECT_EQ(plan[0].key_index, 6);
    EXPECT_EQ(plan[0].offset_index, 6);
    EXPECT_EQ(plan[1].key_index, 7);
    EXPECT_EQ(plan[1].offset_index, 7);
}

}  // namespace
}  // namespace rtp_llm
