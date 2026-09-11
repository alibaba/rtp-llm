#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace rtp_llm {
namespace {

TEST(KVCacheTransferPlannerTest, FullGroupPublishesOnlyRequestedRange) {
    const auto plan = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::FULL, 0, 1, CacheStorePublishRange{2, 5, false});
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].key_index, 2);
    EXPECT_EQ(plan[1].key_index, 3);
    EXPECT_EQ(plan[2].key_index, 4);
}

TEST(KVCacheTransferPlannerTest, LinearGroupWaitsForTerminalPublication) {
    const auto intermediate = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{0, 4, false});
    EXPECT_TRUE(intermediate.empty());

    const auto terminal = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 8, true});
    ASSERT_EQ(terminal.size(), 1u);
    EXPECT_EQ(terminal[0].key_index, 7);
    EXPECT_EQ(terminal[0].offset_index, 7);

    const auto terminal_without_hybrid_flag = buildIncrementalCacheStoreBlockPlan(
        8, 0, false, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 8, true});
    ASSERT_EQ(terminal_without_hybrid_flag.size(), 1u);
    EXPECT_EQ(terminal_without_hybrid_flag[0].key_index, 7);

    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 7, true}),
                 std::invalid_argument);
}

TEST(KVCacheTransferPlannerTest, FullGroupPreservesCpKeyOffsetMapping) {
    const auto plan = buildIncrementalCacheStoreBlockPlan(
        9, 0, true, CacheGroupType::FULL, 1, 2, CacheStorePublishRange{3, 8, false});
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 1);
    EXPECT_EQ(plan[1].key_index, 5);
    EXPECT_EQ(plan[1].offset_index, 2);
    EXPECT_EQ(plan[2].key_index, 7);
    EXPECT_EQ(plan[2].offset_index, 3);
}

TEST(KVCacheTransferPlannerTest, RejectsInvalidRangeAndUnsupportedGroup) {
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     4, 0, true, CacheGroupType::FULL, 0, 1, CacheStorePublishRange{3, 2, false}),
                 std::invalid_argument);
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     4, 0, true, CacheGroupType::SWA, 0, 1, CacheStorePublishRange{0, 2, false}),
                 std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
