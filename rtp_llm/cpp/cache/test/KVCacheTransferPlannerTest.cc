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

    // Keys stay in physical-page units; the final checkpoint row can be partial.
    struct Case { size_t key_count; int row; };
    for (const auto& c : {Case{16, 1}, Case{17, 2}, Case{1, 0}}) {
        SCOPED_TRACE(c.key_count);
        const auto partial = buildIncrementalCacheStoreBlockPlan(
            c.key_count, 0, true, CacheGroupType::LINEAR, 3, 8, CacheStorePublishRange{0, c.key_count, false});
        EXPECT_TRUE(partial.empty());
        const auto final = buildIncrementalCacheStoreBlockPlan(
            c.key_count, 0, true, CacheGroupType::LINEAR, 3, 8, CacheStorePublishRange{0, c.key_count, true});
        ASSERT_EQ(final.size(), 1u);
        EXPECT_EQ(final[0].key_index, c.key_count - 1);
        EXPECT_EQ(final[0].offset_index, c.row);
    }
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

TEST(KVCacheTransferPlannerTest, RejectsInvalidRange) {
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     4, 0, true, CacheGroupType::FULL, 0, 1, CacheStorePublishRange{3, 2, false}),
                 std::invalid_argument);
}

TEST(KVCacheTransferPlannerTest, SwaPublishesPhysicalWindowOnlyAfterTerminalWrite) {
    for (bool hybrid : {false, true}) {
        const auto intermediate = buildIncrementalCacheStoreBlockPlan(
            8, 0, hybrid, CacheGroupType::SWA, 0, 1, CacheStorePublishRange{0, 6, false});
        EXPECT_TRUE(intermediate.empty());
        const auto terminal = buildIncrementalCacheStoreBlockPlan(
            8, 0, hybrid, CacheGroupType::SWA, 0, 1, CacheStorePublishRange{6, 8, true});
        ASSERT_EQ(terminal.size(), 2u);
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(terminal[i].key_index, 6 + i);
            EXPECT_EQ(terminal[i].offset_index, 6 + i);
        }
    }
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     8, 0, true, CacheGroupType::SWA, 0, 1, CacheStorePublishRange{0, 7, true}),
                 std::invalid_argument);
    EXPECT_TRUE(blockPositionsForCacheTransfer(0, 0, true, CacheGroupType::SWA).empty());
    EXPECT_EQ(blockPositionsForCacheTransfer(1, 0, true, CacheGroupType::SWA),
              (std::vector<size_t>{0}));
}

}  // namespace
}  // namespace rtp_llm
