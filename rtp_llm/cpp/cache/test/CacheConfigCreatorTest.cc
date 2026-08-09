#include "gtest/gtest.h"

#include <limits>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

// Built with -fno-access-control (see cache/test/BUILD test_copts), so these private static
// helpers are reachable from the test translation unit.

namespace rtp_llm {
namespace {

constexpr int kIntMax = std::numeric_limits<int>::max();

TEST(CacheConfigCreatorLocalBlockNumTest, SentinelOnlyReturnsOne) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum(0, 0, 0, 0, /*sentinel_only=*/true), 1u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, TestBlockNumOverridesBudget) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum(0, 100, 0, /*test_block_num=*/8, false), 8u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, TestBlockNumBelowTwoThrows) {
    EXPECT_THROW(CacheConfigCreator::localBlockNum(0, 100, 0, /*test_block_num=*/-1, false), std::exception);
    EXPECT_THROW(CacheConfigCreator::localBlockNum(0, 100, 0, /*test_block_num=*/1, false), std::exception);
}

TEST(CacheConfigCreatorLocalBlockNumTest, ZeroDynamicSlotThrows) {
    EXPECT_THROW(CacheConfigCreator::localBlockNum(0, /*dynamic_slot_bytes=*/0, 1000, 0, false), std::exception);
}

TEST(CacheConfigCreatorLocalBlockNumTest, BudgetDividedByDynamicSlot) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum(/*explicit=*/0, /*dynamic_slot=*/100, /*budget=*/1000, 0, false), 10u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, ExplicitReserveIsSubtractedBeforeDivision) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum(/*explicit=*/300, /*dynamic_slot=*/100, /*budget=*/1000, 0, false), 7u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, ExactlyTwoSlotsIsAccepted) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum(/*explicit=*/300, /*dynamic_slot=*/100, /*budget=*/500, 0, false), 2u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, InsufficientBudgetThrows) {
    EXPECT_THROW(CacheConfigCreator::localBlockNum(0, /*dynamic_slot=*/100, /*budget=*/50, 0, false), std::exception);
}

TEST(CacheConfigCreatorLocalBlockNumTest, MinimumCapacityArithmeticOverflowThrows) {
    EXPECT_THROW(CacheConfigCreator::localBlockNum(
                     0, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), 0, false),
                 std::exception);
}

TEST(CacheConfigCreatorLocalBlockNumTest, BlockIdxTypeUpperBoundIsEnforced) {
    const size_t too_many_blocks = static_cast<size_t>(std::numeric_limits<BlockIdxType>::max()) + 1;
    EXPECT_THROW(CacheConfigCreator::localBlockNum(0, 1, too_many_blocks, 0, false), std::exception);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, PublishesOnlyRankSlot) {
    int block_nums[] = {11, 22, 33, 44};
    CacheConfigCreator::publishLocalBlockNum(block_nums, 4, /*world_rank=*/2, 7, /*sentinel_only=*/false);
    EXPECT_EQ(block_nums[0], 11);
    EXPECT_EQ(block_nums[1], 22);
    EXPECT_EQ(block_nums[2], 7);
    EXPECT_EQ(block_nums[3], 44);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, SentinelRankPublishesSentinel) {
    int block_nums[] = {11, 22};
    CacheConfigCreator::publishLocalBlockNum(block_nums, 2, /*world_rank=*/1, 1, /*sentinel_only=*/true);
    EXPECT_EQ(block_nums[0], 11);
    EXPECT_EQ(block_nums[1], kIntMax);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, PublishRejectsBlockNumAboveBlockIdxType) {
    int block_nums[] = {kIntMax};
    EXPECT_THROW(CacheConfigCreator::publishLocalBlockNum(
                     block_nums, 1, 0, std::numeric_limits<uint32_t>::max(), /*sentinel_only=*/false),
                 std::exception);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, SelectsMinWhileIgnoringMixedSentinels) {
    const int block_nums[] = {10, kIntMax, 30, kIntMax};
    EXPECT_EQ(CacheConfigCreator::selectConvergedBlockNum(block_nums, 4, 10, /*sentinel_only=*/false), 10u);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, AllSentinelsAreInvalidForAttentionRank) {
    const int block_nums[] = {kIntMax, kIntMax};
    EXPECT_THROW(CacheConfigCreator::selectConvergedBlockNum(block_nums, 2, 10, /*sentinel_only=*/false),
                 std::exception);
}

TEST(CacheConfigCreatorConvergeBlockNumTest, SentinelOnlyRankKeepsLocalReservation) {
    const int block_nums[] = {2, 3};
    EXPECT_EQ(CacheConfigCreator::selectConvergedBlockNum(block_nums, 2, 1, /*sentinel_only=*/true), 1u);
}

}  // namespace
}  // namespace rtp_llm
