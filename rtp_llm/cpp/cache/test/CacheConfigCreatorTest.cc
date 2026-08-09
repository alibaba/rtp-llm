#include "gtest/gtest.h"

#include <limits>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

// Built with -fno-access-control (see cache/test/BUILD test_copts), so these private static
// helpers are reachable from the test translation unit.

namespace rtp_llm {
namespace {

constexpr int kIntMax = std::numeric_limits<int>::max();

TEST(CacheConfigCreatorLocalBlockNumTest, SentinelOnlyReturnsOne) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum({}, 0, 0, 2, /*sentinel_only=*/true), 1u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, TestBlockNumOverridesBudget) {
    EXPECT_EQ(CacheConfigCreator::localBlockNum({}, 0, /*test_block_num=*/8, 2, false), 8u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, TestBlockNumBelowTwoThrows) {
    EXPECT_THROW(CacheConfigCreator::localBlockNum({}, 0, /*test_block_num=*/-1, 2, false), std::exception);
    EXPECT_THROW(CacheConfigCreator::localBlockNum({}, 0, /*test_block_num=*/1, 2, false), std::exception);
}

TEST(CacheConfigCreatorBlockBudgetTest, ZeroMarginalBytesThrows) {
    EXPECT_THROW(maxKVCacheBlockNumForBudget(1000, {}, 2), std::exception);
}

TEST(CacheConfigCreatorBlockBudgetTest, PagedPoolsConsumeEveryLogicalBlock) {
    EXPECT_EQ(maxKVCacheBlockNumForBudget(1000, {/*explicit=*/0, /*paged=*/100, /*swa=*/0}, 2), 10u);
}

TEST(CacheConfigCreatorBlockBudgetTest, ExplicitReserveIsSubtractedBeforeDynamicPools) {
    EXPECT_EQ(maxKVCacheBlockNumForBudget(1000, {/*explicit=*/300, /*paged=*/100, /*swa=*/0}, 2), 7u);
}

TEST(CacheConfigCreatorBlockBudgetTest, SwaUsesCeilOfLogicalBlocksOverStep) {
    const KVCacheBlockBudget budget{/*explicit=*/0, /*paged=*/100, /*swa=*/60};
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total=*/420, budget, /*linear_step=*/2), 3u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total=*/520, budget, /*linear_step=*/2), 4u);
}

TEST(CacheConfigCreatorBlockBudgetTest, PureSwaBudgetHonorsStep) {
    EXPECT_EQ(maxKVCacheBlockNumForBudget(300, {/*explicit=*/0, /*paged=*/0, /*swa=*/100}, 2), 6u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, ExactlyTwoSlotsIsAccepted) {
    const KVCacheBlockBudget budget{/*explicit=*/300, /*paged=*/100, /*swa=*/0};
    EXPECT_EQ(CacheConfigCreator::localBlockNum(budget, /*total=*/500, 0, 2, false), 2u);
}

TEST(CacheConfigCreatorLocalBlockNumTest, InsufficientBudgetThrows) {
    const KVCacheBlockBudget budget{/*explicit=*/0, /*paged=*/100, /*swa=*/0};
    EXPECT_THROW(CacheConfigCreator::localBlockNum(budget, /*total=*/199, 0, 2, false), std::exception);
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
