#include "rtp_llm/cpp/model_rpc/CacheTransferBlockSelector.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace test {

TEST(CacheTransferBlockSelectorTest, FullGroupLoadsEveryActualBlockWithoutReserve) {
    const BlockIndicesType blocks{10, 11, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/3, /*reuse_block_count=*/0);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, (std::vector<size_t>{0, 1, 2}));
}

TEST(CacheTransferBlockSelectorTest, FullGroupDoesNotLoadReservedTail) {
    const BlockIndicesType blocks{10, 11, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/2, /*reuse_block_count=*/0);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, (std::vector<size_t>{0, 1}));
}

TEST(CacheTransferBlockSelectorTest, FullGroupSkipsReusedPrefixAndReservedTail) {
    const BlockIndicesType blocks{10, 11, 12, 13, 14};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/4, /*reuse_block_count=*/2);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, (std::vector<size_t>{2, 3}));
}

TEST(CacheTransferBlockSelectorTest, LinearGroupLoadsActualTailInsteadOfReservedTail) {
    const BlockIndicesType blocks{NULL_BLOCK_IDX, 11, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::LINEAR, blocks, /*cache_key_count=*/2, /*reuse_block_count=*/0);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, (std::vector<size_t>{1}));
}

TEST(CacheTransferBlockSelectorTest, LinearGroupLoadsUnreusedActualTail) {
    const BlockIndicesType blocks{NULL_BLOCK_IDX, 11, NULL_BLOCK_IDX, 13, 14};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::LINEAR, blocks, /*cache_key_count=*/4, /*reuse_block_count=*/2);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, (std::vector<size_t>{3}));
}

TEST(CacheTransferBlockSelectorTest, AllActualBlocksReusedNeedsNoTransfer) {
    const BlockIndicesType blocks{10, 11, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::LINEAR, blocks, /*cache_key_count=*/2, /*reuse_block_count=*/2);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(result->empty());
}

TEST(CacheTransferBlockSelectorTest, RejectsMoreKeysThanAllocatedSlots) {
    const BlockIndicesType blocks{10, 11};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/3, /*reuse_block_count=*/0);

    EXPECT_FALSE(result.ok());
}

TEST(CacheTransferBlockSelectorTest, RejectsReuseBeyondActualKeys) {
    const BlockIndicesType blocks{10, 11, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/2, /*reuse_block_count=*/3);

    EXPECT_FALSE(result.ok());
}

TEST(CacheTransferBlockSelectorTest, RejectsMissingPhysicalBlockAtSelectedPosition) {
    const BlockIndicesType blocks{10, NULL_BLOCK_IDX, 12};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/2, /*reuse_block_count=*/0);

    EXPECT_FALSE(result.ok());
}

TEST(CacheTransferBlockSelectorTest, EmptyActualSequenceDoesNotLoadReservedBlocks) {
    const BlockIndicesType blocks{10, 11};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/0, /*reuse_block_count=*/0);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(result->empty());
}

TEST(CacheTransferBlockSelectorTest, RejectsNegativeReuseCount) {
    const BlockIndicesType blocks{10};
    auto result = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, blocks, /*cache_key_count=*/1, /*reuse_block_count=*/-1);

    EXPECT_FALSE(result.ok());
}

}  // namespace test
}  // namespace rtp_llm
