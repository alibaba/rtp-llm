#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(BlockReleaseBatchTest, DeduplicatesByGroupAndBlockAndKeepsFinalState) {
    BlockReleaseBatch batch;
    constexpr BlockIdxType block_id = 3;

    batch.append(
        /*group_id=*/7,
        std::vector<BlockRefTransition>{
            BlockRefTransition{block_id, BlockRefType::REQUEST, 3, 2, false}});
    batch.append(
        /*group_id=*/7,
        std::vector<BlockRefTransition>{BlockRefTransition{block_id, BlockRefType::STORAGE_BACKEND, 2, 1, false}});

    const auto receipts = batch.finish();

    ASSERT_EQ(receipts.size(), 1u);
    EXPECT_EQ(receipts[0].group_id, 7u);
    EXPECT_EQ(receipts[0].block_id, block_id);
    EXPECT_EQ(receipts[0].released_ref_type, BlockRefType::STORAGE_BACKEND);
    EXPECT_EQ(receipts[0].old_total_ref_count, 3u);
    EXPECT_EQ(receipts[0].new_total_ref_count, 1u);
    EXPECT_FALSE(receipts[0].block_released);
}

TEST(BlockReleaseBatchTest, FiltersUnsupportedAndInvalidTransitions) {
    BlockReleaseBatch batch;
    batch.append(
        /*group_id=*/2,
        std::vector<BlockRefTransition>{
            BlockRefTransition{1, BlockRefType::BLOCK_CACHE, 2, 1, false},
            BlockRefTransition{NULL_BLOCK_IDX, BlockRefType::REQUEST, 1, 0, true},
            BlockRefTransition{2, BlockRefType::REQUEST, 1, 0, true}});

    const auto receipts = batch.finish();

    ASSERT_EQ(receipts.size(), 1u);
    EXPECT_EQ(receipts[0].block_id, 2);
    EXPECT_TRUE(receipts[0].block_released);
}

TEST(BlockReleaseBatchTest, FinishResetsTheCollector) {
    BlockReleaseBatch batch;
    batch.append(
        /*group_id=*/1,
        std::vector<BlockRefTransition>{
            BlockRefTransition{1, BlockRefType::REQUEST, 1, 0, true}});
    ASSERT_EQ(batch.finish().size(), 1u);
    EXPECT_TRUE(batch.finish().empty());
}

}  // namespace
}  // namespace rtp_llm
