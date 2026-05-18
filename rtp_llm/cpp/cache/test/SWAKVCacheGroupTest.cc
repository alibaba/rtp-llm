#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class SWAKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {
        block_pool_ = createBlockPool();
        block_pool_->init();
        total_blocks_ = block_pool_->freeBlocksNum();
    }

    SWAKVCacheGroup makeGroup(int seq_size_per_block) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->seq_size_per_block = seq_size_per_block;
        return SWAKVCacheGroup({}, spec, block_pool_, 0);
    }

    BlockPoolPtr block_pool_;
    size_t       total_blocks_ = 0;
};

// ==================== needBlocksNum ====================

TEST_F(SWAKVCacheGroupTest, NeedBlocksNum_Basic) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.needBlocksNum(1, 0), 1);
    EXPECT_EQ(group.needBlocksNum(4, 0), 1);
    EXPECT_EQ(group.needBlocksNum(5, 0), 2);
    EXPECT_EQ(group.needBlocksNum(8, 0), 2);
    EXPECT_EQ(group.needBlocksNum(9, 0), 3);
}

TEST_F(SWAKVCacheGroupTest, NeedBlocksNum_WithCurrentBlocks) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.needBlocksNum(10, 1), 2);
    EXPECT_EQ(group.needBlocksNum(10, 3), 0);
    EXPECT_EQ(group.needBlocksNum(10, 5), 0);
}

TEST_F(SWAKVCacheGroupTest, NeedBlocksNum_WithReserveStep) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.needBlocksNum(8, 0, 0), 2);
    EXPECT_EQ(group.needBlocksNum(8, 0, 1), 3);
    EXPECT_EQ(group.needBlocksNum(8, 0, 4), 3);
    EXPECT_EQ(group.needBlocksNum(8, 0, 5), 4);
}

// ==================== countTailAllocations ====================

TEST_F(SWAKVCacheGroupTest, CountTailAllocations_EmptyRange) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.countTailAllocations(3, 3, 5), 0);
    EXPECT_EQ(group.countTailAllocations(5, 3, 5), 0);
    EXPECT_EQ(group.countTailAllocations(0, 0, 0), 0);
    EXPECT_EQ(group.countTailAllocations(0, 5, 0), 0);
}

TEST_F(SWAKVCacheGroupTest, CountTailAllocations_SmallTotalSlots) {
    auto group = makeGroup(4);
    // total_slots=1: tail_begin=0, all in tail
    EXPECT_EQ(group.countTailAllocations(0, 1, 1), 1);
    // total_slots=2: tail_begin=0, all in tail
    EXPECT_EQ(group.countTailAllocations(0, 2, 2), 2);
    EXPECT_EQ(group.countTailAllocations(1, 2, 2), 1);
}

TEST_F(SWAKVCacheGroupTest, CountTailAllocations_LargeTotalSlots) {
    auto group = makeGroup(4);
    // total_slots=5: tail_begin=3
    EXPECT_EQ(group.countTailAllocations(0, 5, 5), 2);  // [3,4] in tail
    EXPECT_EQ(group.countTailAllocations(2, 5, 5), 2);  // [3,4]
    EXPECT_EQ(group.countTailAllocations(3, 5, 5), 2);  // [3,4]
    EXPECT_EQ(group.countTailAllocations(4, 5, 5), 1);  // [4]
    EXPECT_EQ(group.countTailAllocations(5, 5, 5), 0);  // empty
}

// ==================== getNeedBlocks ====================

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_SeqLenZero) {
    auto group = makeGroup(4);
    auto need  = group.getNeedBlocks(0, 0, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_SingleBlock) {
    auto group = makeGroup(4);
    // seq_len=3, total_slots=1, tail_blocks=min(1,2)=1
    auto need = group.getNeedBlocks(3, 3, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ExactlyOneBlock) {
    auto group = makeGroup(4);
    // seq_len=4 = seq_size_per_block, total_slots=1, tail_blocks=min(1,2)=1
    auto need = group.getNeedBlocks(4, 4, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_TwoBlocks) {
    auto group = makeGroup(4);
    // seq_len=5, total_slots=2, tail_blocks=min(2,2)=2
    auto need = group.getNeedBlocks(5, 5, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ManyBlocks) {
    auto group = makeGroup(4);
    // seq_len=20, total_slots=5, tail_blocks=min(5,2)=2
    auto need = group.getNeedBlocks(20, 20, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_WithReserveStep) {
    auto group = makeGroup(4);
    // seq_len=8 → S=2, reserve_step=1 → T=ceil(9/4)=3, reserve_extra=1
    auto need = group.getNeedBlocks(8, 8, 1, 0, false);
    EXPECT_EQ(need.extra_blocks, 2 + 1);

    // seq_len=8 → S=2, reserve_step=4 → T=ceil(12/4)=3, reserve_extra=1
    auto need2 = group.getNeedBlocks(8, 8, 4, 0, false);
    EXPECT_EQ(need2.extra_blocks, 2 + 1);

    // seq_len=8 → S=2, reserve_step=5 → T=ceil(13/4)=4, reserve_extra=2
    auto need3 = group.getNeedBlocks(8, 8, 5, 0, false);
    EXPECT_EQ(need3.extra_blocks, 2 + 2);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReuseDisabled) {
    auto group = makeGroup(4);
    // reuse_enabled=false: reuse_blocks_len ignored
    auto need = group.getNeedBlocks(20, 20, 0, 10, false);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReuseNoOverlap) {
    auto group = makeGroup(4);
    // seq_len=20 → total_slots=5, alloc_begin=3
    // reuse_blocks_len=2 <= 3, no overlap
    auto need = group.getNeedBlocks(20, 20, 0, 2, true);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReusePartialOverlap) {
    auto group = makeGroup(4);
    // seq_len=20 → total_slots=5, alloc_begin=3
    // reuse_blocks_len=4: overlap = 4-3 = 1
    auto need = group.getNeedBlocks(20, 20, 0, 4, true);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReuseFullOverlap) {
    auto group = makeGroup(4);
    // seq_len=20 → total_slots=5, alloc_begin=3
    // reuse_blocks_len=5: overlap = 5-3 = 2
    auto need = group.getNeedBlocks(20, 20, 0, 5, true);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReuseExceedsTotal) {
    auto group = makeGroup(4);
    // seq_len=20 → total_slots=5, alloc_begin=3
    // reuse_blocks_len=10: overlap = 10-3 = 7, need = 2-7 = -5 → clamped to 0
    auto need = group.getNeedBlocks(20, 20, 0, 10, true);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_ReuseSmallSeq) {
    auto group = makeGroup(4);
    // seq_len=3 → total_slots=1, alloc_begin=max(0,-1)=0
    // reuse_blocks_len=1: overlap = 1-0 = 1, need = 1-1 = 0
    auto need = group.getNeedBlocks(3, 3, 0, 1, true);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocks_CommonSeqLenIgnored) {
    auto group = makeGroup(4);
    // common_seq_len should not affect result
    auto need1 = group.getNeedBlocks(0, 20, 0, 0, false);
    auto need2 = group.getNeedBlocks(20, 20, 0, 0, false);
    auto need3 = group.getNeedBlocks(100, 20, 0, 0, false);
    EXPECT_EQ(need1.extra_blocks, need2.extra_blocks);
    EXPECT_EQ(need2.extra_blocks, need3.extra_blocks);
    EXPECT_EQ(need1.common_blocks, 0);
}

// ==================== match ====================

TEST_F(SWAKVCacheGroupTest, MatchAlwaysThrows) {
    auto group = makeGroup(4);
    EXPECT_THROW(group.match({101, 102, 103}), std::exception);
}

TEST_F(SWAKVCacheGroupTest, MatchSingleKey_NotFound) {
    auto group  = makeGroup(4);
    auto result = group.matchSingleKey(999);
    EXPECT_TRUE(result.block_indices.empty());
}

TEST_F(SWAKVCacheGroupTest, MatchSingleKey_Found) {
    auto                  group       = makeGroup(4);
    auto                  block_cache = block_pool_->blockCache();
    BlockCache::CacheItem item        = {101, 0, 1, false};
    ASSERT_TRUE(block_cache->put(item));

    auto result = group.matchSingleKey(101);
    ASSERT_EQ(result.block_indices.size(), 1u);
    EXPECT_EQ(result.block_indices[0], 1);
}

// ==================== malloc ====================

TEST_F(SWAKVCacheGroupTest, Malloc_ShortSeq_OnlyOneBlock) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 3));
    // total_slots=1, all in tail, 1 real block
    EXPECT_EQ(block_ids.blocksNum(), 1u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 1);
}

TEST_F(SWAKVCacheGroupTest, Malloc_TwoBlocks_BothReal) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5));
    // total_slots=2, tail_begin=0, both real
    EXPECT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(SWAKVCacheGroupTest, Malloc_ManyBlocks_OnlyTailReal) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    // total_slots=5, tail_begin=3 → [NULL, NULL, NULL, REAL, REAL]
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[i])) << "position " << i << " should be NULL";
    }
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(SWAKVCacheGroupTest, Malloc_Incremental) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // First: seq_len=8 → 2 blocks, both real
    ASSERT_TRUE(group.malloc(block_ids, 8));
    EXPECT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);

    // Extend: seq_len=20 → need 5 total, 3 new, tail [3,4]
    ASSERT_TRUE(group.malloc(block_ids, 20));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    // New positions [2]: NULL, [3,4]: REAL
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    // 2 old + 2 new = 4 real from pool
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 4);
}

TEST_F(SWAKVCacheGroupTest, Malloc_NoOpWhenEnoughBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 8));
    size_t free_after_first = block_pool_->freeBlocksNum();

    // Same seq_len: no new blocks
    ASSERT_TRUE(group.malloc(block_ids, 8));
    EXPECT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_after_first);
}

TEST_F(SWAKVCacheGroupTest, Malloc_WithReserveStep) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // seq_len=4, reserve=4 → T=ceil(8/4)=2, total=2
    // tail_begin=0, both REAL
    ASSERT_TRUE(group.malloc(block_ids, 4, false, 4));
    ASSERT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(SWAKVCacheGroupTest, Malloc_FailsWhenPoolExhausted) {
    auto group = makeGroup(4);
    // Exhaust the pool
    std::vector<BlockIds> holders;
    for (size_t i = 0; i < total_blocks_ / 2; ++i) {
        holders.emplace_back(1);
        ASSERT_TRUE(group.malloc(holders.back(), 5));  // 2 real blocks each
    }
    EXPECT_LT(block_pool_->freeBlocksNum(), 2u);

    BlockIds block_ids(1);
    EXPECT_FALSE(group.malloc(block_ids, 5));
}

// ==================== removeSkippedBlocks ====================

TEST_F(SWAKVCacheGroupTest, RemoveSkippedBlocks_TwoOrFewer_NoOp) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5));
    ASSERT_EQ(block_ids.blocksNum(), 2u);

    group.removeSkippedBlocks(block_ids);
    EXPECT_EQ(block_ids.blocksNum(), 2u);
}

TEST_F(SWAKVCacheGroupTest, RemoveSkippedBlocks_FreesNonTailReal) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // First alloc: 2 blocks [REAL, REAL]
    ASSERT_TRUE(group.malloc(block_ids, 5));
    // Extend: 5 blocks [REAL_old, REAL_old, NULL, REAL_new, REAL_new]
    ASSERT_TRUE(group.malloc(block_ids, 20));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    size_t free_before = block_pool_->freeBlocksNum();

    group.removeSkippedBlocks(block_ids);

    // The 2 old REAL blocks at positions [0,1] should be freed
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before + 2);
    // BlockIds::remove sets freed positions to NULL_BLOCK_IDX (no shrink)
    EXPECT_EQ(block_ids.blocksNum(), 5u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
}

TEST_F(SWAKVCacheGroupTest, RemoveSkippedBlocks_NullsAreKept) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // Alloc a long sequence: [NULL, NULL, NULL, REAL, REAL]
    ASSERT_TRUE(group.malloc(block_ids, 20));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    size_t free_before = block_pool_->freeBlocksNum();

    group.removeSkippedBlocks(block_ids);
    // NULLs at [0,1,2] are not freed (already NULL), keep_begin=3
    // No real blocks before keep_begin
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before);
    EXPECT_EQ(block_ids.blocksNum(), 5u);
}

// ==================== free ====================

TEST_F(SWAKVCacheGroupTest, Free_ReleasesRealBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);

    group.free(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(SWAKVCacheGroupTest, Free_Empty) {
    auto group = makeGroup(4);
    group.free({});
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(SWAKVCacheGroupTest, Free_SkipsNullBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    // blocks = [NULL, NULL, NULL, REAL, REAL], 2 real
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);

    group.free(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// ==================== reference ====================

TEST_F(SWAKVCacheGroupTest, Reference_AddsAndRefsBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5));
    auto original = block_ids.blocks();  // [REAL_A, REAL_B]

    BlockIds block_ids2(1);
    group.reference(block_ids2, original);
    EXPECT_EQ(block_ids2.blocksNum(), 2u);
    EXPECT_EQ(block_ids2.blocks(), original);
}

TEST_F(SWAKVCacheGroupTest, Reference_NullBlocksNotReffed) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    auto original = block_ids.blocks();  // [NULL, NULL, NULL, REAL, REAL]

    BlockIds block_ids2(1);
    group.reference(block_ids2, original);
    EXPECT_EQ(block_ids2.blocksNum(), original.size());
}

// ==================== insertIntoCache ====================

TEST_F(SWAKVCacheGroupTest, InsertIntoCache_SkipsNullBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    // blocks = [NULL, NULL, NULL, REAL, REAL]
    CacheKeysType keys = {101, 102, 103, 104, 105};
    group.insertIntoCache(keys, block_ids.blocks(), false);

    // Only the 2 real blocks should be cached (keys 104, 105)
    auto result1 = group.matchSingleKey(101);
    EXPECT_TRUE(result1.block_indices.empty());

    auto result4 = group.matchSingleKey(104);
    ASSERT_EQ(result4.block_indices.size(), 1u);
    EXPECT_EQ(result4.block_indices[0], block_ids.blocks()[3]);

    auto result5 = group.matchSingleKey(105);
    ASSERT_EQ(result5.block_indices.size(), 1u);
    EXPECT_EQ(result5.block_indices[0], block_ids.blocks()[4]);
}

TEST_F(SWAKVCacheGroupTest, InsertIntoCache_EmptyInput) {
    auto group = makeGroup(4);
    group.insertIntoCache({}, {}, false);
    // No crash, no-op
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
