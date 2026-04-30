#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

static std::shared_ptr<MHAKVCacheSpec> makeSWASpec(uint32_t seq_size_per_block) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheSpecType::MultiHeadAttention;
    spec->dtype              = rtp_llm::DataType::TYPE_FP16;
    spec->layer_num          = 2;
    spec->local_head_num_kv  = 1;
    spec->size_per_head      = 1;
    spec->seq_size_per_block = seq_size_per_block;
    return spec;
}

class SWAKVCacheGroupTest: public ::testing::Test {};

TEST_F(SWAKVCacheGroupTest, NeedBlocksNumUsesFullBoundaryWithReserveStep) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);

    EXPECT_EQ(group.needBlocksNum(/*seq_len=*/16, /*current_blocks=*/4, /*reserve_step=*/0), 0);
    EXPECT_EQ(group.needBlocksNum(/*seq_len=*/16, /*current_blocks=*/4, /*reserve_step=*/2), 1);
    EXPECT_EQ(group.needBlocksNum(/*seq_len=*/14, /*current_blocks=*/4, /*reserve_step=*/2), 0);
    EXPECT_EQ(group.needBlocksNum(/*seq_len=*/1, /*current_blocks=*/0, /*reserve_step=*/0), 1);
}

TEST_F(SWAKVCacheGroupTest, GetNeedBlocksCountsOnlyTailTwoAndReserveBoundary) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    // common_slots=2; seq=12 reserve=2 => total_slots=4. SWA allocates common tail two
    // once, then the appended total tail two for each batch.
    auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);

    // reserve_step does not allocate per step: it allocates only if the reserved tokens cross a block boundary.
    need =
        group.getNeedBlocks(/*common_seq_len=*/16, /*seq_len=*/16, /*reserve_step=*/2, /*reuse_blocks_len=*/4, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 1);

    need =
        group.getNeedBlocks(/*common_seq_len=*/16, /*seq_len=*/14, /*reserve_step=*/2, /*reuse_blocks_len=*/4, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(SWAKVCacheGroupTest, MallocAllocatesOnlyTailTwoBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));

    ASSERT_EQ(blocks.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(SWAKVCacheGroupTest, MallocReserveAllocatesOnlyWhenTouchingNewBlockBoundary) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false));
    const size_t free_after_init = block_pool->freeBlocksNum();

    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false, /*reserve_step=*/2));
    ASSERT_EQ(blocks.blocksNum(), 5u);
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_EQ(block_pool->freeBlocksNum(), free_after_init - 1);

    BlockIds no_boundary_blocks;
    ASSERT_TRUE(group.malloc(no_boundary_blocks, /*seq_len=*/14, /*enable_reuse_cache=*/false));
    const size_t free_before_no_boundary = block_pool->freeBlocksNum();
    ASSERT_TRUE(group.malloc(no_boundary_blocks, /*seq_len=*/14, /*enable_reuse_cache=*/false, /*reserve_step=*/2));
    EXPECT_EQ(no_boundary_blocks.blocksNum(), 4u);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before_no_boundary);
}

TEST_F(SWAKVCacheGroupTest, RemoveSkippedBlocksKeepsOnlyLastTwo) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(6);
    ASSERT_EQ(allocated.size(), 6u);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, /*enable_reuse_cache=*/true, /*reserve_step=*/3);

    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 4);
}

TEST_F(SWAKVCacheGroupTest, InsertIntoCacheSkipsNullBlocksAndMatchSingleKeyReturnsHit) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/3);
    ASSERT_TRUE(group.init());

    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_cache, nullptr);

    auto block1 = block_pool->malloc(1);
    auto block2 = block_pool->malloc(1);
    ASSERT_EQ(block1.size(), 1u);
    ASSERT_EQ(block2.size(), 1u);

    BlockIndicesType blocks = {NULL_BLOCK_IDX, block1[0], NULL_BLOCK_IDX, block2[0]};
    CacheKeysType    keys   = {100, 101, 102, 103};
    group.insertIntoCache(keys, blocks, /*is_resident=*/false);

    EXPECT_FALSE(block_cache->contains(100, /*group_id=*/3));
    EXPECT_TRUE(block_cache->contains(101, /*group_id=*/3));
    EXPECT_FALSE(block_cache->contains(102, /*group_id=*/3));
    EXPECT_TRUE(block_cache->contains(103, /*group_id=*/3));

    auto hit = group.matchSingleKey(103);
    ASSERT_EQ(hit.block_indices.size(), 1u);
    EXPECT_EQ(hit.block_indices[0], blocks[3]);

    auto miss = group.matchSingleKey(999);
    EXPECT_TRUE(miss.block_indices.empty());
}

TEST_F(SWAKVCacheGroupTest, FreeAndReferenceIgnoreNullBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    auto blocks = block_pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    ASSERT_EQ(block_pool->freeBlocksNum(), 8u);

    BlockIds         dst;
    BlockIndicesType new_blocks = {NULL_BLOCK_IDX, blocks[0]};
    group.reference(dst, new_blocks);
    EXPECT_EQ(dst.blocks(), new_blocks);

    const size_t free_before = block_pool->freeBlocksNum();
    group.free(BlockIndicesType{NULL_BLOCK_IDX});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);

    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 1);
}

TEST_F(SWAKVCacheGroupTest, MallocFailsWhenBlockPoolExhausted) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto all_blocks = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum()));
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    auto            spec = makeSWASpec(/*seq_size_per_block=*/4);
    SWAKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    EXPECT_FALSE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));
    block_pool->requestFree(all_blocks);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
