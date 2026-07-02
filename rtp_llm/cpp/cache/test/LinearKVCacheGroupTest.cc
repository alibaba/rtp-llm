#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/group/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

static std::shared_ptr<LinearKVCacheSpec> makeLinearSpec(uint32_t seq_size_per_block) {
    auto spec                = std::make_shared<LinearKVCacheSpec>();
    spec->type               = KVCacheSpecType::LinearAttention;
    spec->dtype              = rtp_llm::DataType::TYPE_FP16;
    spec->local_num_k_heads  = 1;
    spec->local_num_v_heads  = 1;
    spec->head_k_dim         = 1;
    spec->head_v_dim         = 1;
    spec->conv_kernel_dim    = 2;
    spec->local_head_num_kv  = 1;
    spec->seq_size_per_block = seq_size_per_block;
    return spec;
}

class LinearKVCacheGroupTest: public ::testing::Test {};

TEST_F(LinearKVCacheGroupTest, GetNeedBlocksReuseDisabledCountsLastTwoTailAndReserveStep) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4 => common phase materializes
    // its last two slots; incremental phase adds final tail and reserve slots.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(LinearKVCacheGroupTest, GetNeedBlocksReuseEnabledUsesSparseCountingAndReserveStep) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4. Reuse enabled keeps step
    // hits plus the last two seq slots, so this matches the disabled case here.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, true);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(LinearKVCacheGroupTest, MallocAllocatesStepHitsAndTailWhenReuseEnabled) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));  // 4 slots

    ASSERT_EQ(blocks.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[2]));  // tail-1 protects causal_conv1d boundary read
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));

    // Step hit + tail-1 + tail.
    EXPECT_EQ(block_pool->freeBlocksNum(), 6u);
}

TEST_F(LinearKVCacheGroupTest, MallocAllocatesLastTwoTailBlocksWhenReuseDisabled) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false));  // 4 slots

    ASSERT_EQ(blocks.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[2]));  // tail-1 protects causal_conv1d boundary read
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));

    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(LinearKVCacheGroupTest, MallocAllocatesReserveTailBlocksWhenReuseDisabled) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // seq_len=16 => seq_slots=4; reserve_step=2 => total_slots=5
    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false, /*reserve_step=*/2));

    ASSERT_EQ(blocks.blocksNum(), 5u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[2]));  // tail-1 protects causal_conv1d boundary read
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));  // seq tail
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));  // reserve tail

    EXPECT_EQ(block_pool->freeBlocksNum(), 6u);
}

TEST_F(LinearKVCacheGroupTest, RemoveSkippedBlocksFreesNonStepBlocksButKeepsLastTwo) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // Start with 6 allocated blocks (no NULLs) to test the pruning logic.
    auto allocated = block_pool->malloc(6);
    ASSERT_EQ(allocated.size(), 6u);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, true);

    // For step=2 and size=6:
    // keep index 1(step hit), 3(step hit), and last two (4,5). Free index 0 and 2.
    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 2);
}

TEST_F(LinearKVCacheGroupTest, MatchSingleKeyReturnsMatchedBlockOrEmpty) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto shared_cache = std::make_shared<SharedBlockCache>();
    shared_cache->init(1, {block_pool});

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2, shared_cache.get());
    ASSERT_TRUE(group.init());

    auto blocks = block_pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);

    shared_cache->put(123, {blocks[0]}, false);

    auto hit = group.matchSingleKey(123);
    ASSERT_EQ(hit.block_indices.size(), 1u);
    EXPECT_EQ(hit.block_indices[0], blocks[0]);

    auto miss = group.matchSingleKey(999);
    EXPECT_TRUE(miss.block_indices.empty());
}

TEST_F(LinearKVCacheGroupTest, MallocNoNewBlocksReturnsTrueAndKeepsState) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));  // 4 slots
    const auto   blocks_before = blocks.blocks();
    const size_t free_before   = block_pool->freeBlocksNum();

    // Same seq_len => new_blocks_len == 0.
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));
    EXPECT_EQ(blocks.blocks(), blocks_before);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
}

TEST_F(LinearKVCacheGroupTest, MallocFailsWhenBlockPoolExhausted) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    // Exhaust all free blocks (block 0 is reserved).
    auto all_blocks = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum()));
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    EXPECT_FALSE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));

    // Cleanup to avoid leaking refs in the test process.
    block_pool->requestFree(all_blocks);
}

TEST_F(LinearKVCacheGroupTest, MallocEnsuresFreeBlocksByEvictingCache) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto shared_cache = std::make_shared<SharedBlockCache>();
    shared_cache->init(1, {block_pool});

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2, shared_cache.get());
    ASSERT_TRUE(group.init());

    auto cached = block_pool->malloc(1);
    ASSERT_EQ(cached.size(), 1u);
    shared_cache->put(123, {cached[0]}, false);
    block_pool->requestFree(cached);

    auto occupied = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum()));
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));
    ASSERT_EQ(blocks.blocksNum(), 1u);
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[0]));

    group.free(blocks.blocks());
    block_pool->requestFree(occupied);
}

TEST_F(LinearKVCacheGroupTest, RemoveSkippedBlocksWithReserveStepKeepsLastTwoAndReserveTail) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(6);
    ASSERT_EQ(allocated.size(), 6u);
    BlockIds blocks;
    blocks.assign(allocated);  // no NULLs

    const size_t free_before = block_pool->freeBlocksNum();
    // reserve_step=1 => keep last 2 plus 1 more block (index 3).
    group.removeSkippedBlocks(blocks, /*enable_reuse_cache=*/false, /*reserve_step=*/1);

    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 3);
}

TEST_F(LinearKVCacheGroupTest, FreeIgnoresEmptyOrAllNullBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    const size_t free_before = block_pool->freeBlocksNum();
    group.free(BlockIndicesType{});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);

    group.free(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
}

TEST_F(LinearKVCacheGroupTest, ReferenceAppendsAndIncrementsRefCountForValidBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto blocks = block_pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    ASSERT_EQ(block_pool->freeBlocksNum(), 8u);

    BlockIds         dst;
    BlockIndicesType new_blocks = {NULL_BLOCK_IDX, blocks[0]};
    group.reference(dst, new_blocks);

    ASSERT_EQ(dst.blocksNum(), 2u);
    EXPECT_TRUE(isNullBlockIdx(dst.blocks()[0]));
    EXPECT_EQ(dst.blocks()[1], blocks[0]);

    // Because reference() adds an extra requestReference, it should take two requestFree calls to become free again.
    const size_t free_before = block_pool->freeBlocksNum();
    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);  // still referenced
    block_pool->requestFree(blocks[0]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 1);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
