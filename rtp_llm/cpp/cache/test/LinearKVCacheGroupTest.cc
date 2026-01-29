#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

static std::shared_ptr<LinearKVCacheSpec> makeLinearSpec(uint32_t seq_size_per_block) {
    auto spec                = std::make_shared<LinearKVCacheSpec>();
    spec->type               = KVCacheSpecType::LinearAttention;
    spec->dtype              = rtp_llm::DataType::TYPE_FP16;
    spec->layer_num          = 2;
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

TEST_F(LinearKVCacheGroupTest, MallocAllocatesStepHitsAndTailWhenReuseEnabled) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIndicesType blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));  // 4 slots

    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks[3]));

    // Only 2 real blocks allocated.
    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(LinearKVCacheGroupTest, MallocAllocatesOnlyTailWhenReuseDisabled) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIndicesType blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false));  // 4 slots

    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks[3]));

    // Only 1 real block allocated.
    EXPECT_EQ(block_pool->freeBlocksNum(), 8u);
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
    BlockIndicesType blocks = allocated;

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, true);

    // For step=2 and size=6:
    // keep index 1(step hit), 3(step hit), and last two (4,5). Free index 0 and 2.
    ASSERT_EQ(blocks.size(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 2);
}

TEST_F(LinearKVCacheGroupTest, InsertIntoCacheSkipsNullBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/3, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_cache, nullptr);

    BlockIndicesType blocks;
    blocks.push_back(NULL_BLOCK_IDX);
    blocks.push_back(block_pool->malloc(1)[0]);
    blocks.push_back(NULL_BLOCK_IDX);
    blocks.push_back(block_pool->malloc(1)[0]);

    CacheKeysType keys = {100, 101, 102, 103};
    group.insertIntoCache(keys, blocks, /*is_resident=*/false);

    EXPECT_FALSE(block_cache->contains(100, /*group_id=*/3));
    EXPECT_TRUE(block_cache->contains(101, /*group_id=*/3));
    EXPECT_FALSE(block_cache->contains(102, /*group_id=*/3));
    EXPECT_TRUE(block_cache->contains(103, /*group_id=*/3));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
