#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace test {

static std::shared_ptr<LinearKVCacheSpec> makeLinearSpec(uint32_t seq_size_per_block) {
    return makeResolvedLinearSpec(rtp_llm::DataType::TYPE_FP16,
                                  1,
                                  1,
                                  1,
                                  1,
                                  2,
                                  seq_size_per_block,
                                  rtp_llm::DataType::TYPE_FP16,
                                  rtp_llm::DataType::TYPE_FP16,
                                  "linear");
}

class LinearKVCacheGroupTest: public ::testing::Test {};

TEST_F(LinearKVCacheGroupTest, DefaultPolicyDrivesBehaviorInterfaces) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);

    EXPECT_TRUE(group.prefixReusable());
    EXPECT_TRUE(group.hasSparseSlots());
    EXPECT_FALSE(group.hasKernelBlockSubdiv());
    EXPECT_TRUE(group.transferTailBlocks());
    EXPECT_TRUE(group.isReservable());
    EXPECT_FALSE(group.memoryPlacement() == CacheMemoryPlacement::HOST_PINNED);

    auto disabled_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    disabled_policy.enable_prefix_reuse = false;
    LinearKVCacheGroup disabled_group(/*layer_ids=*/{},
                                      spec,
                                      block_pool,
                                      /*group_id=*/0,
                                      /*linear_step=*/2,
                                      nullptr,
                                      nullptr,
                                      disabled_policy);
    EXPECT_FALSE(disabled_group.prefixReusable());
}

TEST_F(LinearKVCacheGroupTest, GetNeedBlocksReuseDisabledCountsLastTwoTailAndReserveStep) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4 => common phase materializes
    // its last slot; incremental phase adds final tail and reserve slots.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 1);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(LinearKVCacheGroupTest, GetNeedBlocksReuseEnabledUsesSparseCountingAndReserveStep) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4. Reuse enabled keeps step
    // hits plus the last seq slot.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, true);
    EXPECT_EQ(need.common_blocks, 1);
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
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));

    // Step hit + tail.
    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
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
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));

    EXPECT_EQ(block_pool->freeBlocksNum(), 8u);
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
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));  // seq tail
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));  // reserve tail

    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(LinearKVCacheGroupTest, MallocBackfillsExistingNullReadSlot) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(2);
    ASSERT_EQ(allocated.size(), 2u);

    BlockIds blocks;
    blocks.assign(BlockIndicesType{allocated[0], NULL_BLOCK_IDX, allocated[1]});
    const size_t free_before = block_pool->freeBlocksNum();

    // seq_len=12 => seq_slots=3. Only the final tail slot is materialized,
    // so the earlier NULL slot remains sparse when no new slots are appended.
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/12, /*enable_reuse_cache=*/false));

    ASSERT_EQ(blocks.blocksNum(), 3u);
    EXPECT_EQ(blocks.blocks()[0], allocated[0]);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_EQ(blocks.blocks()[2], allocated[1]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
}

TEST_F(LinearKVCacheGroupTest, MallocMaterializesCausalConvReadSlotAtBoundaries) {
    const std::vector<int> seq_lens = {4, 5, 8, 9};

    for (bool enable_reuse_cache : {false, true}) {
        for (int seq_len : seq_lens) {
            auto block_pool = createBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
            LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
            ASSERT_TRUE(group.init());

            BlockIds blocks;
            ASSERT_TRUE(group.malloc(blocks, seq_len, enable_reuse_cache)) << "seq_len=" << seq_len;

            const int tail_pos = (seq_len + 4 - 1) / 4 - 1;
            ASSERT_GE(tail_pos, 0);
            ASSERT_LT(static_cast<size_t>(tail_pos), blocks.blocksNum());
            EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[static_cast<size_t>(tail_pos)]))
                << "seq_len=" << seq_len << " reuse=" << enable_reuse_cache << " tail_pos=" << tail_pos;
        }
    }
}

TEST_F(LinearKVCacheGroupTest, GetNeedBlocksMatchesMallocForReserveSteps) {
    for (bool enable_reuse_cache : {false, true}) {
        for (int reserve_step : {0, 1, 2, 3}) {
            auto block_pool = createBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
            LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
            ASSERT_TRUE(group.init());

            const auto need = group.getNeedBlocks(/*common_seq_len=*/8,
                                                  /*seq_len=*/12,
                                                  reserve_step,
                                                  /*reuse_blocks_len=*/0,
                                                  enable_reuse_cache);

            BlockIds blocks;
            ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/8, enable_reuse_cache));
            ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/12, enable_reuse_cache, reserve_step));

            size_t valid_count = 0;
            for (auto block : blocks.blocks()) {
                if (!isNullBlockIdx(block)) {
                    valid_count++;
                }
            }
            EXPECT_EQ(valid_count, static_cast<size_t>(need.common_blocks + need.extra_blocks))
                << "reserve_step=" << reserve_step << " reuse=" << enable_reuse_cache;
        }
    }
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
    // keep index 1(step hit), 3(step hit), and the last two tails (4, 5). Free index 0 and 2.
    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 2);
}

TEST_F(LinearKVCacheGroupTest, PutIntoCacheSkipsNullBlocks) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                      shared_cache = std::make_shared<SharedBlockCache>();
    std::vector<BlockPoolPtr> group_pools(4, block_pool);
    shared_cache->init(4, group_pools);

    auto block1 = block_pool->malloc(1)[0];
    auto block2 = block_pool->malloc(1)[0];

    // Only put entries with non-NULL blocks (simulating allocator-level filtering)
    std::vector<BlockIdxType> slots1(4, NULL_BLOCK_IDX);
    slots1[3] = block1;
    shared_cache->put(101, slots1, /*is_resident=*/false);

    std::vector<BlockIdxType> slots2(4, NULL_BLOCK_IDX);
    slots2[3] = block2;
    shared_cache->put(103, slots2, /*is_resident=*/false);

    EXPECT_FALSE(shared_cache->contains(100));
    EXPECT_TRUE(shared_cache->contains(101));
    EXPECT_FALSE(shared_cache->contains(102));
    EXPECT_TRUE(shared_cache->contains(103));
}

TEST_F(LinearKVCacheGroupTest, MatchSingleKeyReturnsMatchedBlockOrEmpty) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                      shared_cache = std::make_shared<SharedBlockCache>();
    std::vector<BlockPoolPtr> group_pools(8, block_pool);
    shared_cache->init(8, group_pools);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/7, /*linear_step=*/2, shared_cache.get());
    ASSERT_TRUE(group.init());

    // Allocate a block, then put it into cache for group_id=7.
    auto blocks = block_pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);

    std::vector<BlockIdxType> group_slots(8, NULL_BLOCK_IDX);
    group_slots[7] = blocks[0];
    shared_cache->put(123, group_slots, /*is_resident=*/false);

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

    auto                      shared_cache = std::make_shared<SharedBlockCache>();
    std::vector<BlockPoolPtr> group_pools  = {block_pool};
    shared_cache->init(1, group_pools);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2, shared_cache.get());
    ASSERT_TRUE(group.init());

    // Put one block into cache (non-resident) and release request reference so it becomes evictable.
    auto cached = block_pool->malloc(1);
    ASSERT_EQ(cached.size(), 1u);
    std::vector<BlockIdxType> slots = {cached[0]};
    shared_cache->put(123, slots, /*is_resident=*/false);
    block_pool->requestFree(cached);

    // Exhaust the remaining free blocks so malloc must evict from cache to proceed.
    auto occupied = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum()));
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));
    ASSERT_EQ(blocks.blocksNum(), 1u);
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[0]));

    // Cleanup to avoid leaking refs in the test process.
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
    // reserve_step=1 => cleanup preserves two active tails plus the reserve tail.
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

TEST_F(LinearKVCacheGroupTest, InsertIntoCacheWithEmptyInputsIsNoop) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                      shared_cache = std::make_shared<SharedBlockCache>();
    std::vector<BlockPoolPtr> group_pools(4, block_pool);
    shared_cache->init(4, group_pools);

    auto               spec = makeLinearSpec(/*seq_size_per_block=*/4);
    LinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/3, /*linear_step=*/2, shared_cache.get());
    ASSERT_TRUE(group.init());

    EXPECT_EQ(shared_cache->size(), 0u);
    group.insertIntoCache(CacheKeysType{}, BlockIndicesType{1, 2}, /*is_resident=*/false);
    group.insertIntoCache(CacheKeysType{100, 101}, BlockIndicesType{}, /*is_resident=*/false);
    EXPECT_EQ(shared_cache->size(), 0u);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
