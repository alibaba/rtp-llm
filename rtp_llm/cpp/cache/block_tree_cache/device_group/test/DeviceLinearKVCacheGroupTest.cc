#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceLinearKVCacheGroup.h"
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

class DeviceLinearKVCacheGroupTest: public ::testing::Test {};

TEST_F(DeviceLinearKVCacheGroupTest, DefaultPolicyDrivesBehaviorInterfaces) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);

    EXPECT_FALSE(group.prefixReusable());
    EXPECT_FALSE(group.isCpShardable());
    EXPECT_TRUE(group.hasSparseSlots());
    EXPECT_FALSE(group.hasKernelBlockSubdiv());
    EXPECT_TRUE(group.transferTailBlocks());
    EXPECT_FALSE(group.cpCompactTailBlocks());
    EXPECT_TRUE(group.isReservable());
    EXPECT_FALSE(group.usesPinnedCpuBacking());
}

TEST_F(DeviceLinearKVCacheGroupTest, GetNeedBlocksReuseDisabledCountsLastTwoTailAndReserveStep) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4 => common phase materializes
    // its last two slots; incremental phase adds final tail and reserve slots.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceLinearKVCacheGroupTest, GetNeedBlocksReuseEnabledUsesSparseCountingAndReserveStep) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // common_slots=2, seq_slots=3, total_slots=4. Reuse enabled keeps step
    // hits plus the last two seq slots, so this matches the disabled case here.
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, true);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceLinearKVCacheGroupTest, MallocAllocatesStepHitsAndTailWhenReuseEnabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
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

TEST_F(DeviceLinearKVCacheGroupTest, MallocAllocatesLastTwoTailBlocksWhenReuseDisabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
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

TEST_F(DeviceLinearKVCacheGroupTest, MallocAllocatesReserveTailBlocksWhenReuseDisabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
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

TEST_F(DeviceLinearKVCacheGroupTest, MallocBackfillsExistingNullReadSlot) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(2).value();
    ASSERT_EQ(allocated.size(), 2u);
    // New pool malloc reserves capacity at refCount 0; take a request ref so these
    // pre-existing request blocks are held while the group backfills the read slot.
    block_pool->incRef(allocated);

    BlockIds blocks;
    blocks.assign(BlockIndicesType{allocated[0], NULL_BLOCK_IDX, allocated[1]});
    const size_t free_before = block_pool->freeBlocksNum();

    // seq_len=12 => seq_slots=3. Position 1 is tail-1 and is the read slot
    // for sequence_length=13, so it must be materialized even though no new
    // slots are appended.
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/12, /*enable_reuse_cache=*/false));

    ASSERT_EQ(blocks.blocksNum(), 3u);
    EXPECT_EQ(blocks.blocks()[0], allocated[0]);
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_EQ(blocks.blocks()[2], allocated[1]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before - 1);
}

TEST_F(DeviceLinearKVCacheGroupTest, MallocMaterializesCausalConvReadSlotAtBoundaries) {
    const std::vector<int> seq_lens = {4, 5, 8, 9};

    for (bool enable_reuse_cache : {false, true}) {
        for (int seq_len : seq_lens) {
            auto block_pool = createDeviceBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
            DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
            ASSERT_TRUE(group.init());

            BlockIds blocks;
            ASSERT_TRUE(group.malloc(blocks, seq_len, enable_reuse_cache)) << "seq_len=" << seq_len;

            const int read_pos = (seq_len - 2) / 4;
            ASSERT_GE(read_pos, 0);
            ASSERT_LT(static_cast<size_t>(read_pos), blocks.blocksNum());
            EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[static_cast<size_t>(read_pos)]))
                << "seq_len=" << seq_len << " reuse=" << enable_reuse_cache << " read_pos=" << read_pos;
        }
    }
}

TEST_F(DeviceLinearKVCacheGroupTest, GetNeedBlocksMatchesMallocForReserveSteps) {
    for (bool enable_reuse_cache : {false, true}) {
        for (int reserve_step : {0, 1, 2, 3}) {
            auto block_pool = createDeviceBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
            DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
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

TEST_F(DeviceLinearKVCacheGroupTest, RemoveSkippedBlocksFreesNonStepBlocksButKeepsLastTwo) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    // Start with 6 allocated blocks (no NULLs) to test the pruning logic.
    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    // Hold a request ref on each block; removeSkippedBlocks() releases pruned blocks via
    // decRef(), which requires refCount > 0 (new pool malloc leaves them at 0).
    block_pool->incRef(allocated);
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

TEST_F(DeviceLinearKVCacheGroupTest, MallocNoNewBlocksReturnsTrueAndKeepsState) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
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

TEST_F(DeviceLinearKVCacheGroupTest, MallocFailsWhenBlockPoolExhausted) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    // Exhaust all free blocks (block 0 is reserved).
    auto all_blocks = block_pool->malloc(block_pool->freeBlocksNum()).value();
    // Hold a request ref so the cleanup decRef() below has a holder to drop.
    block_pool->incRef(all_blocks);
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    EXPECT_FALSE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));

    // Cleanup to avoid leaking refs in the test process.
    block_pool->decRef(all_blocks);
}

TEST_F(DeviceLinearKVCacheGroupTest, RemoveSkippedBlocksWithReserveStepKeepsLastTwoAndReserveTail) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    // Hold a request ref on each block; removeSkippedBlocks() releases pruned blocks via
    // decRef(), which requires refCount > 0 (new pool malloc leaves them at 0).
    block_pool->incRef(allocated);
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

TEST_F(DeviceLinearKVCacheGroupTest, FreeIgnoresEmptyOrAllNullBlocks) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    const size_t free_before = block_pool->freeBlocksNum();
    group.free(BlockIndicesType{});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);

    group.free(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
}

TEST_F(DeviceLinearKVCacheGroupTest, ReferenceAppendsAndIncrementsRefCountForValidBlocks) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearKVCacheGroup group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());

    auto blocks = block_pool->malloc(1).value();
    ASSERT_EQ(blocks.size(), 1u);
    // New pool malloc reserves capacity at refCount 0; take the request ref (refCount 1).
    block_pool->incRef(blocks);
    ASSERT_EQ(block_pool->freeBlocksNum(), 8u);

    BlockIds         dst;
    BlockIndicesType new_blocks = {NULL_BLOCK_IDX, blocks[0]};
    group.reference(dst, new_blocks);

    ASSERT_EQ(dst.blocksNum(), 2u);
    EXPECT_TRUE(isNullBlockIdx(dst.blocks()[0]));
    EXPECT_EQ(dst.blocks()[1], blocks[0]);

    // Block is now request-held (refCount 1) and reference() added an extra ref (refCount 2),
    // so it should take two decRef calls to become free again.
    const size_t free_before = block_pool->freeBlocksNum();
    ASSERT_EQ(block_pool->refCount(blocks[0]), 2u);
    block_pool->decRef(blocks[0]);
    EXPECT_EQ(block_pool->refCount(blocks[0]), 1u);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);  // still referenced
    block_pool->decRef(blocks[0]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 1);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
