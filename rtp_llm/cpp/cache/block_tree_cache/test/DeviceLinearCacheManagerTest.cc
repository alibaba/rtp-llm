#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/LinearCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace test {

using DeviceLinearCacheManager = LinearCacheManager;

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

class DeviceLinearCacheManagerTest: public ::testing::Test {};

TEST_F(DeviceLinearCacheManagerTest, DefaultPolicyDrivesBehaviorInterfaces) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);

    EXPECT_TRUE(group.prefixReusable());
    EXPECT_TRUE(group.hasSparseSlots());
    EXPECT_FALSE(group.hasKernelBlockSubdiv());
    EXPECT_TRUE(group.transferTailBlocks());
    EXPECT_TRUE(group.isReservable());
    EXPECT_NE(group.memoryPlacement(), CacheMemoryPlacement::HOST_PINNED);
}

TEST_F(DeviceLinearCacheManagerTest, PrefillNeedBlocksIgnoresDecodeReserveWithReuseDisabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 1);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceLinearCacheManagerTest, PrefillNeedBlocksIgnoresDecodeReserveWithReuseEnabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/2, /*reuse_blocks_len=*/0, true);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceLinearCacheManagerTest, MallocAllocatesOnlyTailWhenReuseEnabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/true));  // 4 slots

    ASSERT_EQ(blocks.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));

    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(DeviceLinearCacheManagerTest, MallocAllocatesActiveTailWhenReuseDisabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

TEST_F(DeviceLinearCacheManagerTest, MallocRequiredPositionsBackfillsPolicySkippedSlot) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    blocks.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    std::vector<size_t> backfilled_positions;
    ASSERT_TRUE(group.malloc(blocks,
                             /*seq_len=*/16,
                             /*enable_reuse_cache=*/false,
                             /*reserve_step=*/0,
                             &backfilled_positions,
                             /*required_positions=*/{1}));

    ASSERT_EQ(blocks.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_EQ(backfilled_positions, (std::vector<size_t>{1}));
    EXPECT_EQ(block_pool->freeBlocksNum(), 8u);

    group.unreference(blocks.blocks());
}

TEST_F(DeviceLinearCacheManagerTest, DecodeAllocatesReserveTailBlocksWhenReuseDisabled) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    // seq_len=16 => seq_slots=4; reserve_step=2 => total_slots=5
    BlockIds blocks;
    ASSERT_TRUE(group.mallocForDecode(blocks, /*seq_len=*/16, /*enable_reuse_cache=*/false, /*reserve_step=*/2));

    ASSERT_EQ(blocks.blocksNum(), 5u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));  // seq tail
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));  // reserve tail

    EXPECT_EQ(block_pool->freeBlocksNum(), 7u);
}

TEST_F(DeviceLinearCacheManagerTest, PrefillDoesNotBackfillSlotsBeforeReuseHit) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(2).value();
    ASSERT_EQ(allocated.size(), 2u);
    // New pool malloc reserves capacity at refCount 0; take a request ref so these
    // pre-existing request blocks are held while the group backfills the read slot.
    block_pool->incRef(allocated);

    BlockIds blocks;
    blocks.assign(BlockIndicesType{allocated[0], NULL_BLOCK_IDX, allocated[1]});
    const size_t free_before = block_pool->freeBlocksNum();

    ASSERT_TRUE(group.malloc(blocks, /*seq_len=*/12, /*enable_reuse_cache=*/false));

    ASSERT_EQ(blocks.blocksNum(), 3u);
    EXPECT_EQ(blocks.blocks()[0], allocated[0]);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_EQ(blocks.blocks()[2], allocated[1]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    group.unreference(blocks.blocks());
}

TEST_F(DeviceLinearCacheManagerTest, MallocMaterializesCausalConvReadSlotAtBoundaries) {
    const std::vector<int> seq_lens = {4, 5, 8, 9};

    for (bool enable_reuse_cache : {false, true}) {
        for (int seq_len : seq_lens) {
            auto block_pool = createDeviceBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
            DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

TEST_F(DeviceLinearCacheManagerTest, GetNeedBlocksMatchesMallocForReserveSteps) {
    for (bool enable_reuse_cache : {false, true}) {
        for (int reserve_step : {0, 1, 2, 3}) {
            auto block_pool = createDeviceBlockPool();
            ASSERT_TRUE(block_pool->init());

            auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
            DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

TEST_F(DeviceLinearCacheManagerTest, DecodeCleanupKeepsReadStateAndCurrentState) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 4);
}

TEST_F(DeviceLinearCacheManagerTest, MallocNoNewBlocksReturnsTrueAndKeepsState) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

TEST_F(DeviceLinearCacheManagerTest, MallocFailsWhenBlockPoolExhausted) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    // Exhaust all free blocks (block 0 is reserved).
    auto all_blocks = block_pool->malloc(block_pool->freeBlocksNum()).value();
    // Hold a request ref so the cleanup decRef() below has a holder to drop.
    block_pool->incRef(all_blocks);
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    BlockIds blocks;
    EXPECT_FALSE(group.malloc(blocks, /*seq_len=*/4, /*enable_reuse_cache=*/false));

    // Cleanup to avoid leaking refs in the test process.
    block_pool->decRef(all_blocks);
}

TEST_F(DeviceLinearCacheManagerTest, DecodeCleanupRetainsReserveSlotsIndependentlyOfPrefillTail) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    // Hold a request ref on each block; removeSkippedBlocks() releases pruned blocks via
    // decRef(), which requires refCount > 0 (new pool malloc leaves them at 0).
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);  // no NULLs

    const size_t free_before = block_pool->freeBlocksNum();
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

TEST_F(DeviceLinearCacheManagerTest, FreeIgnoresEmptyOrAllNullBlocks) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
    ASSERT_TRUE(group.init());

    const size_t free_before = block_pool->freeBlocksNum();
    group.unreference(BlockIndicesType{});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);

    group.unreference(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
}

TEST_F(DeviceLinearCacheManagerTest, ReferenceAppendsAndIncrementsRefCountForValidBlocks) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto                     spec = makeLinearSpec(/*seq_size_per_block=*/4);
    DeviceLinearCacheManager group(/*layer_ids=*/{}, spec, block_pool, /*group_id=*/0, /*tail_blocks=*/2);
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

TEST_F(DeviceLinearCacheManagerTest, CommittedHistoryReclaimPreservesActiveAndSpeculativeStates) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    // INT_MAX tail window keeps every sequence checkpoint materialized, the
    // pre-#1574 semantics these reclaim assertions were written against.
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, std::numeric_limits<int>::max());
    ASSERT_TRUE(group.init());
    BlockIds blocks;
    // Six sequence checkpoints (reserve slots materialize lazily since #1574).
    ASSERT_TRUE(group.malloc(blocks, 24, true, 3));
    const auto original = blocks.blocks();
    EXPECT_EQ(group.historyReclaimEnd(20), 3u);
    group.reclaimHistory(blocks, group.historyReclaimEnd(20));
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(blocks.blocks()[i], NULL_BLOCK_IDX);
    }
    for (size_t i = 3; i < original.size(); ++i) {
        EXPECT_EQ(blocks.blocks()[i], original[i]);
    }
    EXPECT_EQ(pool->referencedBlocksNum(), 3u);
    EXPECT_EQ(pool->freeBlocksNum(), 6u);
    group.reclaimHistory(blocks, group.historyReclaimEnd(16));
    EXPECT_EQ(pool->referencedBlocksNum(), 3u);
    // Reallocation must not backfill retired stride-1 checkpoints.
    ASSERT_TRUE(group.malloc(blocks, 28, true, 3));
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(blocks.blocks()[i], NULL_BLOCK_IDX);
    }
    EXPECT_EQ(pool->referencedBlocksNum(), 4u);
    group.unreference(blocks.blocks());
    EXPECT_EQ(pool->freeBlocksNum(), 9u);
}

TEST_F(DeviceLinearCacheManagerTest, ReclaimDoesNotTurnAnOldReadIntoAnUninitializedState) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, std::numeric_limits<int>::max());
    ASSERT_TRUE(group.init());
    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, 24, true));
    group.reclaimHistory(blocks, group.historyReclaimEnd(20));
    const auto free_before = pool->freeBlocksNum();
    EXPECT_FALSE(group.malloc(blocks, 8, true));
    EXPECT_FALSE(group.malloc(blocks, 24, true, 0, nullptr, {0}));
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
    group.unreference(blocks.blocks());
}

TEST_F(DeviceLinearCacheManagerTest, ReclaimRespectsModelTailPolicyAndPartialCheckpoint) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.active_tail_blocks = 3;
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, std::numeric_limits<int>::max(), policy);
    EXPECT_EQ(group.historyReclaimEnd(15), 0u);
    EXPECT_EQ(group.historyReclaimEnd(19), 1u);
    EXPECT_EQ(group.historyReclaimEnd(-1), 0u);
}

TEST_F(DeviceLinearCacheManagerTest, HistoryForecastBoundsDecodeButPreservesPrefillPeak) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, std::numeric_limits<int>::max());
    ASSERT_TRUE(group.init());
    const int short_peak = group.estimateInitialBatchPeakNeedBlocks(8, 8, 128, 0, true, 1, true);
    const int long_peak  = group.estimateInitialBatchPeakNeedBlocks(8, 8, 4096, 0, true, 1, true);
    EXPECT_EQ(short_peak, long_peak);
    EXPECT_LE(long_peak, 5);
    // Since #1574 both forecasts bound the same retained-decode window, so
    // the reclaim forecast may tie but must never exceed the plain forecast.
    EXPECT_LE(long_peak, group.estimateInitialBatchPeakNeedBlocks(8, 8, 4096, 0, true, 1, false));
    // A long prompt must be materialized before any decode checkpoint can retire.
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(128, 128, 0, 0, true, 1, true), 32);
    EXPECT_GE(group.estimateInitialBatchPeakNeedBlocks(128, 128, 4096, 0, true, 1, true), 32);
    // Forecasting history reclaim must not change the off path or batch expansion.
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(8, 8, 128, 0, true, 2, true),
              group.estimateInitialBatchPeakNeedBlocks(8, 8, 128, 0, true, 2, false));
}

TEST_F(DeviceLinearCacheManagerTest, HistoryForecastCoversRealAllocationWithPinnedInitialBlocks) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    // The prefill materialization window (main #1574's tail_blocks) replaces
    // the retired linear_step reclaim stride as the sizing knob under test.
    for (const int window : {1, 2, 3}) {
        for (const int tail : {1, 3}) {
            auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
            policy.active_tail_blocks = tail;
            DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, window, policy);
            ASSERT_TRUE(group.init());
            for (const int reserve : {0, 2}) {
                for (const int initial : {3, 4, 5}) {
                    SCOPED_TRACE(::testing::Message() << "window=" << window << " tail=" << tail
                                                      << " reserve=" << reserve << " initial=" << initial);
                    BlockIds blocks;
                    ASSERT_TRUE(group.malloc(blocks, initial, true, reserve));
                    BlockIndicesType pinned;
                    for (auto block : blocks.blocks()) {
                        if (!isNullBlockIdx(block)) {
                            pinned.push_back(block);
                        }
                    }
                    group.reference(pinned);
                    const int baseline = static_cast<int>(pool->referencedBlocksNum());
                    const int forecast =
                        group.estimatePeakNeedBlocks(initial, blocks.blocks(), 128, reserve, true, true);
                    int observed_peak = baseline;
                    for (int length = initial + 1; length <= initial + 128; ++length) {
                        group.reclaimHistory(blocks, group.historyReclaimEnd(length - 1));
                        ASSERT_TRUE(group.malloc(blocks, length, true, reserve));
                        observed_peak = std::max(observed_peak, static_cast<int>(pool->referencedBlocksNum()));
                        group.removeSkippedBlocks(blocks, true, reserve);
                    }
                    EXPECT_GE(forecast, observed_peak - baseline);
                    EXPECT_LE(forecast, 7);
                    group.unreference(blocks.blocks());
                    group.unreference(pinned);
                    EXPECT_EQ(pool->freeBlocksNum(), 9u);
                }
            }
        }
    }
}

TEST_F(DeviceLinearCacheManagerTest, TailIsClampedOnlyForReusablePrefill) {
    for (const int tail : {1, 2, 3, 5, std::numeric_limits<int>::max()}) {
        for (const bool reuse : {false, true}) {
            auto pool = createDeviceBlockPool();
            ASSERT_TRUE(pool->init());
            DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, tail);
            ASSERT_TRUE(group.init());
            BlockIds blocks;
            ASSERT_TRUE(group.malloc(blocks, 20, reuse));
            const int retained = reuse ? std::min(tail, 5) : 1;
            EXPECT_EQ(pool->freeBlocksNum(), 9u - retained);
            for (int i = 0; i < 5; ++i) {
                EXPECT_EQ(!isNullBlockIdx(blocks.blocks()[i]), i >= 5 - retained);
            }
            group.unreference(blocks.blocks());
            EXPECT_EQ(pool->freeBlocksNum(), 9u);
        }
    }
}

TEST_F(DeviceLinearCacheManagerTest, DecodeRetentionIsIndependentOfPrefillTail) {
    for (const int tail : {1, 3, std::numeric_limits<int>::max()}) {
        auto pool = createDeviceBlockPool();
        ASSERT_TRUE(pool->init());
        DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, tail);
        ASSERT_TRUE(group.init());
        BlockIds blocks;
        ASSERT_TRUE(group.malloc(blocks, 12, true));
        const int initial_count = std::min(tail, 3);
        EXPECT_EQ(group.estimatePeakNeedBlocks(12, blocks.blocks(), 100, 0, true),
                  std::max(initial_count + 1, 3) - initial_count);
        for (int count = 4; count <= 20; ++count) {
            const auto previous_tail = blocks.blocks().back();
            ASSERT_TRUE(group.mallocForDecode(blocks, count * 4, true));
            group.removeSkippedBlocks(blocks, true);
            EXPECT_EQ(pool->referencedBlocksNum(), 2u);
            EXPECT_EQ(blocks.blocks()[count - 2], previous_tail);
            EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[count - 3]));
        }
        group.unreference(blocks.blocks());
        EXPECT_EQ(pool->freeBlocksNum(), 9u);
    }
}

TEST_F(DeviceLinearCacheManagerTest, DecodeReserveAndReadStateDoNotDependOnPrefillTail) {
    for (const int tail : {1, 6, std::numeric_limits<int>::max()}) {
        auto pool = createDeviceBlockPool();
        ASSERT_TRUE(pool->init());
        DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, tail);
        ASSERT_TRUE(group.init());
        BlockIds blocks;
        ASSERT_TRUE(group.malloc(blocks, 8, true, 4));
        EXPECT_EQ(blocks.blocksNum(), 2u);
        EXPECT_EQ(pool->referencedBlocksNum(), static_cast<size_t>(std::min(tail, 2)));
        for (int count = 3; count <= 10; ++count) {
            const auto read_state = blocks.blocks()[count - 2];
            ASSERT_FALSE(isNullBlockIdx(read_state));
            ASSERT_TRUE(group.mallocForDecode(blocks, count * 4, true, 4));
            group.removeSkippedBlocks(blocks, true, 4);
            EXPECT_EQ(blocks.blocks()[count - 2], read_state);
            EXPECT_EQ(blocks.blocksNum(), static_cast<size_t>(count + 3));
            EXPECT_LE(pool->referencedBlocksNum(), 6u);
            for (int slot = count - 2; slot < count + 3; ++slot) {
                EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[slot]));
            }
        }
        group.unreference(blocks.blocks());
        EXPECT_EQ(pool->freeBlocksNum(), 9u);
    }
}

TEST_F(DeviceLinearCacheManagerTest, PrefillAllocatesOnlyTailAfterReuseHit) {
    for (const int tail : {1, 3, std::numeric_limits<int>::max()}) {
        for (const int reuse_slots : {1, 2, 3, 4, 5, 6}) {
            for (const bool remote_load : {false, true}) {
                SCOPED_TRACE(::testing::Message() << tail << " reuse=" << reuse_slots << " remote=" << remote_load);
                auto pool = createDeviceBlockPool();
                ASSERT_TRUE(pool->init());
                DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, tail);
                ASSERT_TRUE(group.init());
                BlockIds blocks;
                blocks.resize(reuse_slots);
                RequiredPositions required;
                BlockIdxType      reused = NULL_BLOCK_IDX;
                if (remote_load) {
                    required.insert(reuse_slots - 1);
                } else {
                    const auto allocated = pool->malloc(1).value();
                    pool->incRef(allocated);
                    reused = allocated.front();
                    blocks.setAt(reuse_slots - 1, reused);
                }
                const int  new_blocks = std::min(tail, 6 - reuse_slots);
                const auto need       = group.getNeedBlocks(24, 24, 4, reuse_slots, true, required);
                EXPECT_EQ(need.common_blocks, new_blocks + static_cast<int>(remote_load));
                EXPECT_EQ(need.extra_blocks, 0);
                ASSERT_TRUE(group.malloc(blocks, 24, true, 4, nullptr, required));
                EXPECT_EQ(blocks.blocksNum(), 6u);
                EXPECT_EQ(pool->referencedBlocksNum(), static_cast<size_t>(new_blocks + 1));
                for (int pos = 0; pos < 6; ++pos) {
                    EXPECT_EQ(!isNullBlockIdx(blocks.blocks()[pos]), pos == reuse_slots - 1 || pos >= 6 - new_blocks);
                }
                if (!remote_load) {
                    EXPECT_EQ(blocks.blocks()[reuse_slots - 1], reused);
                }
                group.unreference(blocks.blocks());
                EXPECT_EQ(pool->freeBlocksNum(), 9u);
            }
        }
    }
}

TEST_F(DeviceLinearCacheManagerTest, TailRetentionDoesNotChangeTransferPolicy) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, 10000);
    EXPECT_EQ(group.activeTailBlocks(), 1u);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(20, 20, 0, 0, true, 1), 5);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(20, 20, 0, 0, false, 1), 1);
}

TEST_F(DeviceLinearCacheManagerTest, NonReusableGroupIgnoresPrefillTailConfig) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.enable_prefix_reuse = false;
    policy.active_tail_blocks  = 2;
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, 10000, policy);
    BlockIds                 blocks;
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(20, 20, 0, 4, true, 1), 2);
    const auto need = group.getNeedBlocks(20, 20, 4, 0, true);
    EXPECT_EQ(need.common_blocks + need.extra_blocks, 2);
    ASSERT_TRUE(group.malloc(blocks, 20, true, 4));
    EXPECT_EQ(pool->referencedBlocksNum(), 2u);
    ASSERT_TRUE(group.mallocForDecode(blocks, 24, true));
    group.removeSkippedBlocks(blocks, true);
    EXPECT_EQ(pool->referencedBlocksNum(), 2u);
    group.unreference(blocks.blocks());
}

TEST_F(DeviceLinearCacheManagerTest, SmallPrefillTailPreservesActiveBlocksThroughFirstDecode) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.active_tail_blocks = 2;
    DeviceLinearCacheManager group({}, makeLinearSpec(4), pool, 0, 1, policy);
    BlockIds                 blocks;
    ASSERT_TRUE(group.malloc(blocks, 21, true));
    ASSERT_EQ(pool->referencedBlocksNum(), 2u);
    const auto prefill_blocks = blocks.blocks();
    EXPECT_EQ(group.getNeedBlocks(21, 21, 0, 0, true).common_blocks, 2);
    EXPECT_EQ(group.estimatePeakNeedBlocks(21, blocks.blocks(), 1, 0, true), 0);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(21, 21, 1, 0, true, 1), 2);
    ASSERT_TRUE(group.mallocForDecode(blocks, 22, true));
    EXPECT_EQ(blocks.blocks(), prefill_blocks);
    group.removeSkippedBlocks(blocks, true);
    EXPECT_EQ(pool->referencedBlocksNum(), 2u);
    group.unreference(blocks.blocks());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
