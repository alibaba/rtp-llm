#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class DeviceKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== Basic functionality tests ====================

TEST_F(DeviceKVCacheGroupTest, MaterializePositionsDeduplicatesMissingSlotsAndOnlyTakesUnifiedRefs) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    const size_t initial_free = block_pool->freeBlocksNum();

    auto             spec = createTestKvCacheSpec(/*layer_num=*/4,
                                      DataType::TYPE_FP16,
                                      /*local_head_num_kv=*/1,
                                      /*seq_size_per_block=*/4,
                                      /*k_block_stride_bytes=*/512,
                                      /*v_block_stride_bytes=*/512);
    FullKVCacheGroup group({}, spec, block_pool, 0);

    const auto existing = block_pool->malloc(1);
    ASSERT_TRUE(existing.has_value());
    ASSERT_EQ(existing->size(), 1u);
    block_pool->incRef(*existing, BlockRefType::REQUEST);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, existing->front(), NULL_BLOCK_IDX});
    const size_t free_before = block_pool->freeBlocksNum();

    ASSERT_TRUE(group.materializePositions(block_ids, {0, 0, 2, 2}));

    ASSERT_EQ(block_ids.blocksNum(), 3u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_ids.blocks()[1], existing->front());
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_NE(block_ids.blocks()[0], block_ids.blocks()[2]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before - 2);
    for (BlockIdxType block : block_ids.blocks()) {
        EXPECT_EQ(block_pool->refCount(block), 1u);
    }

    group.free(block_ids.blocks());
    EXPECT_EQ(block_pool->freeBlocksNum(), initial_free);
}

TEST_F(DeviceKVCacheGroupTest, MaterializePositionsRejectsOutOfRangeWithoutMutation) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto             spec = createTestKvCacheSpec(/*layer_num=*/4,
                                      DataType::TYPE_FP16,
                                      /*local_head_num_kv=*/1,
                                      /*seq_size_per_block=*/4,
                                      /*k_block_stride_bytes=*/512,
                                      /*v_block_stride_bytes=*/512);
    FullKVCacheGroup group({}, spec, block_pool, 0);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    const auto   before      = block_ids.blocks();
    const size_t free_before = block_pool->freeBlocksNum();

    EXPECT_FALSE(group.materializePositions(block_ids, {0, 2}));
    EXPECT_EQ(block_ids.blocks(), before);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    EXPECT_EQ(block_pool->usedBlocksNum(), 0u);
}

TEST_F(DeviceKVCacheGroupTest, MaterializePositionsRollsBackWhenPoolCannotSatisfyWholeRequest) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    const size_t initial_free = block_pool->freeBlocksNum();

    auto             spec = createTestKvCacheSpec(/*layer_num=*/4,
                                      DataType::TYPE_FP16,
                                      /*local_head_num_kv=*/1,
                                      /*seq_size_per_block=*/4,
                                      /*k_block_stride_bytes=*/512,
                                      /*v_block_stride_bytes=*/512);
    FullKVCacheGroup group({}, spec, block_pool, 0);

    auto pressure = block_pool->malloc(block_pool->freeBlocksNum() - 1);
    ASSERT_TRUE(pressure.has_value());
    block_pool->incRef(*pressure, BlockRefType::REQUEST);
    ASSERT_EQ(block_pool->freeBlocksNum(), 1u);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    const auto   before      = block_ids.blocks();
    const size_t free_before = block_pool->freeBlocksNum();

    EXPECT_FALSE(group.materializePositions(block_ids, {0, 1}));
    EXPECT_EQ(block_ids.blocks(), before);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    for (BlockIdxType block : *pressure) {
        EXPECT_EQ(block_pool->refCount(block), 1u);
    }

    block_pool->decRef(*pressure, BlockRefType::REQUEST);
    EXPECT_EQ(block_pool->freeBlocksNum(), initial_free);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
