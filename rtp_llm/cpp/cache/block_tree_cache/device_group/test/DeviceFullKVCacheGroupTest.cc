#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceFullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class DeviceFullKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== Basic functionality tests ====================

TEST_F(DeviceFullKVCacheGroupTest, NeedBlocksNumTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullKVCacheGroup group1({}, spec, block_pool, 0);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(DeviceFullKVCacheGroupTest, GetNeedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullKVCacheGroup group({}, spec, block_pool, 0);

    // common=8 => 2 blocks, seq=12 reserve=3 => ceil(15/4)=4 blocks => extra=2
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/3, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);

    // no reserve: common=12 => 3, seq=12 => 3 => extra=0
    const auto need2 =
        group.getNeedBlocks(/*common_seq_len=*/12, /*seq_len=*/12, /*reserve_step=*/0, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need2.common_blocks, 3);
    EXPECT_EQ(need2.extra_blocks, 0);
}

TEST_F(DeviceFullKVCacheGroupTest, RemoveSkippedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullKVCacheGroup group1({}, spec, block_pool, 0);

    BlockIndicesType old_indices = {1, 2, 3, 4};
    BlockIds         block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign(old_indices);
    group1.removeSkippedBlocks(block_ids);
    ASSERT_EQ(old_indices, block_ids.blocks());
}

TEST_F(DeviceFullKVCacheGroupTest, MallocFreeTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    DeviceFullKVCacheGroup group1({}, spec, block_pool, 0);

    CacheKeysType cache_keys = {101, 102, 103};
    BlockIds      block_ids(/*kernel_blocks_per_kv_block=*/1);

    ASSERT_TRUE(group1.malloc(block_ids, 7));
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
    ASSERT_EQ(block_ids.blocks().size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_ids.blocks(), expected_result);

    group1.free(block_ids.blocks());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);

    BlockIds block_ids2(/*kernel_blocks_per_kv_block=*/1);
    ASSERT_FALSE(group1.malloc(block_ids2, 180));
}

TEST_F(DeviceFullKVCacheGroupTest, MallocBackfillsMatchedLoadBackPlaceholder) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;
    DeviceFullKVCacheGroup group({}, spec, block_pool, 0);

    auto resident = block_pool->malloc();
    ASSERT_TRUE(resident.has_value());
    block_pool->incRef(*resident);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, *resident});
    ASSERT_TRUE(group.malloc(block_ids, /*seq_len=*/4));

    ASSERT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_ids.blocks()[1], *resident);
    EXPECT_EQ(block_pool->refCount(block_ids.blocks()[0]), 1u);
    EXPECT_EQ(block_pool->refCount(*resident), 1u);

    group.free(block_ids.blocks());
}

// Single-count co-hold: a block held by both a request (via group malloc) and a cache
// holder (extra incRef) must survive the request release and only free on the final
// decRef.
TEST_F(DeviceFullKVCacheGroupTest, RequestReleaseKeepsCacheHeldBlock) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    DeviceFullKVCacheGroup group1({}, spec, block_pool, 0);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    ASSERT_TRUE(group1.malloc(block_ids, /*seq_len=*/2));
    ASSERT_FALSE(block_ids.blocks().empty());
    const auto block = block_ids.blocks()[0];
    EXPECT_EQ(block_pool->refCount(block), 1u);  // request holder

    block_pool->incRef(block);  // additional cache holder
    EXPECT_EQ(block_pool->refCount(block), 2u);

    group1.free(BlockIndicesType{block});  // release request holder
    EXPECT_TRUE(block_pool->isAllocated(block));
    EXPECT_EQ(block_pool->refCount(block), 1u);

    block_pool->decRef(block);  // release cache holder
    EXPECT_FALSE(block_pool->isAllocated(block));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
