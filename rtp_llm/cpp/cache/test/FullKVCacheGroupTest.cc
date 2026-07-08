#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/group/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class FullKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== Basic functionality tests ====================

TEST_F(FullKVCacheGroupTest, NeedBlocksNumTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(FullKVCacheGroupTest, GetNeedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group({}, spec, block_pool, 0);

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

TEST_F(FullKVCacheGroupTest, RemoveSkippedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

    BlockIndicesType old_indices = {1, 2, 3, 4};
    BlockIds         block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign(old_indices);
    group1.removeSkippedBlocks(block_ids);
    ASSERT_EQ(old_indices, block_ids.blocks());
}

TEST_F(FullKVCacheGroupTest, MatchTest) {

    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto                            shared_cache = std::make_shared<SharedBlockCache>();
    std::vector<DeviceBlockPoolPtr> group_pools  = {block_pool};
    shared_cache->init(1, group_pools);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0, shared_cache.get());

    // Put items into shared cache: cache_key -> group_slots (group 0 = block_idx)
    shared_cache->put(101, {1}, false);
    shared_cache->put(102, {2}, false);

    // zero match
    CacheKeysType cache_keys    = {103, 104, 105, 106};
    auto          match_result1 = group1.match(cache_keys);
    ASSERT_EQ(match_result1.reuse_blocks, 0);
    ASSERT_EQ(match_result1.reuse_length, 0);
    BlockIndicesType expected_result = {};
    ASSERT_EQ(match_result1.block_indices, expected_result);

    // part match
    cache_keys         = {101, 102, 103, 1046};
    auto match_result2 = group1.match(cache_keys);
    ASSERT_EQ(match_result2.reuse_blocks, 2);
    ASSERT_EQ(match_result2.reuse_length, 2 * 4);
    expected_result = {1, 2};
    ASSERT_EQ(match_result2.block_indices, expected_result);

    // all match
    shared_cache->put(103, {3}, false);
    shared_cache->put(104, {4}, false);

    cache_keys         = {101, 102, 103, 104};
    auto match_result3 = group1.match(cache_keys);
    ASSERT_EQ(match_result3.reuse_blocks, 4);
    ASSERT_EQ(match_result3.reuse_length, 4 * 4);

    expected_result = {1, 2, 3, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result);
}

TEST_F(FullKVCacheGroupTest, MallocFreeTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

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

// Single-count co-hold: a block held by both a request (via group malloc) and a cache
// holder (extra incRef) must survive the request release and only free on the final
// releaseRef.
TEST_F(FullKVCacheGroupTest, RequestReleaseKeepsCacheHeldBlock) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

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

    block_pool->releaseRef(block);  // release cache holder
    EXPECT_FALSE(block_pool->isAllocated(block));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
