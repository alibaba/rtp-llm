#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class FullKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== 基础功能测试 ====================

TEST_F(FullKVCacheGroupTest, NeedBlocksNumTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(FullKVCacheGroupTest, RemoveSkippedBlocksTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool);

    BlockIndicesType old_indices   = {1, 2, 3, 4};
    BlockIndicesType block_indices = old_indices;
    group1.removeSkippedBlocks(block_indices);
    ASSERT_EQ(old_indices, block_indices);
}

TEST_F(FullKVCacheGroupTest, MatchTest) {

    auto block_pool = createBlockPool();
    block_pool->init();
    auto block_cache = block_pool->blockCache();

    BlockCacheV1::CacheItem item    = {101, 1, false};
    auto                    result1 = block_cache->put(item);
    EXPECT_TRUE(result1);

    BlockCacheV1::CacheItem item2   = {102, 2, false};
    auto                    result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool);

    // zero math
    CacheKeysType cache_keys    = {103, 104, 105, 106};
    auto          match_result2 = group1.match(cache_keys);
    ASSERT_EQ(match_result2.reuse_length, 0);
    BlockIndicesType expected_result = {};
    ASSERT_EQ(match_result2.block_indices, expected_result);

    // part match
    cache_keys         = {101, 102, 103, 1046};
    auto match_result1 = group1.match(cache_keys);
    ASSERT_EQ(match_result1.reuse_length, 2 * 4);
    expected_result = {1, 2};
    ASSERT_EQ(match_result1.block_indices, expected_result);

    // all match
    BlockCacheV1::CacheItem item3   = {103, 3, false};
    auto                    result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);

    BlockCacheV1::CacheItem item4   = {104, 4, false};
    auto                    result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    cache_keys         = {101, 102, 103, 104};
    auto match_result3 = group1.match(cache_keys);
    ASSERT_EQ(match_result3.reuse_length, 4 * 4);

    expected_result = {1, 2, 3, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result);
}

TEST_F(FullKVCacheGroupTest, MallocFreeTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlockNums(), 9);

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool);

    CacheKeysType    cache_keys = {101, 102, 103};
    BlockIndicesType block_indices;

    group1.malloc(cache_keys, block_indices, 7);
    ASSERT_EQ(block_pool->freeBlockNums(), 5);
    ASSERT_EQ(block_indices.size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_indices, expected_result);

    group1.free(block_indices);
    ASSERT_EQ(block_pool->freeBlockNums(), 9);
}

TEST_F(FullKVCacheGroupTest, InsertIntoCacheTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlockNums(), 9);

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool);

    CacheKeysType    cache_keys = {103, 104, 105, 106};
    BlockIndicesType block_indices;

    group1.malloc(cache_keys, block_indices, 8);
    ASSERT_EQ(block_pool->freeBlockNums(), 5);
    ASSERT_EQ(block_indices.size(), 4);
    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_indices, expected_result);

    group1.insertIntoCache(cache_keys, block_indices, false);

    CacheKeysType cache_keys1   = {107, 108};
    auto          match_result1 = group1.match(cache_keys1);
    ASSERT_EQ(match_result1.reuse_length, 0);

    CacheKeysType cache_keys2   = {103, 104, 107};
    auto          match_result2 = group1.match(cache_keys2);
    ASSERT_EQ(match_result2.reuse_length, 2 * 2);
    BlockIndicesType expected_result2 = {1, 2};
    ASSERT_EQ(match_result2.block_indices, expected_result2);

    CacheKeysType cache_keys3   = {103, 104, 105, 106};
    auto          match_result3 = group1.match(cache_keys3);
    ASSERT_EQ(match_result3.reuse_length, 4 * 2);
    BlockIndicesType expected_result3 = {1, 2, 3, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result3);
}

// TODO, modify these ut after block cache not ref blocks
// TEST_F(FullKVCacheGroupTest, EnsureFreeBlocksTest) {
//     auto block_pool = createBlockPool();
//     block_pool->init();
//     auto block_cache = block_pool->blockCache();
//     ASSERT_EQ(block_pool->freeBlockNums(), 9);

//     FullKVCacheGroup group1({}, spec, block_pool);
//     ASSERT_EQ(2, group1.ensureFreeBlocks(5));
// }

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
