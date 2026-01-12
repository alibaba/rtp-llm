#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
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
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(FullKVCacheGroupTest, RemoveSkippedBlocksTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

    BlockIndicesType old_indices   = {1, 2, 3, 4};
    BlockIndicesType block_indices = old_indices;
    group1.removeSkippedBlocks(block_indices);
    ASSERT_EQ(old_indices, block_indices);
}

TEST_F(FullKVCacheGroupTest, MatchTest) {

    auto block_pool = createBlockPool();
    block_pool->init();
    auto block_cache = block_pool->blockCache();

    BlockCache::CacheItem item    = {101, 0, 1, false};
    auto                  result1 = block_cache->put(item);
    EXPECT_TRUE(result1);

    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

    // zero math
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
    BlockCache::CacheItem item3   = {103, 0, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);

    BlockCache::CacheItem item4   = {104, 0, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    cache_keys         = {101, 102, 103, 104};
    auto match_result3 = group1.match(cache_keys);
    ASSERT_EQ(match_result3.reuse_blocks, 4);
    ASSERT_EQ(match_result3.reuse_length, 4 * 4);

    expected_result = {1, 2, 3, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result);
}

TEST_F(FullKVCacheGroupTest, MallocFreeTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->availableBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

    CacheKeysType    cache_keys = {101, 102, 103};
    BlockIndicesType block_indices;

    ASSERT_TRUE(group1.malloc(block_indices, 7));
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
    ASSERT_EQ(block_pool->availableBlocksNum(), 5);
    ASSERT_EQ(block_indices.size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_indices, expected_result);

    group1.free(block_indices);
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->availableBlocksNum(), 9);

    ASSERT_FALSE(group1.malloc(block_indices, 180));
}

TEST_F(FullKVCacheGroupTest, InsertIntoCacheTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->availableBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool, 0);

    CacheKeysType    cache_keys = {103, 104, 105, 106};
    BlockIndicesType block_indices;

    group1.malloc(block_indices, 8);
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
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

TEST_F(FullKVCacheGroupTest, EnsureFreeBlocksTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    auto block_cache  = block_pool->blockCache();
    auto total_blocks = block_pool->freeBlocksNum();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullKVCacheGroup group1({}, spec, block_pool, 0);
    ASSERT_EQ(true, group1.ensureFreeBlocks(5));
    ASSERT_EQ(block_pool->freeBlocksNum(), total_blocks);
    ASSERT_EQ(block_pool->availableBlocksNum(), total_blocks);

    ASSERT_EQ(false, group1.ensureFreeBlocks(10));

    CacheKeysType    cache_keys = {101, 102, 103, 104};
    BlockIndicesType block_indices;

    ASSERT_TRUE(group1.malloc(block_indices, 8));
    ASSERT_EQ(block_indices.size(), 4);
    ASSERT_EQ(block_pool->freeBlocksNum(), total_blocks - 4);
    ASSERT_EQ(block_pool->availableBlocksNum(), total_blocks - 4);

    group1.insertIntoCache(cache_keys, block_indices, false);
    ASSERT_EQ(block_cache->size(), 4);
    ASSERT_EQ(block_pool->freeBlocksNum(), total_blocks - 4);
    ASSERT_EQ(block_pool->availableBlocksNum(), total_blocks - 4);

    group1.free(block_indices);
    ASSERT_EQ(block_cache->size(), 4);
    ASSERT_EQ(block_pool->freeBlocksNum(), total_blocks - 4);
    ASSERT_EQ(block_pool->availableBlocksNum(), total_blocks);

    ASSERT_EQ(true, group1.ensureFreeBlocks(total_blocks - 2));
    ASSERT_EQ(block_cache->size(), 2);
    ASSERT_EQ(block_pool->freeBlocksNum(), total_blocks - 2);
    ASSERT_EQ(block_pool->availableBlocksNum(), total_blocks);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
