#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class LinearKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== 基础功能测试 ====================

TEST_F(LinearKVCacheGroupTest, MallocFreeTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlockNums(), 9);

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    LinearKVCacheGroup group1({}, spec, block_pool);

    CacheKeysType    cache_keys = {101, 102, 103};
    BlockIndicesType block_indices;

    ASSERT_TRUE(group1.malloc(cache_keys, block_indices, 7));
    ASSERT_EQ(block_pool->freeBlockNums(), 5);
    ASSERT_EQ(block_indices.size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_indices, expected_result);

    group1.free(block_indices);
    ASSERT_EQ(block_pool->freeBlockNums(), 9);

    ASSERT_FALSE(group1.malloc(cache_keys, block_indices, 180));
}

TEST_F(LinearKVCacheGroupTest, RemoveSkippedBlocksTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    LinearKVCacheGroup group1({}, spec, block_pool);

    CacheKeysType    cache_keys = {101, 102, 103, 104};
    BlockIndicesType block_indices1;

    ASSERT_TRUE(group1.malloc(cache_keys, block_indices1, 8));
    ASSERT_EQ(block_pool->freeBlockNums(), 5);
    ASSERT_EQ(block_indices1.size(), 4);

    BlockIndicesType old_indices   = {-1, 2, -1, 4};
    BlockIndicesType block_indices = old_indices;
    group1.removeSkippedBlocks(block_indices);
    ASSERT_NE(old_indices, block_indices);
    ASSERT_EQ(block_indices[0], NULL_BLOCK_IDX);
    ASSERT_EQ(block_indices[1], NULL_BLOCK_IDX);
    ASSERT_EQ(block_indices[2], NULL_BLOCK_IDX);
    ASSERT_EQ(block_indices[3], 4);

    ASSERT_EQ(block_pool->freeBlockNums(), 6);
}

TEST_F(LinearKVCacheGroupTest, MatchTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    auto block_cache = block_pool->blockCache();

    BlockCacheV1::CacheItem item2   = {102, 0, 2, false};
    auto                    result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);

    auto spec                = make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    LinearKVCacheGroup group1({}, spec, block_pool);

    // zero math
    CacheKeysType cache_keys    = {103, 104, 105, 106};
    auto          match_result1 = group1.match(cache_keys);
    ASSERT_EQ(match_result1.reuse_blocks, 0);
    ASSERT_EQ(match_result1.reuse_length, 0);
    BlockIndicesType expected_result = {};
    ASSERT_EQ(match_result1.block_indices, expected_result);

    // part match
    cache_keys         = {101, 102, 103, 104};
    auto match_result2 = group1.match(cache_keys);
    ASSERT_EQ(match_result2.reuse_blocks, 2);
    ASSERT_EQ(match_result2.reuse_length, 2 * 4);
    expected_result = {NULL_BLOCK_IDX, 2};
    ASSERT_EQ(match_result2.block_indices, expected_result);

    // all match
    BlockCacheV1::CacheItem item4   = {104, 0, 4, false};
    auto                    result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    cache_keys         = {101, 102, 103, 104};
    auto match_result3 = group1.match(cache_keys);
    ASSERT_EQ(match_result3.reuse_length, 4 * 4);
    ASSERT_EQ(match_result3.reuse_blocks, 4);

    expected_result = {NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
