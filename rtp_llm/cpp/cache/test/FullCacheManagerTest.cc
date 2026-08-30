#include <gtest/gtest.h>
#include <deque>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/FullCacheManager.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class FullCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

static const CacheGroup& makeTestFullGroup(KVCacheSpecPtr spec) {
    static std::deque<CacheGroup> groups;
    CacheGroup                    group;
    group.tag                   = "full";
    group.spec                  = std::move(spec);
    group.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.kv_block_stride_bytes = group.spec->block_size_bytes();
    group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
    groups.push_back(std::move(group));
    return groups.back();
}

// ==================== Basic functionality tests ====================

TEST_F(FullCacheManagerTest, NeedBlocksNumTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullCacheManager group1(makeTestFullGroup(spec), block_pool);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(FullCacheManagerTest, CopiesCanonicalGroupAndRetainsSemanticTag) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    CacheGroup cache_group   = makeTestFullGroup(spec);

    FullCacheManager group(cache_group, block_pool);

    EXPECT_NE(&group.config(), &cache_group);
    EXPECT_EQ(group.config().spec, cache_group.spec);
    EXPECT_EQ(group.config().policy.group_type, cache_group.policy.group_type);
    EXPECT_EQ(group.tag(), "full");

    cache_group.tag = "mutated-after-construction";
    EXPECT_EQ(group.tag(), "full");
}

TEST_F(FullCacheManagerTest, GetNeedBlocksTest) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullCacheManager group(makeTestFullGroup(spec), block_pool);

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

TEST_F(FullCacheManagerTest, RemoveSkippedBlocksTest) {
    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    FullCacheManager group1(makeTestFullGroup(spec), block_pool);

    BlockIndicesType old_indices = {1, 2, 3, 4};
    BlockIds         block_ids;
    block_ids.assign(old_indices);
    group1.removeSkippedBlocks(block_ids);
    ASSERT_EQ(old_indices, block_ids.blocks());
}

TEST_F(FullCacheManagerTest, MatchTest) {

    auto block_pool = createBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    CacheGroup cache_group   = makeTestFullGroup(spec);

    CacheConfig cache_config({cache_group}, {{"full"}}, 1);
    auto        shared_cache = std::make_shared<SharedBlockCache>();
    shared_cache->init(cache_config, {{"full", block_pool}});

    FullCacheManager group1(cache_group, block_pool, shared_cache.get());

    // Put items into shared cache: cache_key -> blocks_by_group (group 0 = block_idx)
    shared_cache->put(101, {{"full", 1}}, false);
    shared_cache->put(102, {{"full", 2}}, false);

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
    shared_cache->put(103, {{"full", 3}}, false);
    shared_cache->put(104, {{"full", 4}}, false);

    cache_keys         = {101, 102, 103, 104};
    auto match_result3 = group1.match(cache_keys);
    ASSERT_EQ(match_result3.reuse_blocks, 4);
    ASSERT_EQ(match_result3.reuse_length, 4 * 4);

    expected_result = {1, 2, 3, 4};
    ASSERT_EQ(match_result3.block_indices, expected_result);
}

TEST_F(FullCacheManagerTest, MallocFreeTest) {
    auto block_pool = createBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->availableBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    FullCacheManager group1(makeTestFullGroup(spec), block_pool);

    CacheKeysType cache_keys = {101, 102, 103};
    BlockIds      block_ids;

    ASSERT_TRUE(group1.malloc(block_ids, 7));
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
    ASSERT_EQ(block_pool->availableBlocksNum(), 5);
    ASSERT_EQ(block_ids.blocks().size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_ids.blocks(), expected_result);

    group1.free(block_ids.blocks());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);
    ASSERT_EQ(block_pool->availableBlocksNum(), 9);

    BlockIds block_ids2;
    ASSERT_FALSE(group1.malloc(block_ids2, 180));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
