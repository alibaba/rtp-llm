#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class KVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== Basic functionality tests ====================

TEST_F(KVCacheGroupTest, MaterializePositionsDeduplicatesMissingSlotsAndOnlyTakesRequestHolds) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    FullKVCacheGroup group({}, spec, block_pool, 0);

    const auto existing = block_pool->malloc(1);
    ASSERT_EQ(existing.size(), 1u);
    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, existing[0], NULL_BLOCK_IDX});

    const size_t free_before = block_pool->freeBlocksNum();
    ASSERT_TRUE(group.materializePositions(block_ids, {0, 0, 2, 2}));

    ASSERT_EQ(block_ids.blocksNum(), 3u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_ids.blocks()[1], existing[0]);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_NE(block_ids.blocks()[0], block_ids.blocks()[2]);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before - 2);
    EXPECT_EQ(block_pool->requestRefBlocksNum(), 3u);
    EXPECT_EQ(block_pool->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(block_pool->blockCacheRefBlocksNum(), 0u);

    group.free(block_ids.blocks());
}

TEST_F(KVCacheGroupTest, MaterializePositionsRejectsOutOfRangeWithoutMutation) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    FullKVCacheGroup group({}, spec, block_pool, 0);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    const auto   before      = block_ids.blocks();
    const size_t free_before = block_pool->freeBlocksNum();

    EXPECT_FALSE(group.materializePositions(block_ids, {0, 2}));
    EXPECT_EQ(block_ids.blocks(), before);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    EXPECT_EQ(block_pool->requestRefBlocksNum(), 0u);
    EXPECT_EQ(block_pool->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(block_pool->blockCacheRefBlocksNum(), 0u);
}

TEST_F(KVCacheGroupTest, MaterializePositionsRollsBackWhenPoolCannotSatisfyWholeRequest) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    FullKVCacheGroup group({}, spec, block_pool, 0);

    const auto pressure = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum() - 1));
    ASSERT_FALSE(pressure.empty());
    ASSERT_EQ(block_pool->freeBlocksNum(), 1u);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    const auto   before              = block_ids.blocks();
    const size_t free_before         = block_pool->freeBlocksNum();
    const size_t request_refs_before = block_pool->requestRefBlocksNum();

    EXPECT_FALSE(group.materializePositions(block_ids, {0, 1}));
    EXPECT_EQ(block_ids.blocks(), before);
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before);
    EXPECT_EQ(block_pool->requestRefBlocksNum(), request_refs_before);
    EXPECT_EQ(block_pool->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(block_pool->blockCacheRefBlocksNum(), 0u);

    block_pool->requestFree(pressure);
}

TEST_F(KVCacheGroupTest, EnsureFreeBlocksWithoutTreeCallbackRetainsLegacySharedCacheFallback) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());
    auto shared_cache = std::make_shared<SharedBlockCache>();
    shared_cache->init(/*group_num=*/1, std::vector<BlockPoolPtr>{block_pool});

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = "default";
    spec->seq_size_per_block = 4;
    GroupBase group_config;
    group_config.tag  = "default";
    group_config.spec = spec;
    FullKVCacheGroup group(group_config, block_pool, /*group_id=*/0, shared_cache.get());
    ASSERT_TRUE(group.init());

    const auto all_blocks = block_pool->malloc(static_cast<int>(block_pool->freeBlocksNum()));
    ASSERT_FALSE(all_blocks.empty());
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);
    const BlockIdxType cached_block = all_blocks.front();
    shared_cache->put(/*cache_key=*/100, BlockIndicesType{cached_block}, /*is_resident=*/false);
    block_pool->requestFree(cached_block);
    ASSERT_EQ(block_pool->freeBlocksNum(), 0u);
    ASSERT_FALSE(shared_cache->empty());

    EXPECT_TRUE(group.ensureFreeBlocks(/*required_blocks=*/1));
    EXPECT_EQ(block_pool->freeBlocksNum(), 1u);
    EXPECT_TRUE(shared_cache->empty());

    block_pool->requestFree(BlockIndicesType(all_blocks.begin() + 1, all_blocks.end()));
    EXPECT_EQ(block_pool->freeBlocksNum(), block_pool->totalBlocksNum());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
