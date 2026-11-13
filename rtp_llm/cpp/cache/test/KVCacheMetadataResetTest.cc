#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

// Exercises BlockPool::resetMetadata + BlockCache::clear/generation.
// BlockPool part needs a GPU (the pool buffer is allocated via torch::empty(kCUDA)).
class KVCacheMetadataResetTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();
    }

    BlockPoolPtr makePool() {
        auto pool = std::make_shared<BlockPool>(createTestConfig());
        EXPECT_TRUE(pool->init());
        return pool;
    }
};

// ==================== BlockPool::resetMetadata ====================

TEST_F(KVCacheMetadataResetTest, ResetMetadataRejectsLiveRequestRefsWithoutMutation) {
    auto pool   = makePool();
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const auto free_before = pool->freeBlocksNum();
    EXPECT_THROW(pool->resetMetadata(), std::runtime_error);
    EXPECT_EQ(pool->requestRefBlocksNum(), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
    if (pool->requestRefBlocksNum() != 0) {
        pool->requestFree(blocks);
    }
}

TEST_F(KVCacheMetadataResetTest, ResetMetadataRejectsLiveConnectorRefsWithoutMutation) {
    auto pool   = makePool();
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    pool->connectorReference(blocks);
    pool->requestFree(blocks);
    const auto free_before = pool->freeBlocksNum();
    EXPECT_THROW(pool->resetMetadata(), std::runtime_error);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
    if (pool->connectorRefBlocksNum() != 0) {
        pool->connectorFree(blocks);
    }
}

TEST_F(KVCacheMetadataResetTest, ResetMetadataAllowsRetainedCacheRefs) {
    auto pool   = makePool();
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    pool->blockCacheReference(blocks);
    pool->requestFree(blocks);
    ASSERT_EQ(pool->requestRefBlocksNum(), 0u);
    ASSERT_EQ(pool->connectorRefBlocksNum(), 0u);
    ASSERT_EQ(pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_NO_THROW(pool->resetMetadata());
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

TEST_F(KVCacheMetadataResetTest, ReleaseHostBufferRejectsLiveReferencesWithoutMutation) {
    auto pool = std::make_shared<BlockPool>(createTestConfig(), AllocationType::HOST);
    ASSERT_TRUE(pool->init());
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const auto base_before = pool->getBaseAddress();
    EXPECT_THROW(pool->releaseHostBuffer(), std::runtime_error);
    EXPECT_EQ(pool->getBaseAddress(), base_before);
    pool->connectorReference(blocks);
    pool->requestFree(blocks);
    EXPECT_THROW(pool->releaseHostBuffer(), std::runtime_error);
    EXPECT_EQ(pool->getBaseAddress(), base_before);
    pool->connectorFree(blocks);
    EXPECT_NO_THROW(pool->releaseHostBuffer());
    EXPECT_EQ(pool->getBaseAddress(), nullptr);
    EXPECT_NO_THROW(pool->reallocateHostBuffer());
    EXPECT_NE(pool->getBaseAddress(), nullptr);
}

TEST_F(KVCacheMetadataResetTest, ResetMetadataRestoresFreshPoolState) {
    auto pool       = makePool();
    auto fresh_pool = makePool();  // reference: never-touched pool with the same config

    const size_t total = pool->totalBlocksNum();
    ASSERT_EQ(total, fresh_pool->totalBlocksNum());

    // Dirty the pool: request refs, block-cache refs, connector refs.
    auto blocks = pool->malloc(3);
    ASSERT_EQ(blocks.size(), 3u);
    pool->blockCacheReference(blocks[0]);
    pool->connectorReference(blocks[1]);
    // double request-ref one block to get a refcount > 1
    pool->requestReference(blocks[2]);

    ASSERT_LT(pool->freeBlocksNum(), total);
    ASSERT_GT(pool->requestRefBlocksNum(), 0u);
    ASSERT_GT(pool->blockCacheRefBlocksNum(), 0u);
    ASSERT_GT(pool->connectorRefBlocksNum(), 0u);
    ASSERT_LT(pool->availableBlocksNum(), total);

    // Destructive lifecycle operations require active owners to finish first.
    // Retained cache references are intentionally left for the discard reset.
    pool->requestFree(blocks);
    pool->requestFree(blocks[2]);
    pool->connectorFree(blocks[1]);

    void* base_before  = pool->getBaseAddress();
    auto  cache_before = pool->blockCache();

    pool->resetMetadata();

    // Equivalent to a fresh pool: full free set, all ref counters zeroed.
    EXPECT_EQ(pool->freeBlocksNum(), fresh_pool->freeBlocksNum());
    EXPECT_EQ(pool->freeBlocksNum(), total);
    EXPECT_EQ(pool->availableBlocksNum(), fresh_pool->availableBlocksNum());
    EXPECT_EQ(pool->notInUseBlocksNum(), fresh_pool->notInUseBlocksNum());
    EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(pool->blockCacheRefBlocksNum(), 0u);

    // -fno-access-control: compare internals against the fresh pool directly.
    EXPECT_EQ(pool->free_block_ids_, fresh_pool->free_block_ids_);
    for (const auto block_id : pool->free_block_ids_) {
        EXPECT_EQ(pool->request_ref_counter_.getRefCounter(block_id), 0);
        EXPECT_EQ(pool->connector_ref_counter_.getRefCounter(block_id), 0);
        EXPECT_EQ(pool->req_con_ref_counter_.getRefCounter(block_id), 0);
        EXPECT_EQ(pool->block_cache_ref_counter_.getRefCounter(block_id), 0);
        EXPECT_EQ(pool->req_cache_ref_counter_.getRefCounter(block_id), 0);
    }

    // The buffer and the BlockCache object identity must survive the reset (VA stability /
    // shared_ptr aliases held by kv cache groups).
    EXPECT_EQ(pool->getBaseAddress(), base_before);
    EXPECT_EQ(pool->blockCache().get(), cache_before.get());
}

TEST_F(KVCacheMetadataResetTest, ResetMetadataPoolIsFullyAllocatableAgain) {
    auto pool = makePool();

    const size_t total = pool->totalBlocksNum();
    auto         held_blocks = pool->malloc(static_cast<int>(total));
    ASSERT_FALSE(held_blocks.empty());
    ASSERT_EQ(pool->freeBlocksNum(), 0u);
    pool->requestFree(held_blocks);

    pool->resetMetadata();

    // A fresh pool can hand out every block; so must a reset pool.
    auto all_blocks = pool->malloc(static_cast<int>(total));
    EXPECT_EQ(all_blocks.size(), total);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);

    // And the lifecycle still works after reset.
    pool->requestFree(all_blocks);
    EXPECT_EQ(pool->freeBlocksNum(), total);
}

TEST_F(KVCacheMetadataResetTest, ResetMetadataOnFreshPoolIsNoOp) {
    auto pool       = makePool();
    auto fresh_pool = makePool();

    pool->resetMetadata();

    EXPECT_EQ(pool->freeBlocksNum(), fresh_pool->freeBlocksNum());
    EXPECT_EQ(pool->availableBlocksNum(), fresh_pool->availableBlocksNum());
    EXPECT_EQ(pool->free_block_ids_, fresh_pool->free_block_ids_);
}

// ==================== BlockCache::clear + generation ====================

TEST(BlockCacheClearTest, ClearDropsEntriesAndBumpsGeneration) {
    BlockCache cache;
    EXPECT_EQ(cache.generation(), 0u);

    BlockCache::CacheItem item1 = {101, 0, 1, false};
    BlockCache::CacheItem item2 = {102, 0, 2, false};
    BlockCache::CacheItem item3 = {103, 0, 3, true};  // resident entries are dropped too
    ASSERT_TRUE(cache.put(item1));
    ASSERT_TRUE(cache.put(item2));
    ASSERT_TRUE(cache.put(item3));
    ASSERT_EQ(cache.size(), 3u);
    ASSERT_EQ(cache.match(101).matched_index, 1);

    cache.clear();

    // Every pre-clear cache key now misses; generation moved forward.
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_TRUE(isNullBlockIdx(cache.match(101).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache.match(102).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache.match(103).matched_index));
    EXPECT_FALSE(cache.contains(101));
    EXPECT_EQ(cache.generation(), 1u);
}

TEST(BlockCacheClearTest, GenerationIsMonotonicAcrossClears) {
    BlockCache cache;
    for (uint64_t i = 1; i <= 5; ++i) {
        BlockCache::CacheItem item = {static_cast<CacheKeyType>(100 + i), 0, static_cast<BlockIdxType>(i), false};
        cache.put(item);
        cache.clear();
        EXPECT_EQ(cache.generation(), i);
    }
}

TEST(BlockCacheClearTest, CacheIsUsableAfterClear) {
    BlockCache cache;

    BlockCache::CacheItem old_item = {201, 0, 7, false};
    ASSERT_TRUE(cache.put(old_item));
    cache.clear();

    // New-generation entries behave normally.
    BlockCache::CacheItem new_item = {202, 0, 8, false};
    EXPECT_TRUE(cache.put(new_item));
    EXPECT_EQ(cache.match(202).matched_index, 8);
    // Old-generation key still misses until explicitly re-inserted.
    EXPECT_TRUE(isNullBlockIdx(cache.match(201).matched_index));
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.generation(), 1u);
}

}  // namespace test
}  // namespace rtp_llm
