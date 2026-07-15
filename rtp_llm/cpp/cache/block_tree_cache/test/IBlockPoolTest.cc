#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

#include <algorithm>
#include <memory>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

struct TestPoolConfig: public BlockPoolConfigBase {
    TestPoolConfig(std::string name, size_t blocks) {
        pool_type            = BlockPoolType::HOST;
        pool_name            = std::move(name);
        physical_block_count = blocks;
    }
};

class TestPool: public IBlockPool {
public:
    explicit TestPool(std::shared_ptr<const TestPoolConfig> config): IBlockPool(std::move(config)) {}
    bool init() {
        markInitialized();
        return true;
    }
};

std::shared_ptr<TestPool> makeInitializedPool(size_t physical_block_count) {
    auto pool = std::make_shared<TestPool>(std::make_shared<TestPoolConfig>("test", physical_block_count));
    pool->init();
    return pool;
}

}  // namespace

TEST(IBlockPoolTest, BlockZeroIsInvalidAndNeverAllocated) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4));
    ASSERT_TRUE(pool.init());

    EXPECT_FALSE(pool.validBlock(NULL_BLOCK_IDX));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_EQ(pool.totalBlocksNum(), 3u);

    auto blocks = pool.malloc(3);
    ASSERT_TRUE(blocks.has_value());
    EXPECT_EQ(blocks->size(), 3u);
    EXPECT_EQ(std::find(blocks->begin(), blocks->end(), 0), blocks->end());
}

TEST(IBlockPoolTest, MallocReturnsAllocatedRefcountZeroBlocks) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);
    EXPECT_EQ(pool.unreferencedBlocksNum(), 1u);
}

TEST(IBlockPoolTest, BatchMallocIsAtomic) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4));
    ASSERT_TRUE(pool.init());

    auto first = pool.malloc(2);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(pool.freeBlocksNum(), 1u);

    auto second = pool.malloc(2);
    EXPECT_FALSE(second.has_value());
    EXPECT_EQ(pool.freeBlocksNum(), 1u);
}

TEST(IBlockPoolTest, RefcountMetricsFollowSingleRefcountModel) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_EQ(pool.unreferencedBlocksNum(), 1u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    EXPECT_EQ(pool.treeCachedBlocksNum(), 1u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 2u);
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 1u);

    pool.decRef(*block);
    pool.free(*block);
    EXPECT_EQ(pool.freeBlocksNum(), 3u);
}

TEST(IBlockPoolTest, AscendingOrderReturnsSortedBlockIds) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 6));
    ASSERT_TRUE(pool.init());

    auto blocks = pool.malloc(3);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(*blocks, (BlockIdList{1, 2, 3}));
    pool.free(BlockIdList{2});

    auto more = pool.malloc(2);
    ASSERT_TRUE(more.has_value());
    EXPECT_EQ(*more, (BlockIdList{4, 5}));

    pool.free(BlockIdList{1, 3});
    auto afterMerge = pool.malloc(2);
    ASSERT_TRUE(afterMerge.has_value());
    EXPECT_EQ(*afterMerge, (BlockIdList{1, 2}));
}

TEST(IBlockPoolTest, DecRefDoesNotFreeWhileAnotherHolderExists) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    // malloc() only reserves capacity; owners must explicitly take refs.
    pool->incRef(*block);  // cache holder
    pool->incRef(*block);  // request holder
    EXPECT_EQ(pool->refCount(*block), 2u);

    pool->decRef(*block);
    EXPECT_TRUE(pool->isAllocated(*block));
    EXPECT_EQ(pool->refCount(*block), 1u);

    pool->decRef(*block);
    EXPECT_FALSE(pool->isAllocated(*block));
}

TEST(IBlockPoolTest, DecRefFreesSingleRequestHolder) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    pool->incRef(*block);  // request holder
    pool->decRef(*block);

    EXPECT_FALSE(pool->isAllocated(*block));
}

TEST(IBlockPoolTest, DecRefRejectsUnheldAllocatedBlock) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_EQ(pool->refCount(*block), 0u);

    // RTP_LLM_CHECK aborts unless core-dump-on-exception is disabled; flip it so the
    // guard is observable as a throw in this test env.
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool->decRef(*block));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_TRUE(pool->isAllocated(*block));
}

}  // namespace rtp_llm
