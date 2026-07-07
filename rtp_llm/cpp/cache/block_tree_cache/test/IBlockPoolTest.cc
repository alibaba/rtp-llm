#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace rtp_llm::block_tree_cache {
namespace {

struct TestPoolConfig: public BlockPoolConfigBase {
    TestPoolConfig(std::string name, size_t blocks, FreeBlockOrderPolicy policy) {
        pool_type               = BlockPoolType::HOST;
        pool_name               = std::move(name);
        physical_block_count    = blocks;
        free_block_order_policy = policy;
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

}  // namespace

TEST(IBlockPoolTest, BlockZeroIsInvalidAndNeverAllocated) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4, FreeBlockOrderPolicy::ANY_ORDER));
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
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4, FreeBlockOrderPolicy::ANY_ORDER));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);
    EXPECT_EQ(pool.unreferencedBlocksNum(), 1u);
}

TEST(IBlockPoolTest, BatchMallocIsAtomic) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4, FreeBlockOrderPolicy::ANY_ORDER));
    ASSERT_TRUE(pool.init());

    auto first = pool.malloc(2);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(pool.freeBlocksNum(), 1u);

    auto second = pool.malloc(2);
    EXPECT_FALSE(second.has_value());
    EXPECT_EQ(pool.freeBlocksNum(), 1u);
}

TEST(IBlockPoolTest, RefcountMetricsFollowSingleRefcountModel) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4, FreeBlockOrderPolicy::ANY_ORDER));
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
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 6, FreeBlockOrderPolicy::ASCENDING_ORDER));
    ASSERT_TRUE(pool.init());

    auto blocks = pool.malloc(3);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(*blocks, (BlockIds{1, 2, 3}));
    pool.free(BlockIds{2});

    auto more = pool.malloc(2);
    ASSERT_TRUE(more.has_value());
    EXPECT_EQ(*more, (BlockIds{4, 5}));

    pool.free(BlockIds{1, 3});
    auto afterMerge = pool.malloc(2);
    ASSERT_TRUE(afterMerge.has_value());
    EXPECT_EQ(*afterMerge, (BlockIds{1, 2}));
}

}  // namespace rtp_llm::block_tree_cache
