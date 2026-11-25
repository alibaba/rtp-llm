#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

struct TestPoolConfig: public BlockPoolConfigBase {
    TestPoolConfig(std::string name, size_t blocks, BlockPoolType type = BlockPoolType::HOST) {
        pool_type            = type;
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
    size_t blockSizeBytes() const override {
        return 0;
    }

    size_t first_tree_edges{0};
    size_t last_tree_edges{0};

protected:
    void onFirstTreeRefNoLock(BlockIdxType) override {
        ++first_tree_edges;
    }

    bool onLastTreeRefNoLock(BlockIdxType) override {
        ++last_tree_edges;
        return true;
    }
};

std::shared_ptr<TestPool> makeInitializedPool(size_t physical_block_count, BlockPoolType type = BlockPoolType::HOST) {
    auto pool = std::make_shared<TestPool>(std::make_shared<TestPoolConfig>("test", physical_block_count, type));
    pool->init();
    return pool;
}

std::shared_ptr<TestPool> makeSharedPool(const std::shared_ptr<SharedBlockArena>& arena, size_t group) {
    auto config                = std::make_shared<TestPoolConfig>("shared", arena->slotCount(group));
    config->shared_arena       = arena;
    config->shared_arena_group = group;
    auto pool                  = std::make_shared<TestPool>(config);
    pool->init();
    return pool;
}

}  // namespace

TEST(IBlockPoolTest, EntirePoolCanFreezeAndResumeAllocation) {
    for (const BlockPoolType type : {BlockPoolType::DEVICE, BlockPoolType::HOST, BlockPoolType::DISK}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto       pool   = makeInitializedPool(11, type);
        const auto blocks = pool->malloc(pool->totalBlocksNum());
        ASSERT_TRUE(blocks.has_value());
        EXPECT_ANY_THROW(pool->incTreeRef(*blocks, BlockTreeRefType::FROZEN));
        pool->incTreeRef(*blocks, BlockTreeRefType::EVICT);
        auto release = pool->prepareFrozenReferenceRelease(*blocks);
        for (const auto block : *blocks) {
            pool->freezeBlock(block);
        }
        pool->decTreeRef(*blocks, BlockTreeRefType::EVICT);
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), pool->totalBlocksNum());
        EXPECT_EQ(pool->availableBlocksNum(), 0u);
        EXPECT_FALSE(pool->malloc().has_value());

        release();
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
        EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
        EXPECT_EQ(pool->availableBlocksNum(), pool->totalBlocksNum());
        EXPECT_TRUE(pool->malloc().has_value());
    }
}

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

TEST(IBlockPoolTest, SharedArenaBorrowsBytesOnlyAfterFinalActiveHold) {
    auto arena = std::make_shared<SharedBlockArena>(4096, std::vector<size_t>{512, 1024});
    auto small = makeSharedPool(arena, 0);
    auto large = makeSharedPool(arena, 1);
    EXPECT_EQ(small->totalBlocksNum(), 6u);
    EXPECT_EQ(large->totalBlocksNum(), 3u);
    const auto allocated = small->malloc(6);
    ASSERT_TRUE(allocated.has_value());
    const auto ids = *allocated;
    small->incTreeRef(ids, BlockTreeRefType::CACHE);
    EXPECT_EQ(arena->freeBytes(), 0u);
    EXPECT_EQ(arena->availableBytes(), 3072u);
    EXPECT_EQ(large->freeBlocksNum(), 0u);
    EXPECT_EQ(large->availableBlocksNum(), 3u);
    EXPECT_FALSE(large->malloc().has_value());  // CACHE metadata must be evicted first.

    for (const auto hold : {BlockTreeRefType::LOAD, BlockTreeRefType::EVICT}) {
        small->incTreeRef(ids[0], hold);
        EXPECT_EQ(arena->availableBytes(), 2560u);
        EXPECT_EQ(large->availableBlocksNum(), 2u);
        small->decTreeRef(ids[0], hold);
        EXPECT_EQ(arena->availableBytes(), 3072u);
    }

    size_t notifications = 0;
    small->setCapacityChangeCallback([&] {
        // Notification must run outside both the local pool and arena locks.
        (void)large->freeBlocksNum();
        ++notifications;
    });
    small->decTreeRef(BlockIdList{ids[0], ids[1]}, BlockTreeRefType::CACHE);
    EXPECT_GT(notifications, 0u);
    const auto borrowed = large->malloc();
    ASSERT_TRUE(borrowed.has_value());
    const size_t large_begin = static_cast<size_t>(*borrowed) * arena->stride(1);
    for (size_t i = 2; i < ids.size(); ++i) {
        const size_t small_begin = static_cast<size_t>(ids[i]) * arena->stride(0);
        EXPECT_TRUE(large_begin + arena->stride(1) <= small_begin || small_begin + arena->stride(0) <= large_begin);
    }
    large->incTreeRef(*borrowed, BlockTreeRefType::CACHE);
    large->decTreeRef(*borrowed, BlockTreeRefType::CACHE);
    small->decTreeRef(BlockIdList(ids.begin() + 2, ids.end()), BlockTreeRefType::CACHE);
    EXPECT_EQ(arena->freeBytes(), arena->usableBytes());
    EXPECT_EQ(arena->availableBytes(), arena->usableBytes());
}

TEST(IBlockPoolTest, SharedArenaJointAdmissionAndFailedAllocationRollback) {
    SharedBlockArena arena(4096, {512, 1024});
    EXPECT_EQ(arena.freeSlots(0), 6u);
    EXPECT_EQ(arena.freeSlots(1), 3u);
    EXPECT_FALSE(arena.canAllocate({4, 2}, SharedBlockArena::Capacity::TOTAL));
    EXPECT_TRUE(arena.canAllocate({2, 2}, SharedBlockArena::Capacity::TOTAL));
    EXPECT_THROW(arena.canAllocate({1}, SharedBlockArena::Capacity::FREE), std::invalid_argument);
    const auto large = arena.malloc(1, 1);
    ASSERT_EQ(large.size(), 1u);
    arena.setReclaimable(1, large[0], true);
    const auto free_before      = arena.freeBytes();
    const auto available_before = arena.availableBytes();
    EXPECT_TRUE(arena.malloc(0, 6).empty());  // Partial allocation must roll back.
    EXPECT_EQ(arena.freeBytes(), free_before);
    EXPECT_EQ(arena.availableBytes(), available_before);
    EXPECT_EQ(arena.freeSlots(0), 4u);
    arena.release(1, large[0]);
    EXPECT_EQ(arena.freeBytes(), arena.usableBytes());
    EXPECT_EQ(arena.availableBytes(), arena.usableBytes());
}

TEST(IBlockPoolTest, SharedArenaRejectsWrongOwnerAndProtectsFrozenSlots) {
    auto arena = std::make_shared<SharedBlockArena>(4096, std::vector<size_t>{512, 1024});
    auto small = makeSharedPool(arena, 0);
    auto large = makeSharedPool(arena, 1);
    auto block = small->malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_THROW(arena->release(1, *block), std::logic_error);
    EXPECT_THROW(arena->release(0, 0), std::logic_error);
    small->freezeBlock(*block);
    small->incTreeRef(*block, BlockTreeRefType::CACHE);
    small->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_TRUE(small->isFrozen(*block));
    EXPECT_TRUE(small->isAllocated(*block));
    EXPECT_EQ(arena->availableBytes(), arena->usableBytes() - 512);
    EXPECT_EQ(large->availableBlocksNum(), 2u);
}

TEST(IBlockPoolTest, MallocReturnsAllocatedTreeRefcountZeroBlocks) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 4));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.treeRefCount(*block), 0u);
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

TEST(IBlockPoolTest, TreeHooksRunOnlyOnTotalEdges) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    pool->incTreeRef(*block, BlockTreeRefType::LOAD);
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);

    EXPECT_EQ(pool->first_tree_edges, 1u);
    EXPECT_EQ(pool->last_tree_edges, 0u);

    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->first_tree_edges, 1u);
    EXPECT_EQ(pool->last_tree_edges, 0u);

    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->first_tree_edges, 1u);
    EXPECT_EQ(pool->last_tree_edges, 0u);
    EXPECT_TRUE(pool->isAllocated(*block));

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->last_tree_edges, 1u);
    EXPECT_FALSE(pool->isAllocated(*block));
}

TEST(IBlockPoolTest, TreeRefMetricsCountDistinctBlocks) {
    auto pool         = makeInitializedPool(/*physical_block_count=*/4);
    auto first_block  = pool->malloc();
    auto second_block = pool->malloc();
    ASSERT_TRUE(first_block.has_value());
    ASSERT_TRUE(second_block.has_value());

    pool->incTreeRef(*first_block, BlockTreeRefType::CACHE);
    pool->incTreeRef(*first_block, BlockTreeRefType::CACHE);
    pool->incTreeRef(*second_block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);

    pool->decTreeRef(*first_block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    pool->decTreeRef(*first_block, BlockTreeRefType::CACHE);
    pool->decTreeRef(*second_block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
}

TEST(IBlockPoolTest, ActiveBlocksDeduplicateReferenceTypes) {
    std::shared_ptr<TestPool>   pool  = makeInitializedPool(/*physical_block_count=*/4);
    std::optional<BlockIdxType> block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    size_t active_blocks = 0;
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 0u);

    pool->incTreeRef(*block, BlockTreeRefType::LOAD);
    pool->incTreeRef(*block, BlockTreeRefType::EVICT);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool->decTreeRef(*block, BlockTreeRefType::EVICT);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 0u);

    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 0u);
}

TEST(IBlockPoolTest, AvailableBlocksCountOnlyFreeAndCacheOnlyBlocks) {
    std::shared_ptr<TestPool>   pool  = makeInitializedPool(/*physical_block_count=*/4);
    std::optional<BlockIdxType> block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    const size_t total_blocks = pool->totalBlocksNum();
    ASSERT_EQ(total_blocks, 3u);
    EXPECT_EQ(pool->freeBlocksNum(), total_blocks - 1);
    // A freshly allocated block has no owner yet, but is not physically free or
    // cache-only and therefore must not be reported as available.
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);

    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks);

    // Duplicate CACHE refs do not double count a block.
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks);

    pool->incTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);
    pool->incTreeRef(*block, BlockTreeRefType::EVICT);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);
    pool->decTreeRef(*block, BlockTreeRefType::EVICT);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks);

    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(pool->isAllocated(*block));
    EXPECT_EQ(pool->freeBlocksNum(), total_blocks);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks);
}

TEST(IBlockPoolTest, AscendingOrderReturnsSortedBlockIds) {
    auto pool = TestPool(std::make_shared<TestPoolConfig>("test", 6));
    ASSERT_TRUE(pool.init());

    auto blocks = pool.malloc(3);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(*blocks, (BlockIdList{1, 2, 3}));
    pool.incTreeRef(BlockIdList{2}, BlockTreeRefType::EVICT);
    pool.decTreeRef(BlockIdList{2}, BlockTreeRefType::EVICT);

    auto more = pool.malloc(2);
    ASSERT_TRUE(more.has_value());
    EXPECT_EQ(*more, (BlockIdList{4, 5}));

    pool.incTreeRef(BlockIdList{1, 3}, BlockTreeRefType::EVICT);
    pool.decTreeRef(BlockIdList{1, 3}, BlockTreeRefType::EVICT);
    auto afterMerge = pool.malloc(2);
    ASSERT_TRUE(afterMerge.has_value());
    EXPECT_EQ(*afterMerge, (BlockIdList{1, 2}));
}

TEST(IBlockPoolTest, MultipleTreeRefsReleaseOnlyAtLastHolder) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());

    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    pool->incTreeRef(*block, BlockTreeRefType::LOAD);

    EXPECT_TRUE(pool->isAllocated(*block));
    EXPECT_EQ(pool->treeRefCount(*block), 2u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_TRUE(pool->isAllocated(*block));
    EXPECT_EQ(pool->treeRefCount(*block), 1u);

    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(pool->isAllocated(*block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
}

TEST(IBlockPoolTest, FrozenHoldsArePerExecutionAndReleaseOnlyTheirOwnReferences) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    const auto first  = blocks->at(0);
    const auto second = blocks->at(1);
    pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
    pool->incTreeRef(first, BlockTreeRefType::LOAD);
    pool->incTreeRef(second, BlockTreeRefType::EVICT);
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] { ++notifications; });

    std::vector<std::function<void()>> releases;
    for (size_t i = 0; i < 8; ++i) {
        releases.push_back(pool->prepareFrozenReferenceRelease({first}));
    }
    auto                     release_second = pool->prepareFrozenReferenceRelease({second});
    std::vector<std::thread> threads;
    for (size_t i = 0; i < releases.size(); ++i) {
        threads.emplace_back([&] { pool->freezeBlock(first); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    pool->freezeBlock(second);
    EXPECT_EQ(pool->treeRefCount(first), 10u);
    EXPECT_EQ(pool->treeRefCount(second), 3u);
    // Stock metrics count physical blocks, independently of the number of executions.
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    pool->decTreeRef(first, BlockTreeRefType::LOAD);
    pool->decTreeRef(second, BlockTreeRefType::EVICT);
    pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(first), 8u);
    EXPECT_EQ(pool->treeRefCount(second), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICT), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    EXPECT_EQ(pool->activeBlocksNum(), 2u);
    EXPECT_EQ(pool->availableBlocksNum(), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(notifications, 0u);
    for (size_t i = 0; i + 1 < releases.size(); ++i) {
        releases[i]();
        EXPECT_TRUE(pool->isFrozen(first));
        EXPECT_TRUE(pool->isAllocated(first));
    }
    EXPECT_EQ(pool->treeRefCount(first), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    releases.back()();
    EXPECT_FALSE(pool->isAllocated(first));
    EXPECT_TRUE(pool->isFrozen(second));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_EQ(notifications, 1u);
    release_second();
    EXPECT_FALSE(pool->isAllocated(second));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
    EXPECT_EQ(notifications, 2u);
    EXPECT_EQ(pool->first_tree_edges, 2u);
    EXPECT_EQ(pool->last_tree_edges, 2u);
}

TEST(IBlockPoolTest, DecTreeRefRejectsWrongInternalType) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);

    EXPECT_ANY_THROW(pool->decTreeRef(*block, BlockTreeRefType::LOAD));
    EXPECT_ANY_THROW(pool->incTreeRef(*block, BlockTreeRefType::FROZEN));
    EXPECT_ANY_THROW(pool->incTreeRef(BlockIdList{*block}, BlockTreeRefType::FROZEN));
    EXPECT_EQ(pool->treeRefCount(*block), 1u);
    pool->freezeBlock(*block);
    EXPECT_ANY_THROW(pool->decTreeRef(*block, BlockTreeRefType::FROZEN));
    EXPECT_ANY_THROW(pool->decTreeRef(BlockIdList{*block}, BlockTreeRefType::FROZEN));

    EXPECT_TRUE(pool->isAllocated(*block));
    EXPECT_EQ(pool->treeRefCount(*block), 2u);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(*block), 1u);
    EXPECT_TRUE(pool->isFrozen(*block));
}

TEST(IBlockPoolTest, BatchIncTreeRefRejectsInvalidTailWithoutMutatingPrefix) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    const BlockIdxType unallocated_block = *block + 1;

    EXPECT_ANY_THROW(pool->incTreeRef(BlockIdList{*block, unallocated_block}, BlockTreeRefType::CACHE));

    EXPECT_EQ(pool->treeRefCount(*block), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    if (pool->treeRefCount(*block) == 0) {
        pool->incTreeRef(*block, BlockTreeRefType::EVICT);
        pool->decTreeRef(*block, BlockTreeRefType::EVICT);
    } else {
        pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    }
}

TEST(IBlockPoolTest, BatchDecTreeRefRejectsInvalidTailWithoutMutatingPrefix) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    pool->incTreeRef(blocks->back(), BlockTreeRefType::LOAD);

    EXPECT_ANY_THROW(pool->decTreeRef(*blocks, BlockTreeRefType::CACHE));

    EXPECT_TRUE(pool->isAllocated(blocks->front()));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
    if (pool->isAllocated(blocks->front())) {
        EXPECT_EQ(pool->treeRefCount(blocks->front()), 1u);
        pool->decTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    }
    pool->decTreeRef(blocks->back(), BlockTreeRefType::LOAD);
}

TEST(IBlockPoolTest, CapacityNotificationsOnlyPublishReleasedCapacityOutsidePoolLock) {
    auto   pool          = makeInitializedPool(4);
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] {
        ++notifications;
        // Re-entering a getter verifies the observer is not called under the pool mutex.
        EXPECT_GT(pool->availableBlocksNum(), 0u);
    });
    const auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_EQ(notifications, 0u);
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(notifications, 1u);
    pool->incTreeRef(*block, BlockTreeRefType::LOAD);
    pool->incTreeRef(*block, BlockTreeRefType::EVICT);
    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(notifications, 1u);
    pool->decTreeRef(*block, BlockTreeRefType::EVICT);
    EXPECT_EQ(notifications, 2u);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(notifications, 3u);
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

}  // namespace rtp_llm
