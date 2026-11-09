#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <limits>
#include <memory>
#include <optional>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>

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

std::shared_ptr<TestPool> makeInitializedPool(size_t                physical_block_count,
                                            BlockPoolType         pool_type   = BlockPoolType::HOST,
                                            std::optional<double> abort_ratio = std::nullopt) {
    auto config = std::make_shared<TestPoolConfig>("test", physical_block_count, pool_type);
    if (abort_ratio) {
        config->frozen_block_abort_ratio = *abort_ratio;
    }
    auto pool = std::make_shared<TestPool>(std::move(config));
    pool->init();
    return pool;
}

// Ownership tests intentionally freeze most of their small pools.
std::shared_ptr<TestPool> makeUnprotectedPool(size_t physical_block_count) {
    return makeInitializedPool(physical_block_count, BlockPoolType::HOST, /*abort_ratio=*/0.0);
}

void disableCoreDumpInChild() {
    const rlimit limit{0, 0};
    if (setrlimit(RLIMIT_CORE, &limit) != 0) {
        _exit(1);
    }
}

}  // namespace

class IBlockPoolFrozenAbortRatioTest: public ::testing::TestWithParam<BlockPoolType> {};

TEST_P(IBlockPoolFrozenAbortRatioTest, DefaultJitteredRatioAllowsThirtyPercentAndAbortsByFiftyPercent) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam());
    auto blocks = pool->malloc(5);
    ASSERT_TRUE(blocks.has_value());
    for (size_t index = 0; index < 3; ++index) {
        pool->freezeBlock(blocks->at(index));
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 3u);
    EXPECT_EXIT(
        {
            disableCoreDumpInChild();
            pool->freezeBlock(blocks->at(3));
            pool->freezeBlock(blocks->at(4));
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
}

TEST_P(IBlockPoolFrozenAbortRatioTest, JitteredRatioCountsDistinctBlocksAndAllowsConfiguredBoundary) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), /*abort_ratio=*/0.5);
    auto blocks = pool->malloc(7);
    ASSERT_TRUE(blocks.has_value());
    for (size_t index = 0; index < 5; ++index) {
        pool->freezeBlock(blocks->at(index));
        pool->freezeBlock(blocks->at(index));
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 5u);
    EXPECT_EXIT(
        {
            disableCoreDumpInChild();
            pool->freezeBlock(blocks->at(5));
            pool->freezeBlock(blocks->at(6));
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
    EXPECT_ANY_THROW(pool->incTreeRef(blocks->at(5), BlockTreeRefType::FROZEN));
    EXPECT_ANY_THROW(pool->incTreeRef(blocks->at(6), BlockTreeRefType::FROZEN));

}

TEST_P(IBlockPoolFrozenAbortRatioTest, BatchFrozenReferencesCannotBypassGuard) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), /*abort_ratio=*/0.5);
    auto blocks = pool->malloc(7);
    ASSERT_TRUE(blocks.has_value());
    EXPECT_ANY_THROW(pool->incTreeRef(*blocks, BlockTreeRefType::FROZEN));

}

TEST_P(IBlockPoolFrozenAbortRatioTest, ZeroAndOneAllowEntirePoolToFreeze) {
    for (const double ratio : {0.0, 1.0}) {
        SCOPED_TRACE(ratio);
        auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), ratio);
        EXPECT_EQ(pool->effective_frozen_block_abort_ratio_, ratio);
        auto blocks = pool->malloc(10);
        ASSERT_TRUE(blocks.has_value());
        for (const auto block : *blocks) {
            pool->freezeBlock(block);
        }
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), pool->totalBlocksNum());
    }
}

TEST(IBlockPoolTest, FrozenAbortJitterIsStableAcrossPoolsAndPreservesConfiguredRatio) {
    for (const double ratio : {0.1, 0.3, 0.5, 0.98, std::nextafter(1.0, 0.0)}) {
        SCOPED_TRACE(ratio);
        auto reference = makeInitializedPool(11, BlockPoolType::HOST, ratio);
        const double effective = reference->effective_frozen_block_abort_ratio_;
        EXPECT_GE(effective, ratio);
        EXPECT_LE(effective, ratio + 0.10);
        EXPECT_LT(effective, 1.0);
        for (const auto type : {BlockPoolType::DEVICE, BlockPoolType::HOST, BlockPoolType::DISK}) {
            SCOPED_TRACE(static_cast<int>(type));
            for (int repetition = 0; repetition < 3; ++repetition) {
                auto config = std::make_shared<TestPoolConfig>("jitter", 11, type);
                config->frozen_block_abort_ratio = ratio;
                auto pool = std::make_shared<TestPool>(config);
                ASSERT_TRUE(pool->init());
                EXPECT_EQ(config->frozen_block_abort_ratio, ratio);
                EXPECT_EQ(pool->effective_frozen_block_abort_ratio_, effective);
                const auto block = pool->malloc();
                ASSERT_TRUE(block.has_value());
                pool->freezeBlock(*block);
                EXPECT_EQ(pool->effective_frozen_block_abort_ratio_, effective);
            }
        }
    }
}

TEST_P(IBlockPoolFrozenAbortRatioTest, RuntimeJitteredThresholdControlsFreezeApi) {
    constexpr size_t usable_blocks = 1000;
    auto pool = makeInitializedPool(usable_blocks + 1, GetParam(), 0.3);
    const double effective = pool->effective_frozen_block_abort_ratio_;
    size_t allowed = static_cast<size_t>(std::floor(effective * usable_blocks));
    if (static_cast<double>(allowed) / usable_blocks > effective) {
        --allowed;
    }
    if (static_cast<double>(allowed + 1) / usable_blocks <= effective) {
        ++allowed;
    }
    ASSERT_GE(allowed, 300u);
    ASSERT_LE(allowed, 400u);
    ASSERT_LE(static_cast<double>(allowed) / usable_blocks, effective);
    ASSERT_GT(static_cast<double>(allowed + 1) / usable_blocks, effective);
    auto blocks = pool->malloc(allowed + 1);
    ASSERT_TRUE(blocks.has_value());
    for (size_t index = 0; index < allowed; ++index) {
        pool->freezeBlock(blocks->at(index));
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), allowed);
    EXPECT_EQ(pool->effective_frozen_block_abort_ratio_, effective);
    EXPECT_EXIT(
        {
            disableCoreDumpInChild();
            pool->freezeBlock(blocks->back());
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
    EXPECT_ANY_THROW(pool->incTreeRef(blocks->back(), BlockTreeRefType::FROZEN));

}

TEST_P(IBlockPoolFrozenAbortRatioTest, NearOneJitterStillAbortsWhenEntirePoolFreezes) {
    for (const double ratio : {0.99, std::nextafter(1.0, 0.0)}) {
        SCOPED_TRACE(ratio);
        auto pool = makeInitializedPool(101, GetParam(), ratio);
        EXPECT_GE(pool->effective_frozen_block_abort_ratio_, ratio);
        EXPECT_LT(pool->effective_frozen_block_abort_ratio_, 1.0);
        auto blocks = pool->malloc(100);
        ASSERT_TRUE(blocks.has_value());
        for (size_t index = 0; index < 99; ++index) {
            pool->freezeBlock(blocks->at(index));
        }
        EXPECT_EXIT(
            {
                disableCoreDumpInChild();
                pool->freezeBlock(blocks->back());
            },
            ::testing::KilledBySignal(SIGABRT),
            "");
    }
}

TEST_P(IBlockPoolFrozenAbortRatioTest, TinyPositiveRatioRemainsEnabled) {
    const double ratio = std::numeric_limits<double>::denorm_min();
    auto pool = makeInitializedPool(6, GetParam(), ratio);
    EXPECT_GE(pool->effective_frozen_block_abort_ratio_, ratio);
    EXPECT_LE(pool->effective_frozen_block_abort_ratio_, ratio + 0.10);
    const auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_EXIT(
        {
            disableCoreDumpInChild();
            pool->freezeBlock(*block);
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
}

TEST_P(IBlockPoolFrozenAbortRatioTest, RejectsInvalidRatios) {

    for (const double ratio : {-0.1,
                               1.1,
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()}) {
        SCOPED_TRACE(ratio);
        EXPECT_THROW(makeInitializedPool(11, GetParam(), ratio), RTPException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllPoolTypes,
                         IBlockPoolFrozenAbortRatioTest,
                         ::testing::Values(BlockPoolType::DEVICE, BlockPoolType::HOST, BlockPoolType::DISK));

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
    pool->incTreeRef(*block, BlockTreeRefType::STORE);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    active_blocks = pool->activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool->decTreeRef(*block, BlockTreeRefType::STORE);
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
    pool->incTreeRef(*block, BlockTreeRefType::STORE);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);

    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool->availableBlocksNum(), total_blocks - 1);
    pool->decTreeRef(*block, BlockTreeRefType::STORE);
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
    pool.incTreeRef(BlockIdList{2}, BlockTreeRefType::STORE);
    pool.decTreeRef(BlockIdList{2}, BlockTreeRefType::STORE);

    auto more = pool.malloc(2);
    ASSERT_TRUE(more.has_value());
    EXPECT_EQ(*more, (BlockIdList{4, 5}));

    pool.incTreeRef(BlockIdList{1, 3}, BlockTreeRefType::STORE);
    pool.decTreeRef(BlockIdList{1, 3}, BlockTreeRefType::STORE);
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

TEST(IBlockPoolTest, FreezeDeduplicatesOperationsAndSurvivesOrdinaryReferenceRelease) {
    auto pool   = makeUnprotectedPool(/*physical_block_count=*/4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    const auto load_block     = blocks->at(0);
    const auto eviction_block = blocks->at(1);
    pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
    pool->incTreeRef(eviction_block, BlockTreeRefType::EVICTION);
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] { ++notifications; });

    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_FALSE(pool->isFrozen(load_block));
    // Freezing needs no operation reference and concurrent calls share one hold.
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&] { pool->freezeBlock(load_block); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    pool->incTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->freezeBlock(load_block);
    pool->freezeBlock(load_block);
    pool->freezeBlock(eviction_block);
    pool->freezeBlock(eviction_block);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    EXPECT_TRUE(pool->isFrozen(load_block));
    EXPECT_TRUE(pool->isFrozen(eviction_block));
    EXPECT_EQ(pool->treeRefCount(load_block), 3u);
    EXPECT_EQ(pool->treeRefCount(eviction_block), 3u);

    // A successful peer Task releases only its own ordinary reference.
    pool->incTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(eviction_block, BlockTreeRefType::EVICTION);
    pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(load_block), 1u);
    EXPECT_EQ(pool->treeRefCount(eviction_block), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    EXPECT_EQ(pool->activeBlocksNum(), 2u);
    EXPECT_EQ(pool->availableBlocksNum(), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(notifications, 0u);
    EXPECT_EQ(pool->first_tree_edges, 2u);
    EXPECT_EQ(pool->last_tree_edges, 0u);
    EXPECT_EQ(pool->malloc(), std::optional<BlockIdxType>{3});
    EXPECT_FALSE(pool->malloc().has_value());

    // The same block ID in another pool has its own FROZEN reference, even without other holders.
    auto other_pool = makeUnprotectedPool(3);
    ASSERT_EQ(other_pool->malloc(), std::optional<BlockIdxType>{load_block});
    other_pool->freezeBlock(load_block);
    EXPECT_EQ(other_pool->treeRefCount(load_block), 1u);
    EXPECT_EQ(other_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_EQ(other_pool->activeBlocksNum(), 1u);
    EXPECT_EQ(other_pool->availableBlocksNum(), 1u);
    EXPECT_NE(other_pool->debugString().find("frozen_ref_blocks=1"), std::string::npos);
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
        pool->incTreeRef(*block, BlockTreeRefType::STORE);
        pool->decTreeRef(*block, BlockTreeRefType::STORE);
    } else {
        pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    }
}

TEST(IBlockPoolTest, BatchDecTreeRefRejectsInvalidTailWithoutMutatingPrefix) {
    auto pool   = makeUnprotectedPool(/*physical_block_count=*/4);
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
    pool->incTreeRef(*block, BlockTreeRefType::STORE);
    pool->decTreeRef(*block, BlockTreeRefType::LOAD);
    EXPECT_EQ(notifications, 1u);
    pool->decTreeRef(*block, BlockTreeRefType::STORE);
    EXPECT_EQ(notifications, 2u);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(notifications, 3u);
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

}  // namespace rtp_llm
