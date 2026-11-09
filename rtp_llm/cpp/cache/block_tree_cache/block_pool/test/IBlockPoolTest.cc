#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <csignal>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

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

// Ownership and observer tests exercise frozen holds independently of the
// production capacity guard, including pools with just two usable blocks.
std::shared_ptr<TestPool> makeUnprotectedPool(size_t physical_block_count) {
    return makeInitializedPool(physical_block_count, BlockPoolType::HOST, /*abort_ratio=*/0.0);
}

}  // namespace

TEST(IBlockPoolTest, UnfreezeReleasesOnlyFrozenReference) {
    for (auto remaining : {BlockTreeRefType::CACHE,
                           BlockTreeRefType::LOAD,
                           BlockTreeRefType::EVICTION,
                           BlockTreeRefType::STORE,
                           BlockTreeRefType::FROZEN}) {
        SCOPED_TRACE(static_cast<size_t>(remaining));
        auto pool = makeUnprotectedPool(4);
        const auto block = pool->malloc().value();
        const bool has_remaining = remaining != BlockTreeRefType::FROZEN;
        const bool cache_only = remaining == BlockTreeRefType::CACHE;
        if (has_remaining) pool->incTreeRef(block, remaining);
        pool->freezeBlock(block);
        size_t notifications = 0;
        pool->setCapacityChangeCallback([&] {
            ++notifications;
            EXPECT_GT(pool->availableBlocksNum(), 2u);  // Also verifies callback runs outside the pool lock.
        });
        EXPECT_EQ(pool->availableBlocksNum(), 2u);
        pool->unfreezeBlock(block);
        if (has_remaining) {
            EXPECT_FALSE(pool->isFrozen(block));
            EXPECT_EQ(pool->treeRefCount(block), 1u);
            EXPECT_EQ(pool->referencedBlocksNum(remaining), 1u);
            EXPECT_EQ(pool->last_tree_edges, 0u);
        } else {
            EXPECT_EQ(pool->last_tree_edges, 1u);
        }
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
        EXPECT_EQ(pool->isAllocated(block), has_remaining);
        EXPECT_EQ(pool->freeBlocksNum(), has_remaining ? 2u : 3u);
        EXPECT_EQ(pool->availableBlocksNum(), has_remaining && !cache_only ? 2u : 3u);
        EXPECT_EQ(pool->activeBlocksNum(), has_remaining && !cache_only ? 1u : 0u);
        EXPECT_EQ(notifications, !has_remaining || cache_only ? 1u : 0u);
        if (has_remaining) pool->decTreeRef(block, remaining);
        EXPECT_EQ(pool->freeBlocksNum(), 3u);
        EXPECT_EQ(pool->availableBlocksNum(), 3u);
        EXPECT_EQ(pool->activeBlocksNum(), 0u);
        EXPECT_EQ(pool->first_tree_edges, 1u);
        EXPECT_EQ(pool->last_tree_edges, 1u);
        EXPECT_EQ(notifications, cache_only ? 2u : 1u);
    }
}

class IBlockPoolFrozenAbortRatioTest: public ::testing::TestWithParam<BlockPoolType> {};

TEST_P(IBlockPoolFrozenAbortRatioTest, DefaultFreezeAbortRatioAllowsThirtyPercentAndRejectsMore) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam());
    auto blocks = pool->malloc(4);
    ASSERT_TRUE(blocks.has_value());

    for (size_t index = 0; index < 3; ++index) {
        pool->freezeBlock(blocks->at(index));
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 3u);

    EXPECT_EXIT(pool->freezeBlock(blocks->at(3)), ::testing::KilledBySignal(SIGABRT), "");
}

TEST_P(IBlockPoolFrozenAbortRatioTest, ConfiguredFreezeAbortRatioAllowsEqualityAndRejectsMore) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), /*abort_ratio=*/0.5);
    auto blocks = pool->malloc(6);
    ASSERT_TRUE(blocks.has_value());

    for (size_t index = 0; index < 5; ++index) {
        pool->freezeBlock(blocks->at(index));
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 5u);

    EXPECT_EXIT(pool->freezeBlock(blocks->at(5)), ::testing::KilledBySignal(SIGABRT), "");
}

TEST_P(IBlockPoolFrozenAbortRatioTest, ZeroFreezeAbortRatioDisablesGuard) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), /*abort_ratio=*/0.0);
    auto blocks = pool->malloc(10);
    ASSERT_TRUE(blocks.has_value());

    for (const auto block : *blocks) {
        pool->freezeBlock(block);
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), pool->totalBlocksNum());
}

TEST_P(IBlockPoolFrozenAbortRatioTest, FullFreezeAbortRatioAllowsEntirePool) {
    auto pool   = makeInitializedPool(/*physical_block_count=*/11, GetParam(), /*abort_ratio=*/1.0);
    auto blocks = pool->malloc(10);
    ASSERT_TRUE(blocks.has_value());

    for (const auto block : *blocks) {
        pool->freezeBlock(block);
    }
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), pool->totalBlocksNum());
}

TEST_P(IBlockPoolFrozenAbortRatioTest, RejectsInvalidFrozenBlockAbortRatios) {
    const bool previous = StaticConfig::user_ft_core_dump_on_exception;
    block_tree_cache_detail::ScopeRollback restore_config(
        [previous] { StaticConfig::user_ft_core_dump_on_exception = previous; });
    StaticConfig::user_ft_core_dump_on_exception = false;

    for (const auto ratio :
         {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
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

TEST(IBlockPoolTest, OriginTrackingIsOptionalAndFollowsAllocatedPhysicalLifetime) {
    auto       pool   = makeInitializedPool(2);
    const auto block  = *pool->malloc();
    const auto parsed = DecodeCacheOrigin::parse("v1." + std::string(32, 'a') + ".7680.7680.8192.8200.8201");
    ASSERT_TRUE(parsed);
    auto source = std::make_shared<const DecodeCacheOrigin>(*parsed);
    EXPECT_EQ(pool->decodeOriginIncarnation(block), uint64_t{0});
    EXPECT_FALSE(pool->bindDecodeOrigin(block, 1, "full", source));
    ASSERT_TRUE(pool->enableDecodeOriginTracking());
    const auto stamp = pool->decodeOriginIncarnation(block);
    ASSERT_NE(stamp, uint64_t{0});
    ASSERT_TRUE(pool->bindDecodeOrigin(block, stamp, "full", source));
    pool->incTreeRef(block, BlockTreeRefType::CACHE);
    pool->incTreeRef(block, BlockTreeRefType::CACHE);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->decodeOrigin(block, "full"), source);
    ASSERT_TRUE(pool->enableDecodeOriginTracking());  // idempotent, not a new incarnation
    EXPECT_EQ(pool->decodeOriginIncarnation(block), stamp);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(pool->decodeOrigin(block, "full"));
    EXPECT_EQ(pool->decodeOriginIncarnation(block), uint64_t{0});
    ASSERT_EQ(*pool->malloc(), block);
    EXPECT_NE(pool->decodeOriginIncarnation(block), stamp);
    EXPECT_FALSE(pool->bindDecodeOrigin(block, stamp, "full", source));
    EXPECT_EQ(pool->decodeOriginIncarnation(NULL_BLOCK_IDX), uint64_t{0});
    EXPECT_FALSE(pool->bindDecodeOrigin(NULL_BLOCK_IDX, stamp, "full", source));
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

TEST(IBlockPoolTest, FrozenReferencesCountHoldsAndTrackDistinctBlocks) {
    auto pool   = makeUnprotectedPool(/*physical_block_count=*/4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    const auto load_block     = blocks->at(0);
    const auto eviction_block = blocks->at(1);
    pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
    pool->incTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->incTreeRef(eviction_block, BlockTreeRefType::EVICTION);
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] { ++notifications; });

    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_FALSE(pool->isFrozen(load_block));
    // Two tasks freeze the load block independently; each holds one reference.
    pool->freezeBlock(load_block);
    pool->freezeBlock(load_block);
    pool->incTreeRef(*blocks, BlockTreeRefType::FROZEN);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 2u);
    EXPECT_TRUE(pool->isFrozen(load_block));
    EXPECT_TRUE(pool->isFrozen(eviction_block));
    EXPECT_EQ(pool->treeRefCount(load_block), 5u);
    EXPECT_EQ(pool->treeRefCount(eviction_block), 3u);

    // Ordinary reference releases must not remove the frozen hold.
    pool->incTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(load_block, BlockTreeRefType::LOAD);
    pool->decTreeRef(eviction_block, BlockTreeRefType::EVICTION);
    pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(load_block), 3u);
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
    pool->incTreeRef(3, BlockTreeRefType::STORE);
    pool->decTreeRef(3, BlockTreeRefType::STORE);

    pool->decTreeRef(*blocks, BlockTreeRefType::FROZEN);
    EXPECT_EQ(pool->treeRefCount(load_block), 2u);
    EXPECT_TRUE(pool->isFrozen(load_block));
    EXPECT_FALSE(pool->isFrozen(eviction_block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), 2u);
    // Releasing one holder keeps the block frozen until the last hold goes away.
    pool->decTreeRef(load_block, BlockTreeRefType::FROZEN);
    EXPECT_TRUE(pool->isFrozen(load_block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    pool->decTreeRef(load_block, BlockTreeRefType::FROZEN);
    EXPECT_FALSE(pool->isFrozen(load_block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(pool->activeBlocksNum(), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
    EXPECT_EQ(pool->availableBlocksNum(), pool->totalBlocksNum());

    // The same block ID in a separate pool has independent ownership.
    auto other_pool = makeUnprotectedPool(3);
    ASSERT_EQ(other_pool->malloc(), std::optional<BlockIdxType>{load_block});
    other_pool->freezeBlock(load_block);
    EXPECT_EQ(other_pool->treeRefCount(load_block), 1u);
    EXPECT_EQ(other_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_EQ(other_pool->activeBlocksNum(), 1u);
    EXPECT_EQ(other_pool->availableBlocksNum(), 1u);
    EXPECT_NE(other_pool->debugString().find("frozen_ref_blocks=1"), std::string::npos);
    other_pool->decTreeRef(load_block, BlockTreeRefType::FROZEN);
    EXPECT_EQ(other_pool->freeBlocksNum(), other_pool->totalBlocksNum());
}

TEST(IBlockPoolTest, FrozenObserverSeesEveryMutationOutsidePoolLock) {
    auto pool   = makeUnprotectedPool(4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
    const BlockIdxType first  = blocks->front();
    const BlockIdxType second = blocks->back();

    std::mutex                                 cache_mutex;
    std::vector<std::pair<BlockIdxType, bool>> changes;
    auto                                       observer = std::make_shared<FrozenBlockObserver>();
    observer->mutex                                     = &cache_mutex;
    observer->on_changed                                = [&](IBlockPool* changed_pool, BlockIdxType block) {
        EXPECT_EQ(changed_pool, pool.get());
        // Getters take the pool mutex; the observer must run after releasing it.
        changes.emplace_back(block, changed_pool->isFrozen(block));
        EXPECT_GE(changed_pool->treeRefCount(block), 1u);
    };
    pool->setFrozenBlockObserver(observer);
    size_t capacity_notifications = 0;
    pool->setCapacityChangeCallback([&] {
        ++capacity_notifications;
        EXPECT_GT(pool->availableBlocksNum(), 0u);
    });

    pool->freezeBlock(first);
    pool->incTreeRef(*blocks, BlockTreeRefType::FROZEN);
    pool->decTreeRef(*blocks, BlockTreeRefType::FROZEN);
    EXPECT_TRUE(pool->isFrozen(first));
    EXPECT_FALSE(pool->isFrozen(second));
    pool->unfreezeBlock(first);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        pool->freezeBlockLocked(first);
        pool->unfreezeBlockLocked(first);
    }

    EXPECT_EQ(changes,
              (std::vector<std::pair<BlockIdxType, bool>>{{first, true},
                                                          {first, true},
                                                          {second, true},
                                                          {first, true},
                                                          {second, false},
                                                          {first, false},
                                                          {first, true},
                                                          {first, false}}));
    EXPECT_EQ(capacity_notifications, 3u);
    pool->clearFrozenBlockObserver(observer);
    pool->setCapacityChangeCallback({});
    pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
}

TEST(IBlockPoolTest, FrozenObserverRunsBeforeThrowingCapacityCallback) {
    for (const bool batch : {false, true}) {
        SCOPED_TRACE(batch);
        auto pool   = makeUnprotectedPool(4);
        auto blocks = pool->malloc(batch ? 2 : 1);
        ASSERT_TRUE(blocks.has_value());
        pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
        pool->incTreeRef(*blocks, BlockTreeRefType::FROZEN);

        std::mutex                cache_mutex;
        std::vector<BlockIdxType> changes;
        auto                      observer = std::make_shared<FrozenBlockObserver>();
        observer->mutex                    = &cache_mutex;
        observer->on_changed               = [&](IBlockPool* changed_pool, BlockIdxType block) {
            EXPECT_FALSE(changed_pool->isFrozen(block));
            changes.push_back(block);
        };
        pool->setFrozenBlockObserver(observer);
        pool->setCapacityChangeCallback([&] {
            EXPECT_EQ(changes, *blocks);
            throw std::runtime_error("capacity observer failed");
        });

        if (batch) {
            EXPECT_THROW(pool->decTreeRef(*blocks, BlockTreeRefType::FROZEN), std::runtime_error);
        } else {
            EXPECT_THROW(pool->unfreezeBlock(blocks->front()), std::runtime_error);
        }
        EXPECT_EQ(changes, *blocks);
        for (const auto block : *blocks) {
            EXPECT_FALSE(pool->isFrozen(block));
        }
        pool->clearFrozenBlockObserver(observer);
        pool->setCapacityChangeCallback({});
        pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
    }
}

TEST(IBlockPoolTest, FrozenObserverRunsWhenCapacityCallbackCopyThrows) {
    struct ThrowingCopyObserver {
        explicit ThrowingCopyObserver(bool& fail): fail(fail) {}
        ThrowingCopyObserver(const ThrowingCopyObserver& other): fail(other.fail) {
            if (fail) {
                throw std::runtime_error("capacity observer copy failed");
            }
        }
        void  operator()() const {}
        bool& fail;
    };

    auto       pool  = makeUnprotectedPool(3);
    const auto block = pool->malloc().value();
    pool->incTreeRef(block, BlockTreeRefType::CACHE);
    pool->freezeBlock(block);
    std::mutex cache_mutex;
    size_t     changes   = 0;
    auto       observer  = std::make_shared<FrozenBlockObserver>();
    observer->mutex      = &cache_mutex;
    observer->on_changed = [&](IBlockPool* changed_pool, BlockIdxType changed_block) {
        EXPECT_FALSE(changed_pool->isFrozen(changed_block));
        ++changes;
    };
    pool->setFrozenBlockObserver(observer);
    bool fail_copy = false;
    pool->setCapacityChangeCallback(ThrowingCopyObserver(fail_copy));
    fail_copy = true;
    EXPECT_THROW(pool->decTreeRef(block, BlockTreeRefType::FROZEN), std::runtime_error);
    EXPECT_EQ(changes, 1u);
    EXPECT_FALSE(pool->isFrozen(block));

    pool->clearFrozenBlockObserver(observer);
    pool->setCapacityChangeCallback({});
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(IBlockPoolTest, ClearingOldFrozenObserverPreservesReplacement) {
    auto       pool  = makeUnprotectedPool(3);
    const auto block = pool->malloc().value();
    pool->incTreeRef(block, BlockTreeRefType::CACHE);
    std::mutex cache_mutex;
    size_t     old_notifications = 0;
    size_t     new_notifications = 0;
    auto       old_observer      = std::make_shared<FrozenBlockObserver>();
    old_observer->mutex          = &cache_mutex;
    old_observer->on_changed     = [&](IBlockPool*, BlockIdxType) { ++old_notifications; };
    auto new_observer            = std::make_shared<FrozenBlockObserver>();
    new_observer->mutex          = &cache_mutex;
    new_observer->on_changed     = [&](IBlockPool*, BlockIdxType) { ++new_notifications; };

    pool->setFrozenBlockObserver(old_observer);
    pool->freezeBlock(block);
    pool->setFrozenBlockObserver(new_observer);
    pool->clearFrozenBlockObserver(old_observer);
    pool->unfreezeBlock(block);
    EXPECT_EQ(old_notifications, 1u);
    EXPECT_EQ(new_notifications, 1u);

    pool->clearFrozenBlockObserver(new_observer);
    pool->freezeBlock(block);
    pool->unfreezeBlock(block);
    EXPECT_EQ(new_notifications, 1u);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(IBlockPoolTest, ConcurrentFreezeCountsEachHold) {
    auto pool  = makeUnprotectedPool(3);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    std::vector<std::thread> threads;
    constexpr size_t         thread_count = 8;
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back([&] { pool->freezeBlock(*block); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(pool->treeRefCount(*block), 1u + thread_count);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    for (size_t i = 0; i < thread_count; ++i) {
        pool->decTreeRef(*block, BlockTreeRefType::FROZEN);
        EXPECT_EQ(pool->isFrozen(*block), i + 1 < thread_count);
    }
    EXPECT_EQ(pool->treeRefCount(*block), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

TEST(IBlockPoolTest, FrozenDecrementRejectsEmptyBalanceWithoutMutatingBatch) {
    auto pool   = makeUnprotectedPool(4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incTreeRef(*blocks, BlockTreeRefType::CACHE);
    pool->freezeBlock(blocks->front());
    pool->incTreeRef(blocks->front(), BlockTreeRefType::FROZEN);

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool->decTreeRef(*blocks, BlockTreeRefType::FROZEN));
    EXPECT_EQ(pool->treeRefCount(blocks->front()), 3u);
    EXPECT_EQ(pool->treeRefCount(blocks->back()), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    pool->decTreeRef(blocks->front(), BlockTreeRefType::FROZEN);
    pool->decTreeRef(blocks->front(), BlockTreeRefType::FROZEN);
    EXPECT_ANY_THROW(pool->decTreeRef(blocks->front(), BlockTreeRefType::FROZEN));
    EXPECT_ANY_THROW(pool->decTreeRef(*blocks, BlockTreeRefType::FROZEN));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_EQ(pool->treeRefCount(blocks->front()), 1u);
    EXPECT_EQ(pool->treeRefCount(blocks->back()), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    pool->decTreeRef(*blocks, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

TEST(IBlockPoolTest, DecTreeRefRejectsWrongInternalType) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool->decTreeRef(*block, BlockTreeRefType::LOAD));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_TRUE(pool->isAllocated(*block));
    EXPECT_EQ(pool->treeRefCount(*block), 1u);
    pool->decTreeRef(*block, BlockTreeRefType::CACHE);
}

TEST(IBlockPoolTest, BatchIncTreeRefRejectsInvalidTailWithoutMutatingPrefix) {
    auto pool  = makeInitializedPool(/*physical_block_count=*/4);
    auto block = pool->malloc();
    ASSERT_TRUE(block.has_value());
    const BlockIdxType unallocated_block = *block + 1;

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool->incTreeRef(BlockIdList{*block, unallocated_block}, BlockTreeRefType::CACHE));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

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
    auto pool   = makeInitializedPool(/*physical_block_count=*/4);
    auto blocks = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    pool->incTreeRef(blocks->back(), BlockTreeRefType::LOAD);

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool->decTreeRef(*blocks, BlockTreeRefType::CACHE));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

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
