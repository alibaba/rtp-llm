#include <gtest/gtest.h>

#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full_group                       = std::make_shared<FullComponentGroup>();
        full_group->component_group_id        = 0;
        std::vector<ComponentGroupPtr> groups = {full_group};
        std::vector<Component>         components;

        cache_ = BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyThenFullAndPartialPath) {
    BlockTreeMatchResult empty_result = cache_->match({100, 200, 300});
    EXPECT_EQ(empty_result.matched_node, nullptr);
    EXPECT_EQ(empty_result.matched_blocks, 0u);
    EXPECT_TRUE(empty_result.group_block_indices.empty());

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult full_result = cache_->match({100, 200, 300});
    ASSERT_NE(full_result.matched_node, nullptr);
    EXPECT_EQ(full_result.matched_node->cache_key, 300);
    EXPECT_EQ(full_result.matched_blocks, 3u);
    ASSERT_EQ(full_result.group_block_indices.count(0), 1u);
    EXPECT_EQ(full_result.group_block_indices.at(0), (BlockIndicesType{42, 43, 44}));
    cache_->releaseMatchedBlocks(full_result.matched_block_sets);

    BlockTreeMatchResult partial_result = cache_->match({100, 200, 999});
    ASSERT_NE(partial_result.matched_node, nullptr);
    EXPECT_EQ(partial_result.matched_node->cache_key, 200);
    EXPECT_EQ(partial_result.matched_blocks, 2u);
    ASSERT_EQ(partial_result.group_block_indices.count(0), 1u);
    EXPECT_EQ(partial_result.group_block_indices.at(0), (BlockIndicesType{42, 43}));
    cache_->releaseMatchedBlocks(partial_result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, FullMatch_StopsAtFirstGap) {
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[2][0].device_blocks = {12};

    cache_->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache_->match({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 100);
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10}));

    cache_->releaseMatchedBlocks(result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, DuplicateInsertDoesNotCreateNodes) {
    CacheStats stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<std::vector<GroupSlot>> original_slots(2, std::vector<GroupSlot>(1));
    original_slots[0][0].device_blocks = {10};
    original_slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, original_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    std::vector<std::vector<GroupSlot>> duplicate_slots(2, std::vector<GroupSlot>(1));
    duplicate_slots[0][0].device_blocks = {20};
    duplicate_slots[1][0].device_blocks = {21};
    cache_->insert(nullptr, {100, 200}, duplicate_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    BlockTreeFindResult find_result = cache_->tree()->findNode({100, 200});
    ASSERT_EQ(find_result.path.size(), 2u);
    EXPECT_EQ(find_result.path[0]->group_slots[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(find_result.path[1]->group_slots[0].device_blocks, (BlockIndicesType{11}));
}

TEST_F(BlockTreeCacheTest, ReclaimCascadesToLowerPriorityGroup) {
    // Build a cache with Full + SWA groups
    auto tree = std::make_unique<BlockTree>(2);  // 2 component groups

    auto full_group                = std::make_shared<FullComponentGroup>();
    full_group->component_group_id = 0;

    auto swa_group                = std::make_shared<SWAComponentGroup>(128, 64);
    swa_group->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full_group, swa_group};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};  // Full
    slots[0][1].device_blocks = {20};  // SWA

    multi_cache->insert(nullptr, {100}, slots);

    // Reclaim Full group at DEVICE → should cascade to SWA.
    int reclaimed = multi_cache->reclaimBlocks(1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);

    multi_cache->waitForPendingTasks();
}

TEST_F(BlockTreeCacheTest, MultiGroupConstruction) {
    auto tree = std::make_unique<BlockTree>(3);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;

    auto linear                = std::make_shared<LinearComponentGroup>();
    linear->component_group_id = 2;

    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    EXPECT_EQ(multi_cache->componentGroups().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSlotCount(), 3);
}

TEST(BlockTreeCacheConstructionTest, OutOfRangeComponentGroupIdFailsInitializationWithoutThrowing) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 1;
    std::vector<ComponentGroupPtr> groups = {full};
    std::vector<Component>         components;

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  nullptr,
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
}

TEST_F(BlockTreeCacheTest, EmptyKeysAreNoOps) {
    const CacheStats stats_before = cache_->getStats();
    cache_->insert(nullptr, {}, {});
    const CacheStats stats_after = cache_->getStats();
    EXPECT_EQ(stats_after.tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(stats_after.device_heap_total_size, stats_before.device_heap_total_size);

    BlockTreeMatchResult result = cache_->match({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.load_back_ticket, nullptr);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
            slots[0][0].device_blocks = {static_cast<BlockIdxType>(i * 100)};
            CacheKeysType keys        = {static_cast<CacheKeyType>(i * 1000 + 1)};
            cache_->insert(nullptr, keys, slots);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 4u);
}

TEST_F(BlockTreeCacheTest, FullMatch_PreservesPathAndPoolOrder) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 8;
    auto             pool0         = makeDevicePool({{64, 0}}, kUsableBlocks, "full_order_pool0");
    auto             pool1         = makeDevicePool({{64, 0}}, kUsableBlocks, "full_order_pool1");

    auto pool0_prefix = pool0->malloc(1);
    auto pool1_prefix = pool1->malloc(3);
    ASSERT_TRUE(pool0_prefix.has_value());
    ASSERT_TRUE(pool1_prefix.has_value());

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool0, pool1});
    std::vector<ComponentGroupPtr> groups = {full};
    auto cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 2);
    ASSERT_EQ(request_blocks.per_node.size(), 2u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    ASSERT_EQ(request_blocks.per_node[1].size(), 2u);

    const BlockIdxType a_pool0 = request_blocks.per_node[0][0];
    const BlockIdxType a_pool1 = request_blocks.per_node[0][1];
    const BlockIdxType b_pool0 = request_blocks.per_node[1][0];
    const BlockIdxType b_pool1 = request_blocks.per_node[1][1];
    EXPECT_NE(a_pool0, a_pool1);
    EXPECT_NE(b_pool0, b_pool1);

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {a_pool0};
    slots[0][1].device_blocks = {a_pool1};
    slots[1][0].device_blocks = {b_pool0};
    slots[1][1].device_blocks = {b_pool1};
    cache->insert(nullptr, {100, 200}, slots);
    full->unreferenceBlocks(request_blocks);
    EXPECT_TRUE(pool0->isAllocated(a_pool0));
    EXPECT_TRUE(pool0->isAllocated(b_pool0));
    EXPECT_TRUE(pool1->isAllocated(a_pool1));
    EXPECT_TRUE(pool1->isAllocated(b_pool1));

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    ASSERT_EQ(result.group_block_indices.count(1), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{a_pool0, b_pool0}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{a_pool1, b_pool1}));
    cache->releaseMatchedBlocks(result.matched_block_sets);

    EXPECT_EQ(cache->reclaimBlocks(2, Tier::DEVICE), 2);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool0->isAllocated(a_pool0));
    EXPECT_FALSE(pool0->isAllocated(b_pool0));
    EXPECT_FALSE(pool1->isAllocated(a_pool1));
    EXPECT_FALSE(pool1->isAllocated(b_pool1));

    pool0->free(*pool0_prefix);
    pool1->free(*pool1_prefix);
    EXPECT_EQ(pool0->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(pool1->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, DuplicateInsert_KeepsExistingSlotAndCallerOwnsLoser) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "duplicate_insert_pool");

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet existing = full->allocateBlocks(Tier::DEVICE, 1);
    GroupBlockSet loser    = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(existing.per_node.size(), 1u);
    ASSERT_EQ(loser.per_node.size(), 1u);
    ASSERT_EQ(existing.per_node[0].size(), 1u);
    ASSERT_EQ(loser.per_node[0].size(), 1u);
    const BlockIdxType existing_block = existing.per_node[0][0];
    const BlockIdxType loser_block    = loser.per_node[0][0];
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    std::vector<std::vector<GroupSlot>> first_slots(1, std::vector<GroupSlot>(1));
    first_slots[0][0].device_blocks = existing.per_node[0];
    cache->insert(nullptr, {100}, first_slots);
    EXPECT_EQ(pool->refCount(existing_block), 2u);
    BlockTreeFindResult initial_find = cache->tree()->findNode({100});
    ASSERT_NE(initial_find.matched_node, nullptr);
    GroupBlockSet released_existing = existing;
    released_existing.nodes         = {initial_find.matched_node};
    cache->releaseMatchedBlocks({released_existing});
    EXPECT_EQ(pool->refCount(existing_block), 1u);

    std::vector<std::vector<GroupSlot>> duplicate_slots(1, std::vector<GroupSlot>(1));
    duplicate_slots[0][0].device_blocks = loser.per_node[0];
    cache->insert(nullptr, {100}, duplicate_slots);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(find.matched_node->group_slots[0].device_blocks, (std::vector<BlockIdxType>{existing_block}));
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    full->unreferenceBlocks(loser);
    EXPECT_FALSE(pool->isAllocated(loser_block));
    EXPECT_TRUE(pool->isAllocated(existing_block));

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(existing_block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, InsertMatchReleaseReclaim_RefcountLifecycle) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "refcount_lifecycle_pool");

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks - 1);
    EXPECT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);
    EXPECT_EQ(pool->refCount(block), 2u);

    full->unreferenceBlocks(request_blocks);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{block}));
    ASSERT_EQ(result.matched_block_sets.size(), 1u);
    EXPECT_EQ(result.matched_block_sets[0].component_group_id, 0);
    EXPECT_EQ(result.matched_block_sets[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[0].per_node, (std::vector<std::vector<BlockIdxType>>{{block}}));
    EXPECT_EQ(pool->refCount(block), 2u);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 0);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    result.matched_block_sets.clear();
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotAllocateHostBlock) {
    auto host_pool           = makeHostPool(256, 4);
    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig ce_cfg;
    ce_cfg.eviction_thread_pool_size = 2;
    ce_cfg.enable_device_cache       = true;
    ce_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(ce_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    ce_cache->waitForPendingTasks();

    // Direct reclaim bypasses demotion/copy, so host capacity is unchanged.
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
    auto find = ce_cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
}

TEST_F(BlockTreeCacheTest, SequentialReclaimDrainsChainWithoutHostBlocks) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // No Host pool, Host disabled → direct release on reclaim.
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.eviction_thread_pool_size = 2;
    seq_cfg.enable_device_cache       = true;
    seq_cfg.enable_memory_cache       = false;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(seq_cfg));

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    ce_cache->insert(nullptr, {100, 200, 300}, slots);

    // Reclaim all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int reclaimed = ce_cache->reclaimBlocks(1, Tier::DEVICE);
        EXPECT_EQ(reclaimed, 1) << "Reclaim " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ReusableReclaimDoesNotAllocateHostBlock) {
    auto host_pool = makeHostPool(256, 4);

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig reuse_cfg;
    reuse_cfg.eviction_thread_pool_size = 2;
    reuse_cfg.enable_device_cache       = true;
    reuse_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(reuse_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    // Synchronous, no wait needed

    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, DiskRequiresHostValidation) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.eviction_thread_pool_size = 2;
    config.enable_device_cache       = true;
    config.enable_memory_cache       = false;
    config.enable_disk_cache         = true;
    config.enable_remote_cache       = false;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(config));
    EXPECT_EQ(cache, nullptr);
}

TEST_F(BlockTreeCacheTest, ReclaimDisabledTierReturnsZero) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Device enabled, Host disabled (default)
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim HOST tier — disabled → returns 0
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::HOST), 0);
    // Reclaim DEVICE tier — enabled → returns 1
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
}

TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = makeHostPool(256, 4);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Host disabled (default): Device reclaim → direct release.
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {
    auto tree = std::make_unique<BlockTree>(1);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Reclaim device data.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Node should be deleted: group empty
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, SWABuildTransferSupportsHostToDisk) {
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;

    // Create a mock tree node with host data
    auto                                tree = std::make_unique<BlockTree>(1);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    tree->insertNode(nullptr, {100}, slots);
    auto find = tree->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;

    // Verify HOST_TO_DISK transfer descriptor is correct
    TransferDescriptor desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    EXPECT_EQ(desc.host_block, 7);

}

TEST_F(BlockTreeCacheTest, MatchCollectsBlocksSelectedByGroupPolicy) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(3);

    std::shared_ptr<FullComponentGroup> full     = std::make_shared<FullComponentGroup>();
    full->component_group_id                     = 0;
    std::shared_ptr<LinearComponentGroup> linear = std::make_shared<LinearComponentGroup>();
    linear->component_group_id                   = 1;
    std::shared_ptr<SWAComponentGroup> swa       = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                      = 2;

    std::vector<ComponentGroupPtr>  component_groups = {full, linear, swa};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(component_groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{22}));
    EXPECT_EQ(result.group_block_indices.at(2), (BlockIndicesType{31, 32}));
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({DeviceBlockPoolPtr{}, DeviceBlockPoolPtr{}});

    std::vector<ComponentGroupPtr>             component_groups = {full};
    std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping  = {{0, 0}, {0, 1}};
    std::unique_ptr<BlockTreeCache>            cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::vector<Component>{},
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::shared_ptr<BroadcastManager>{},
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr, nullptr},
                                         std::move(per_tag_mapping));
    ASSERT_TRUE(cache->init());

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[0][1].device_blocks = {20};
    slots[1][0].device_blocks = {11};
    slots[1][1].device_blocks = {21};
    cache->insert(nullptr, {100, 200}, slots);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{20, 21}));
}

TEST_F(BlockTreeCacheTest, InitializationRequiresPerTagMapping) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({DeviceBlockPoolPtr{}});

    std::vector<ComponentGroupPtr>  component_groups = {full};
    std::unique_ptr<BlockTreeCache> cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::vector<Component>{},
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::shared_ptr<BroadcastManager>{},
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                         std::vector<BlockTreeCache::PerTagMapping>{});
    EXPECT_FALSE(cache->init());
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 1;

    std::vector<ComponentGroupPtr>  groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    slots[2][0].device_blocks = {12};
    slots[3][0].device_blocks = {13};
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};

    cache->insert(nullptr, {100, 200, 300, 400}, slots);

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_blocks, 1u);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_blocks, 4u);
}

TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache->insert(nullptr, {100, 200, 300}, slots);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim C -> B becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim B -> A becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotUpdateHostSlot) {
    auto host_pool = makeHostPool(256, 4);

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->component_indices  = {0};
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    // Create a component with MemoryBlockLayerTagSlot so deviceToHost attempts real copy
    Component comp;
    comp.component_id                 = 0;
    comp.component_group_id           = 0;
    comp.memory_block_layer_tag_slots = {{0, "kv", 128}};
    std::vector<Component> components = {comp};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::move(components), std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
}

}  // namespace
}  // namespace rtp_llm
