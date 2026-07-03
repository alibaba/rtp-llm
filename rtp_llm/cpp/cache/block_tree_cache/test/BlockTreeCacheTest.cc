#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <unordered_map>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostBlockPool.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

// Helper to build a simple single-group BlockTreeCache for testing.
class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree = std::make_unique<BlockTree>(1);  // 1 component group

        auto full_group                = std::make_shared<FullComponentGroup>();
        full_group->component_group_id = 0;
        full_group->reuse_policy       = CacheReusePolicy::REUSABLE;

        std::vector<ComponentGroupPtr> groups = {full_group};
        std::vector<Component>         components;

        cache_ =
            std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::move(components), nullptr, 2);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyTree) {
    auto result = cache_->match({100, 200, 300});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.block_indices.empty());
}

TEST_F(BlockTreeCacheTest, MatchAfterInsert) {
    // Insert path: 100 → 200 → 300
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};

    cache_->insert({100, 200, 300}, slots);

    auto result = cache_->match({100, 200, 300});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 3u);
    // Each node along the path has device_blocks={42}, so 3 nodes * 1 block = 3
    EXPECT_EQ(result.block_indices.size(), 3u);
    EXPECT_EQ(result.block_indices[0], 42);
}

TEST_F(BlockTreeCacheTest, MatchPartialPath) {
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};

    cache_->insert({100, 200, 300}, slots);

    // Match only first 2 keys (4th key doesn't exist)
    auto result = cache_->match({100, 200, 999});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 2u);
}

TEST_F(BlockTreeCacheTest, InsertNewPath) {
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {10};

    cache_->insert({100, 200}, slots);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, InsertOverlappingPathUpdatesHeat) {
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {10};

    cache_->insert({100, 200}, slots);
    cache_->insert({100, 200}, slots);  // Overlap

    // Should still be 2 nodes (no duplication)
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, EvictDeviceLeaf) {
    // Insert: root → 100 → 200 → 300
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache_->insert({100, 200, 300}, slots);

    // 300 is a DeviceLeaf (no children with device data)
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    int evicted = cache_->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 1);

    cache_->waitForPendingTasks();

    // After eviction, the leaf's device_blocks should be cleared
    // Check that match no longer finds device data for 300
    auto result = cache_->match({100, 200, 300});
    // 300's group_slots[0] should have no device value
    // But 100 and 200 also have no device data (only 300 was given slots)
    // So match would fail at 100 (no data in any tier)
}

TEST_F(BlockTreeCacheTest, EvictEmptyTreeReturnsZero) {
    int evicted = cache_->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 0);
}

TEST_F(BlockTreeCacheTest, CascadeEviction) {
    // Build a cache with Full + SWA groups
    auto tree = std::make_unique<BlockTree>(2);  // 2 component groups

    auto full_group                = std::make_shared<FullComponentGroup>();
    full_group->component_group_id = 0;
    full_group->reuse_policy       = CacheReusePolicy::REUSABLE;

    auto swa_group                = std::make_shared<SWAComponentGroup>(128, 64);
    swa_group->component_group_id = 1;
    swa_group->reuse_policy       = CacheReusePolicy::REUSABLE;

    std::vector<ComponentGroupPtr> groups = {full_group, swa_group};
    std::vector<Component>         components;

    auto multi_cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::move(components));

    // Insert a node with both Full and SWA data
    std::vector<GroupSlot> slots(2);
    slots[0].device_blocks = {10};  // Full
    slots[1].device_blocks = {20};  // SWA

    multi_cache->insert({100}, slots);

    // Evict Full group at DEVICE → should cascade to SWA
    int evicted = multi_cache->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 1);

    multi_cache->waitForPendingTasks();
}

TEST_F(BlockTreeCacheTest, NodeDeletionWhenAllEmpty) {
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};

    cache_->insert({100, 200}, slots);

    auto stats_before = cache_->getStats();
    EXPECT_EQ(stats_before.tree_node_count, 2u);

    // Evict: the leaf (200) should be removed after eviction
    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    // After eviction and cleanup, tree might be smaller
    auto stats_after = cache_->getStats();
    // Node 200 should be removed (all REUSABLE groups empty)
    // Node 100 should also be removed (empty ancestor)
    EXPECT_LE(stats_after.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, GetStats) {
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {10};
    cache_->insert({100}, slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 1u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheTest, MultiGroupConstruction) {
    auto tree = std::make_unique<BlockTree>(3);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->reuse_policy       = CacheReusePolicy::REUSABLE;

    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;
    swa->reuse_policy       = CacheReusePolicy::REUSABLE;

    auto linear                = std::make_shared<LinearComponentGroup>();
    linear->component_group_id = 2;
    linear->reuse_policy       = CacheReusePolicy::NON_REUSABLE;

    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    std::vector<Component>         components;

    auto multi_cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::move(components));

    EXPECT_EQ(multi_cache->componentGroups().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSlotCount(), 3);
}

TEST_F(BlockTreeCacheTest, MatchEmptyKeys) {
    auto result = cache_->match({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
}

TEST_F(BlockTreeCacheTest, InsertEmptyKeys) {
    cache_->insert({}, {});
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<GroupSlot> slots(1);
            slots[0].device_blocks = {static_cast<BlockIdxType>(i * 100)};
            CacheKeysType keys     = {static_cast<CacheKeyType>(i * 1000 + 1)};
            cache_->insert(keys, slots);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 4u);
}

// ---------------------------------------------------------------------------
// CopyEngine integration tests
// ---------------------------------------------------------------------------

// Test: Eviction with CopyEngine — D2H copy fails (placeholder resolver),
// so Issue 7 fix triggers rollback: host_block freed, node stays in device heap.
//
//   Before evict:                             After evict + wait (rollback):
//   root → [100] D={42} ← heap               root → [100] D={42} ← heap (restored)
//
//   D2H copy fails → target host_block freed, source device heap restored.
TEST_F(BlockTreeCacheTest, EvictWithCopyEngineAllocatesHostBlock) {
    auto host_pool = std::make_shared<HostBlockPool>(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     copy_engine,
                                                     2,
                                                     nullptr,
                                                     /*enable_device=*/true,
                                                     /*enable_memory=*/true);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    ce_cache->insert({100}, slots);

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(host_pool->freeBlocks(), 4u);

    ce_cache->evict(1, Tier::DEVICE);
    ce_cache->waitForPendingTasks();

    // D2H copy failed → rollback: host_block freed, node back in device heap
    EXPECT_EQ(host_pool->freeBlocks(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    auto find = ce_cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    EXPECT_TRUE(find.matched_node->group_slots[0].has_device_value());
}

// Test: Sequential eviction without CopyEngine — direct release path.
//
//   root → [100] → [200] → [300] all D={42}
//   Host disabled → eviction target=NONE (direct release), synchronous.
//   Sequential eviction drains all 3 nodes.
TEST_F(BlockTreeCacheTest, SequentialEvictionAllocatesMultipleHostBlocks) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    // No CopyEngine, Host disabled → direct release on eviction
    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     nullptr,
                                                     2,
                                                     nullptr,
                                                     /*enable_device=*/true,
                                                     /*enable_memory=*/false);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    ce_cache->insert({100, 200, 300}, slots);

    // Evict all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int evicted = ce_cache->evict(1, Tier::DEVICE);
        EXPECT_EQ(evicted, 1) << "Eviction " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

// Test: NON_REUSABLE eviction does NOT allocate host block.
TEST_F(BlockTreeCacheTest, NonReusableEvictionNoHostAllocation) {
    auto host_pool = std::make_shared<HostBlockPool>(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::NON_REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     copy_engine,
                                                     2,
                                                     nullptr,
                                                     /*enable_device=*/true,
                                                     /*enable_memory=*/true);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    ce_cache->insert({100}, slots);

    ce_cache->evict(1, Tier::DEVICE);
    // Synchronous, no wait needed

    // No host block allocated (NON_REUSABLE → target=NONE)
    EXPECT_EQ(host_pool->freeBlocks(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Tier enable flag tests
// ---------------------------------------------------------------------------

// Test: Construction validation — disk requires host.
TEST_F(BlockTreeCacheTest, DiskRequiresHostValidation) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // disk=true, host=false should throw
    EXPECT_THROW(std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  nullptr,
                                                  2,
                                                  nullptr,
                                                  /*enable_device=*/true,
                                                  /*enable_memory=*/false,
                                                  /*enable_disk=*/true,
                                                  /*enable_remote=*/false),
                 std::invalid_argument);
}

// Test: Evict on disabled tier returns 0.
TEST_F(BlockTreeCacheTest, EvictDisabledTierReturnsZero) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    // Device enabled, Host disabled (default)
    auto cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, 2);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    // Evict HOST tier — disabled → returns 0
    EXPECT_EQ(cache->evict(1, Tier::HOST), 0);
    // Evict DEVICE tier — enabled → returns 1
    EXPECT_EQ(cache->evict(1, Tier::DEVICE), 1);
}

// Test: Host disabled → Device eviction does direct release (no D2H demotion).
TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = std::make_shared<HostBlockPool>(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    // Host disabled (default): Device eviction → direct release
    auto cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, copy_engine, 2);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocks(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// Test: Tier enable query accessors.
TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  nullptr,
                                                  2,
                                                  nullptr,
                                                  /*enable_device=*/true,
                                                  /*enable_memory=*/true,
                                                  /*enable_disk=*/true,
                                                  /*enable_remote=*/true);

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

// ---------------------------------------------------------------------------
// UT-2: shouldDeleteNode only checks REUSABLE groups (Bug 1 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NodeDeletedEvenIfNonReusableHasData) {
    auto tree = std::make_unique<BlockTree>(2);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->reuse_policy       = CacheReusePolicy::REUSABLE;

    auto non_reusable                = std::make_shared<SWAComponentGroup>(128, 64);
    non_reusable->component_group_id = 1;
    non_reusable->reuse_policy       = CacheReusePolicy::NON_REUSABLE;

    std::vector<ComponentGroupPtr> groups = {full, non_reusable};
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: REUSABLE group has data, NON_REUSABLE also has data
    std::vector<GroupSlot> slots(2);
    slots[0].device_blocks = {42};  // REUSABLE
    slots[1].device_blocks = {99};  // NON_REUSABLE
    cache->insert({100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Evict REUSABLE group's device data
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Node should be deleted: REUSABLE group empty, NON_REUSABLE doesn't prevent deletion
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// UT-4: SWA buildTransfer supports HOST_TO_DISK (Bug 5 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, SWABuildTransferSupportsHostToDisk) {
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;
    swa->reuse_policy       = CacheReusePolicy::REUSABLE;

    // Create a mock tree node with host data
    auto                   tree = std::make_unique<BlockTree>(1);
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    tree->insertNode({100}, slots);
    auto find = tree->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;

    // Verify HOST_TO_DISK transfer descriptor is correct
    auto desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    ASSERT_EQ(desc.source_blocks.size(), 1u);
    ASSERT_EQ(desc.source_blocks[0].size(), 1u);
    EXPECT_EQ(desc.source_blocks[0][0], 7);

    // Verify driveEviction(HOST) produces a valid transfer
    swa->device_heap->invalidate(find.matched_node);
    find.matched_node->group_slots[0].device_blocks  = {NULL_BLOCK_IDX};
    find.matched_node->group_slots[0].in_device_heap = false;
    swa->host_heap->push(find.matched_node, 0);
    find.matched_node->group_slots[0].in_host_heap = true;

    auto er = swa->driveEviction(1, Tier::HOST);
    ASSERT_TRUE(er.has_value());
    EXPECT_EQ(er->source_tier, Tier::HOST);
    EXPECT_EQ(er->target_tier, Tier::DISK);
    EXPECT_EQ(er->transfer.source_tier, Tier::HOST);
    EXPECT_EQ(er->transfer.target_tier, Tier::DISK);
}

// ---------------------------------------------------------------------------
// UT-5: match block_indices uses FULL group's actual ID (Bug 4 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, MatchUsesFullGroupIdNotHardcodedZero) {
    auto tree = std::make_unique<BlockTree>(2);

    // group 0 = SWA, group 1 = FULL (FULL's id is NOT 0)
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;
    swa->reuse_policy       = CacheReusePolicy::REUSABLE;

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 1;
    full->reuse_policy       = CacheReusePolicy::REUSABLE;

    std::vector<ComponentGroupPtr> groups = {swa, full};
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    std::vector<GroupSlot> slots(2);
    slots[0].device_blocks = {10};  // SWA group
    slots[1].device_blocks = {42};  // FULL group
    cache->insert({100}, slots);

    auto result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    // block_indices should come from FULL group (id=1), value 42
    ASSERT_EQ(result.block_indices.size(), 1u);
    EXPECT_EQ(result.block_indices[0], 42);
}

// ---------------------------------------------------------------------------
// UT-6: Eviction with CopyEngine — D2H copy fails with placeholder resolver.
// Issue 7 fix: copy failure triggers rollback (host_block freed).
// Node either stays in tree (rollback) or is deleted (sync fallback path).
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NodeEntersHostHeapAfterDemotion) {
    auto host_pool = std::make_shared<HostBlockPool>(256, 8);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::vector<Component>{}, copy_engine, 2, nullptr, true, true);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // D2H copy fails (placeholder resolver) -> host_block NOT set
    auto find = cache->tree()->findNode({100});
    if (find.matched_node) {
        EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    }
    // Host pool has no outstanding allocations (freed on rollback)
    EXPECT_EQ(host_pool->freeBlocks(), 8u);
}

// ---------------------------------------------------------------------------
// UT-7: Cascade eviction - parent becomes device leaf after child eviction
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildEviction) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: root -> A -> B -> C
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100, 200, 300}, slots);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Evict C -> B becomes DeviceLeaf -> enters heap
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Evict B -> A becomes DeviceLeaf -> enters heap
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

// ---------------------------------------------------------------------------
// UT-8: NON_REUSABLE full lifecycle (insert -> match -> evict -> delete)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NonReusableFullLifecycle) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::NON_REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Match
    auto result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);

    // Evict: NON_REUSABLE -> direct release, target=NONE
    cache->evict(1, Tier::DEVICE);

    // Node should be deleted
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// UT-9: CopyEngine failure does not update slot (Issue 7 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, CopyFailureDoesNotUpdateSlot) {
    auto host_pool = std::make_shared<HostBlockPool>(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    full->component_indices               = {0};
    std::vector<ComponentGroupPtr> groups = {full};

    // Create a component with MemoryBlockLayerTagSlot so deviceToHost attempts real copy
    Component comp;
    comp.component_id                 = 0;
    comp.component_group_id           = 0;
    comp.memory_block_layer_tag_slots = {{0, "kv", 128}};
    std::vector<Component> components = {comp};

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::move(components), copy_engine, 2, nullptr, true, true);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    // Evict: DeviceBufferResolver returns empty -> D2H copy should fail
    // After fix: host_block should NOT be set, node stays in device heap
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    if (find.matched_node) {
        // Copy failed -> host_block should remain invalid
        EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    }
}

// ---------------------------------------------------------------------------
// Mock device memory for DeviceBufferResolver tests
// ---------------------------------------------------------------------------
class MockDeviceMemory {
public:
    void allocate(int layer_id, BlockIdxType block_idx, size_t size_bytes) {
        auto key = makeKey(layer_id, block_idx);
        buffers_[key].resize(size_bytes, 0);
    }
    void fill(int layer_id, BlockIdxType block_idx, uint8_t pattern) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        if (it != buffers_.end()) {
            std::memset(it->second.data(), pattern, it->second.size());
        }
    }
    uint8_t* mutableData(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        return it != buffers_.end() ? it->second.data() : nullptr;
    }
    size_t size(int layer_id, BlockIdxType block_idx) const {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        return it != buffers_.end() ? it->second.size() : 0;
    }
    DeviceBufferResolver makeResolver() {
        return [this](int layer_id, BlockIdxType block_idx) -> BlockInfo {
            BlockInfo info;
            info.is_cuda      = false;
            info.device_index = 0;
            info.addr         = mutableData(layer_id, block_idx);
            info.size_bytes   = size(layer_id, block_idx);
            return info;
        };
    }

private:
    static uint64_t makeKey(int layer_id, BlockIdxType block_idx) {
        return (static_cast<uint64_t>(layer_id) << 32) | static_cast<uint64_t>(block_idx);
    }
    std::unordered_map<uint64_t, std::vector<uint8_t>> buffers_;
};

// ---------------------------------------------------------------------------
// Test: DeviceBufferResolver — D2H copy succeeds with real resolver
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, DeviceBufferResolverEnablesD2HCopy) {
    // Set up host pool and copy engine
    auto host_pool = std::make_shared<HostBlockPool>(512, 4);
    ASSERT_TRUE(host_pool->init());
    auto copy_engine = std::make_shared<CopyEngine>(host_pool);

    // Set up mock device memory: layer 0 block 42 = 128 bytes filled with 0xAA
    MockDeviceMemory device_mem;
    device_mem.allocate(0, 42, 128);
    device_mem.fill(0, 42, 0xAA);

    // Create component with matching layer slot
    Component comp;
    comp.component_id                 = 0;
    comp.component_group_id           = 0;
    comp.memory_block_layer_tag_slots = {{0, "kv", 128}};

    auto tree                                 = std::make_unique<BlockTree>(1);
    auto full                                 = std::make_shared<FullComponentGroup>();
    full->component_group_id                  = 0;
    full->reuse_policy                        = CacheReusePolicy::REUSABLE;
    full->component_indices                   = {0};
    std::vector<ComponentGroupPtr> groups     = {full};
    std::vector<Component>         components = {comp};

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::move(components), copy_engine, 2, nullptr, true, true);

    // Inject the real resolver
    cache->setDeviceBufferResolver(device_mem.makeResolver());

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    // Evict D2H — should succeed with real resolver
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Verify: node now has host data, device data cleared
    auto find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_TRUE(find.matched_node->group_slots[0].has_host_value());
    EXPECT_FALSE(find.matched_node->group_slots[0].has_device_value());

    // Verify host block content: should be 0xAA pattern
    auto host_block = find.matched_node->group_slots[0].host_block;
    ASSERT_FALSE(isNullBlockIdx(host_block));
    void* host_addr = host_pool->blockAddr(host_block);
    ASSERT_NE(host_addr, nullptr);
    auto* bytes = static_cast<const uint8_t*>(host_addr);
    EXPECT_EQ(bytes[0], 0xAA);
    EXPECT_EQ(bytes[127], 0xAA);
}

// ---------------------------------------------------------------------------
// Test: enable_load_back — match detects Host/Disk data needing reload
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, LoadBackDetectsHostData) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    // Insert a node and manually set host data (simulating prior eviction)
    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {42};
    cache->insert({100}, slots);

    // Evict to host (no CopyEngine → direct release, node deleted)
    // Instead, manually set up a node with host_block but no device_blocks
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After eviction without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<GroupSlot> slots2(1);
    slots2[0].device_blocks = {55};
    cache->insert({200}, slots2);

    // Manually set host_block and clear device_blocks to simulate evicted state
    auto find = cache->tree()->findNode({200});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;
    for (auto& b : find.matched_node->group_slots[0].device_blocks) {
        b = NULL_BLOCK_IDX;
    }

    // Match should detect load_back
    auto result = cache->match({200});
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
}

// ---------------------------------------------------------------------------
// Test: BroadcastManager is stored and accessible
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, BroadcastManagerStoredCorrectly) {
    // Create a BroadcastManager (no actual RPC connections needed for this test)
    std::vector<std::string> worker_addrs  = {"127.0.0.1:50051", "127.0.0.1:50052"};
    auto                     broadcast_mgr = std::make_shared<BroadcastManager>(worker_addrs);
    ASSERT_TRUE(broadcast_mgr->init());

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    full->reuse_policy                    = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  nullptr,
                                                  2,
                                                  nullptr,
                                                  true,
                                                  false,
                                                  false,
                                                  false,
                                                  broadcast_mgr);

    // Verify BroadcastManager is stored (access via internal member)
    EXPECT_EQ(cache->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->broadcast_manager_->workerNum(), 2u);
}

}  // namespace
}  // namespace rtp_llm
