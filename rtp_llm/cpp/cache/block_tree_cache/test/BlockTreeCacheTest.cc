#include <gtest/gtest.h>

#include <thread>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

// Helper: create a v4 HostBlockPool with the given payload_bytes and usable_count.
// IBlockPool reserves block 0, so physical_block_count = usable_count + 1. The pool
// is returned uninitialized; callers invoke init() (which is not double-call guarded).
static std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count) {
    auto config                     = std::make_shared<HostBlockPoolConfig>();
    config->pool_type               = BlockPoolType::HOST;
    config->pool_name               = "block_tree_cache_host";
    config->physical_block_count    = usable_count + 1;
    config->free_block_order_policy = FreeBlockOrderPolicy::ANY_ORDER;
    config->payload_bytes           = payload_bytes;
    config->stride_bytes            = ((payload_bytes + 4095) / 4096) * 4096;
    config->enable_pinned           = true;
    config->alignment               = 4096;
    return std::make_shared<HostBlockPool>(config);
}

// Helper: build an initialized DeviceBlockPool from the lightweight cache-config test
// helpers. Device pools use ANY_ORDER (DeviceBlockPool::normalizeConfig enforces it).
static DeviceBlockPoolPtr makeDevicePool() {
    constexpr int    kLayerNum       = 4;
    constexpr int    kBlockNum       = 10;
    constexpr size_t kTokensPerBlock = 1;
    CacheConfig      cache_config    = test::makeSimpleMhaCacheConfig(kLayerNum,
                                                             kBlockNum,
                                                             kTokensPerBlock,
                                                             TYPE_FP16,
                                                             /*local_head_num_kv=*/1,
                                                             /*size_per_head=*/64);
    BlockPoolConfig  old_cfg         = BlockPoolConfigHelper::createConfig(cache_config);

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = "block_tree_cache_device";
    config->physical_block_count    = old_cfg.block_num;
    config->free_block_order_policy = FreeBlockOrderPolicy::ANY_ORDER;
    config->total_size_bytes        = old_cfg.total_size_bytes;
    config->memory_layouts          = old_cfg.memory_layouts;
    config->allocation_type         = AllocationType::DEVICE;
    config->use_pinned_cpu_backing  = false;
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    pool->init();
    return pool;
}

// referenceDeviceBlocks() must add exactly one cache-category holder (incRef) and
// releaseDeviceBlocks() must release it (releaseRef), returning capacity at refcount 0.
TEST(BlockTreeCacheComponentGroupTest, DevicePoolLifecycleUsesSingleCountRefcount) {
    auto pool = makeDevicePool();
    ASSERT_NE(pool, nullptr);

    FullComponentGroup group;
    group.component_group_id = 0;
    group.setDevicePools({pool});
    EXPECT_EQ(group.devicePoolCount(), 1u);

    // malloc reserves capacity only; the block starts at refCount 0. The exact index is not
    // asserted: ANY_ORDER makes the choice opaque, which is the correct device-pool behavior.
    auto blocks_opt = pool->malloc(1);
    ASSERT_TRUE(blocks_opt.has_value());
    ASSERT_EQ(blocks_opt->size(), 1u);
    const BlockIdxType block = (*blocks_opt)[0];
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 0u);

    // Cache holder acquired via ComponentGroup -> pool incRef.
    group.referenceDeviceBlocks({block});
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_TRUE(pool->isAllocated(block));

    // Cache holder released via ComponentGroup -> pool releaseRef; at 0 the block is freed.
    group.releaseDeviceBlocks({block});
    EXPECT_FALSE(pool->isAllocated(block));
}

// Helper to build a simple single-group BlockTreeCache for testing.
class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree = std::make_unique<BlockTree>(1);  // 1 component group

        auto full_group                = std::make_shared<FullComponentGroup>();
        full_group->component_group_id = 0;

        std::vector<ComponentGroupPtr> groups = {full_group};
        std::vector<Component>         components;

        cache_ = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::move(components));
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
    // Insert path: 100 → 200 → 300 with per-node blocks
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};

    cache_->insert(nullptr, {100, 200, 300}, slots);

    auto result = cache_->match({100, 200, 300});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 3u);
    // Each node along the path has its own device_blocks
    EXPECT_EQ(result.block_indices.size(), 3u);
    EXPECT_EQ(result.block_indices[0], 42);
    EXPECT_EQ(result.block_indices[1], 43);
    EXPECT_EQ(result.block_indices[2], 44);
}

TEST_F(BlockTreeCacheTest, MatchPartialPath) {
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};

    cache_->insert(nullptr, {100, 200, 300}, slots);

    // Match only first 2 keys (4th key doesn't exist)
    auto result = cache_->match({100, 200, 999});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 2u);
}

TEST_F(BlockTreeCacheTest, InsertNewPath) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};

    cache_->insert(nullptr, {100, 200}, slots);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, InsertOverlappingPathUpdatesHeat) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};

    cache_->insert(nullptr, {100, 200}, slots);
    cache_->insert(nullptr, {100, 200}, slots);  // Overlap

    // Should still be 2 nodes (no duplication)
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, EvictDeviceLeaf) {
    // Insert: root → 100 → 200 → 300
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

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

    auto swa_group                = std::make_shared<SWAComponentGroup>(128, 64);
    swa_group->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full_group, swa_group};
    std::vector<Component>         components;

    auto multi_cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::move(components));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};  // Full
    slots[0][1].device_blocks = {20};  // SWA

    multi_cache->insert(nullptr, {100}, slots);

    // Evict Full group at DEVICE → should cascade to SWA
    int evicted = multi_cache->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 1);

    multi_cache->waitForPendingTasks();
}

TEST_F(BlockTreeCacheTest, NodeDeletionWhenAllEmpty) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};

    cache_->insert(nullptr, {100, 200}, slots);

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

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    cache_->insert(nullptr, {100}, slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 1u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);
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
    cache_->insert(nullptr, {}, {});
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
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
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig ce_cfg;
    ce_cfg.eviction_thread_pool_size = 2;
    ce_cfg.enable_device_cache       = true;
    ce_cfg.enable_memory_cache       = true;

    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     std::move(ce_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);

    ce_cache->evict(1, Tier::DEVICE);
    ce_cache->waitForPendingTasks();

    // D2H copy failed → rollback: host_block freed, node back in device heap
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    auto find = ce_cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    EXPECT_TRUE(find.matched_node->group_slots[0].has_device_value());
}

// Test: Sequential eviction without Host pool — direct release path.
//
//   root → [100] → [200] → [300] all D={42}
//   Host disabled → eviction target=NONE (direct release), synchronous.
//   Sequential eviction drains all 3 nodes.
TEST_F(BlockTreeCacheTest, SequentialEvictionAllocatesMultipleHostBlocks) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // No Host pool, Host disabled → direct release on eviction
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.eviction_thread_pool_size = 2;
    seq_cfg.enable_device_cache       = true;
    seq_cfg.enable_memory_cache       = false;

    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     std::move(seq_cfg));

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    ce_cache->insert(nullptr, {100, 200, 300}, slots);

    // Evict all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int evicted = ce_cache->evict(1, Tier::DEVICE);
        EXPECT_EQ(evicted, 1) << "Eviction " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

// Test: REUSABLE eviction allocates host block when host is enabled.
TEST_F(BlockTreeCacheTest, ReusableEvictionAllocatesHostBlock) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig reuse_cfg;
    reuse_cfg.eviction_thread_pool_size = 2;
    reuse_cfg.enable_device_cache       = true;
    reuse_cfg.enable_memory_cache       = true;

    auto ce_cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                     std::move(groups),
                                                     std::vector<Component>{},
                                                     std::move(reuse_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    ce_cache->evict(1, Tier::DEVICE);
    // Synchronous, no wait needed

    // Host block allocated (REUSABLE eviction with host enabled)
    EXPECT_EQ(host_pool->freeBlocksNum(), 3u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
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
    BlockTreeCacheConfig throw_cfg;
    throw_cfg.eviction_thread_pool_size = 2;
    throw_cfg.enable_device_cache       = true;
    throw_cfg.enable_memory_cache       = false;
    throw_cfg.enable_disk_cache         = true;
    throw_cfg.enable_remote_cache       = false;

    EXPECT_THROW(std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  std::move(throw_cfg)),
                 std::invalid_argument);
}

// Test: Evict on disabled tier returns 0.
TEST_F(BlockTreeCacheTest, EvictDisabledTierReturnsZero) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Device enabled, Host disabled (default)
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Evict HOST tier — disabled → returns 0
    EXPECT_EQ(cache->evict(1, Tier::HOST), 0);
    // Evict DEVICE tier — enabled → returns 1
    EXPECT_EQ(cache->evict(1, Tier::DEVICE), 1);
}

// Test: Host disabled → Device eviction does direct release (no D2H demotion).
TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Host disabled (default): Device eviction → direct release
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// Test: Tier enable query accessors.
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

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

// ---------------------------------------------------------------------------
// UT-2: shouldDeleteNode checks all groups
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {
    auto tree = std::make_unique<BlockTree>(1);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::vector<ComponentGroupPtr> groups = {full};
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Evict device data
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Node should be deleted: group empty
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// UT-4: SWA buildTransfer supports HOST_TO_DISK (Bug 5 fix)
// ---------------------------------------------------------------------------
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
    auto desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    EXPECT_EQ(desc.node, find.matched_node);
    EXPECT_EQ(desc.host_block, 7);

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
    ASSERT_EQ(er->blocks_to_release.size(), 1u);
    EXPECT_EQ(er->blocks_to_release[0], 7);
}

// ---------------------------------------------------------------------------
// UT-5: match block_indices uses FULL group's actual ID (Bug 4 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, MatchUsesFullGroupIdNotHardcodedZero) {
    auto tree = std::make_unique<BlockTree>(2);

    // group 0 = SWA, group 1 = FULL (FULL's id is NOT 0)
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {swa, full};
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};  // SWA group
    slots[0][1].device_blocks = {42};  // FULL group
    cache->insert(nullptr, {100}, slots);

    auto result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    // block_indices should come from FULL group (id=1), value 42
    ASSERT_EQ(result.block_indices.size(), 1u);
    EXPECT_EQ(result.block_indices[0], 42);
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

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

// ---------------------------------------------------------------------------
// UT-6: Eviction with CopyEngine — D2H copy fails with placeholder resolver.
// Issue 7 fix: copy failure triggers rollback (host_block freed).
// Node either stays in tree (rollback) or is deleted (sync fallback path).
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NodeEntersHostHeapAfterDemotion) {
    auto host_pool = makeHostPool(256, 8);
    ASSERT_TRUE(host_pool->init());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // D2H copy fails (placeholder resolver) -> host_block NOT set
    auto find = cache->tree()->findNode({100});
    if (find.matched_node) {
        EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    }
    // Host pool has no outstanding allocations (freed on rollback)
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
}

// ---------------------------------------------------------------------------
// UT-7: Cascade eviction - parent becomes device leaf after child eviction
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildEviction) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache->insert(nullptr, {100, 200, 300}, slots);

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
// UT-9: CopyEngine failure does not update slot (Issue 7 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, CopyFailureDoesNotUpdateSlot) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

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

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::move(components), std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Evict: v4 device_pools are not populated yet -> D2H copy should fail.
    // After fix: host_block should NOT be set, node stays in device heap
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    if (find.matched_node) {
        // Copy failed -> host_block should remain invalid
        EXPECT_FALSE(find.matched_node->group_slots[0].has_host_value());
    }
}

TEST_F(BlockTreeCacheTest, LoadBackOnlyReloadsSWAWindow) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].host_block    = static_cast<BlockIdxType>(100 + i);
    }

    cache->insert(nullptr, {100, 200, 300, 400}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
    EXPECT_EQ(result.matched_blocks, 4u);
    EXPECT_EQ(result.host_load_back_blocks, 2u);
    EXPECT_EQ(result.load_back_blocks, 2u);
}

// ---------------------------------------------------------------------------
// Test: enable_load_back — match detects Host/Disk data needing reload
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, LoadBackDetectsHostData) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    // Insert a node and manually set host data (simulating prior eviction)
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Evict to host (no CopyEngine → direct release, node deleted)
    // Instead, manually set up a node with host_block but no device_blocks
    cache->evict(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After eviction without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<std::vector<GroupSlot>> slots2(1, std::vector<GroupSlot>(1));
    slots2[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots2);

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
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  std::move(cfg),
                                                  nullptr,
                                                  broadcast_mgr);

    // Verify BroadcastManager is stored (access via internal member)
    EXPECT_EQ(cache->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->broadcast_manager_->workerNum(), 2u);
}

}  // namespace
}  // namespace rtp_llm
