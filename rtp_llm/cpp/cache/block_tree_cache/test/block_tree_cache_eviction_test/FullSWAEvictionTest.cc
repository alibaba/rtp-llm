#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using block_tree_cache_test::BlockTreeCacheTestPeer;

// Helper: build a BlockTreeCache with Full(REUSABLE, gid=0) + SWA(REUSABLE, gid=1).
class FullSWAEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        std::unique_ptr<BlockTree> tree       = std::make_unique<BlockTree>(2);
        auto                       full       = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
        swa->component_group_id               = 1;
        std::vector<ComponentGroupPtr> groups = {full, swa};
        cache_                                = block_tree_cache_test::makeBlockTreeCacheForTest(std::move(tree),
                                                                  std::move(groups),
                                                                  std::vector<Component>{},
                                                                  BlockTreeCacheConfig{.eviction_thread_pool_size = 2});
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_block, BlockIdxType swa_block) {
        std::vector<std::vector<GroupSlot>> slots(keys.size(), std::vector<GroupSlot>(2));
        for (size_t i = 0; i < keys.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(full_block + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(swa_block + i)};
        }
        cache_->insert(nullptr, keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full reclaim cascades to SWA on same node.
//
//   Before reclaimBlocksForTest(1, DEVICE):              After reclaim + wait:
//   root → [100] F:{10} S:{20}            root → [100] F:{10} S:{20}
//          → [200] F:{10} S:{20} ←leaf
//   Full heap: {[200]}  SWA heap: {[100],[200]}
//   Total device heap: 3
//
//   Full[200] reclaimed → cascade clears SWA[200] device.
//   Both REUSABLE groups empty → [200] deleted.
//   [100] survives, Full[100] becomes leaf → enters Full heap.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, FullReclaimCascadesToSWA) {
    insertPath({100, 200}, 10, 20);

    auto stats_before = cache_->getStats();
    EXPECT_EQ(stats_before.tree_node_count, 2u);
    EXPECT_EQ(stats_before.device_heap_total_size, 3u);  // 1 Full + 2 SWA

    int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100]
}

// ---------------------------------------------------------------------------
// Test: SWA-only cache — sequential reclaim drains chain.
//
//   SWA-only: root → [100] → [200] → [300]
//   SWA heap: {[100],[200],[300]}
//
//   LRU reclaims [100], then [200]. Both remain as empty internal nodes.
//   Reclaiming [300] deletes the leaf and prunes both empty ancestors.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, SWAOnlySequentialDrain) {
    std::unique_ptr<BlockTree> tree        = std::make_unique<BlockTree>(1);
    auto                       swa         = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 0;
    std::vector<ComponentGroupPtr>  groups = {swa};
    std::unique_ptr<BlockTreeCache> swa_cache =
        block_tree_cache_test::makeBlockTreeCacheForTest(std::move(tree),
                                                         std::move(groups),
                                                         std::vector<Component>{},
                                                         BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {20};
    slots[1][0].device_blocks = {21};
    slots[2][0].device_blocks = {22};
    swa_cache->insert(nullptr, {100, 200, 300}, slots);

    EXPECT_EQ(swa_cache->getStats().device_heap_total_size, 3u);

    // Reclaim [100]; it stays as an empty internal node.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*swa_cache, 1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 3u);
    EXPECT_EQ(swa_cache->getStats().device_heap_total_size, 2u);

    // Reclaim [200]; it also stays until its child is removed.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*swa_cache, 1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 3u);
    EXPECT_EQ(swa_cache->getStats().device_heap_total_size, 1u);

    // Reclaim [300] and prune [200] and [100].
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*swa_cache, 1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Full+SWA — sequential Full reclaim clears both via cascade.
//
//   Step 1: reclaim Full[200] → cascade SWA[200] → [200] deleted
//   Step 2: reclaim Full[100] → cascade SWA[100] → [100] deleted
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, SequentialFullReclaimClearsBothGroups) {
    insertPath({100, 200}, 10, 20);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Fork with Full+SWA — both branches have leaves.
//
//   root → [100] → [200] F:{10} S:{20} ← leaf
//                → [300] F:{40} S:{50} ← leaf
//   Full heap: {[200],[300]}  SWA heap: {[100],[200],[300]}
//
//   After reclaiming both leaves, [100] becomes Full leaf → 3rd reclaim needed.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, ForkBothBranchesEvictable) {
    insertPath({100, 200}, 10, 20);
    insertPath({100, 300}, 40, 50);

    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 5u);  // 2 Full + 3 SWA

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
