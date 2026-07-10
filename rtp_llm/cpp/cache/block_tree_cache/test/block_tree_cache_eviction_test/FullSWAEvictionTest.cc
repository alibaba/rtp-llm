#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

namespace rtp_llm {
namespace {

// Helper: build a BlockTreeCache with Full(REUSABLE, gid=0) + SWA(REUSABLE, gid=1).
class FullSWAEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(2);
        auto full                             = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
        swa->component_group_id               = 1;
        std::vector<ComponentGroupPtr> groups = {full, swa};
        cache_ = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});
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
//   Before reclaimBlocks(1, DEVICE):              After reclaim + wait:
//   root → [100] F:{10} S:{20}            root → [100] F:{10} S:{20}
//          → [200] F:{10} S:{20} ←leaf
//   Full heap: {[200]}  SWA heap: {[200]}
//   Total device heap: 2
//
//   Full[200] reclaimed → cascade clears SWA[200] device.
//   Both REUSABLE groups empty → [200] deleted.
//   [100] survives, Full[100] becomes leaf → enters Full heap.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, FullReclaimCascadesToSWA) {
    insertPath({100, 200}, 10, 20);

    auto stats_before = cache_->getStats();
    EXPECT_EQ(stats_before.tree_node_count, 2u);
    EXPECT_EQ(stats_before.device_heap_total_size, 2u);  // 1 Full + 1 SWA

    int reclaimed = cache_->reclaimBlocks(1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100]
}

// ---------------------------------------------------------------------------
// Test: SWA-only cache — sequential reclaim drains chain.
//
//   SWA-only: root → [100] → [200] → [300]
//   SWA heap: {[300]} (insert-leaf only)
//
//   After reclaim [300]: [200] promoted to heap (all groups now promote parents).
//   After reclaim [200]: [100] promoted.
//   After reclaim [100]: empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, SWAOnlySequentialDrain) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id               = 0;
    std::vector<ComponentGroupPtr> groups = {swa};
    auto swa_cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {20};
    slots[1][0].device_blocks = {21};
    slots[2][0].device_blocks = {22};
    swa_cache->insert(nullptr, {100, 200, 300}, slots);

    EXPECT_EQ(swa_cache->getStats().device_heap_total_size, 1u);  // [300]

    // Reclaim [300] → [200] promoted
    swa_cache->reclaimBlocks(1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 2u);

    // Reclaim [200] → [100] promoted
    swa_cache->reclaimBlocks(1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 1u);

    // Reclaim [100] → empty
    swa_cache->reclaimBlocks(1, Tier::DEVICE);
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

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Fork with Full+SWA — both branches have leaves.
//
//   root → [100] → [200] F:{10} S:{20} ← leaf
//                → [300] F:{40} S:{50} ← leaf
//   Full heap: {[200],[300]}  SWA heap: {[200],[300]}
//
//   After reclaiming both leaves, [100] becomes Full leaf → 3rd reclaim needed.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, ForkBothBranchesEvictable) {
    insertPath({100, 200}, 10, 20);
    insertPath({100, 300}, 40, 50);

    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 4u);  // 2 Full + 2 SWA

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
