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
        full->reuse_policy                    = CacheReusePolicy::REUSABLE;
        auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
        swa->component_group_id               = 1;
        swa->reuse_policy                     = CacheReusePolicy::REUSABLE;
        std::vector<ComponentGroupPtr> groups = {full, swa};
        cache_ =
            std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, 2);
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_block, BlockIdxType swa_block) {
        std::vector<GroupSlot> slots(2);
        slots[0].device_blocks = {full_block};
        slots[1].device_blocks = {swa_block};
        cache_->insert(keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full eviction cascades to SWA on same node.
//
//   Before evict(1, DEVICE):              After evict + wait:
//   root → [100] F:{10} S:{20}            root → [100] F:{10} S:{20}
//          → [200] F:{10} S:{20} ←leaf
//   Full heap: {[200]}  SWA heap: {[200]}
//   Total device heap: 2
//
//   Full[200] evicted → cascade clears SWA[200] device.
//   Both REUSABLE groups empty → [200] deleted.
//   [100] survives, Full[100] becomes leaf → enters Full heap.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, FullEvictionCascadesToSWA) {
    insertPath({100, 200}, 10, 20);

    auto stats_before = cache_->getStats();
    EXPECT_EQ(stats_before.tree_node_count, 2u);
    EXPECT_EQ(stats_before.device_heap_total_size, 2u);  // 1 Full + 1 SWA

    int evicted = cache_->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 1);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100]
}

// ---------------------------------------------------------------------------
// Test: SWA-only cache — sequential eviction drains chain.
//
//   SWA-only: root → [100] → [200] → [300]
//   SWA heap: {[300]} (insert-leaf only)
//
//   After evict [300]: [200] promoted to heap (all groups now promote parents).
//   After evict [200]: [100] promoted.
//   After evict [100]: empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, SWAOnlySequentialDrain) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id               = 0;
    swa->reuse_policy                     = CacheReusePolicy::REUSABLE;
    std::vector<ComponentGroupPtr> groups = {swa};
    auto                           swa_cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, 2);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {20};
    swa_cache->insert({100, 200, 300}, slots);

    EXPECT_EQ(swa_cache->getStats().device_heap_total_size, 1u);  // [300]

    // Evict [300] → [200] promoted
    swa_cache->evict(1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 2u);

    // Evict [200] → [100] promoted
    swa_cache->evict(1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 1u);

    // Evict [100] → empty
    swa_cache->evict(1, Tier::DEVICE);
    swa_cache->waitForPendingTasks();
    EXPECT_EQ(swa_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Full+SWA — sequential Full eviction clears both via cascade.
//
//   Step 1: evict Full[200] → cascade SWA[200] → [200] deleted
//   Step 2: evict Full[100] → cascade SWA[100] → [100] deleted
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, SequentialFullEvictionClearsBothGroups) {
    insertPath({100, 200}, 10, 20);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->evict(1, Tier::DEVICE);
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
//   After evicting both leaves, [100] becomes Full leaf → 3rd evict needed.
// ---------------------------------------------------------------------------
TEST_F(FullSWAEvictionTest, ForkBothBranchesEvictable) {
    insertPath({100, 200}, 10, 20);
    insertPath({100, 300}, 40, 50);

    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 4u);  // 2 Full + 2 SWA

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
