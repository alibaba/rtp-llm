#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"

namespace rtp_llm {
namespace {

// Helper: BlockTreeCache with Full(gid=0) + SWA(gid=1) + Linear(gid=2), all REUSABLE.
class FullSWALinearEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        std::unique_ptr<BlockTree> tree       = std::make_unique<BlockTree>(3);
        auto                       full       = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        auto swa                              = std::make_shared<SWAComponentGroup>(128, 64);
        swa->component_group_id               = 1;
        auto linear                           = std::make_shared<LinearComponentGroup>();
        linear->component_group_id            = 2;
        std::vector<ComponentGroupPtr> groups = {full, swa, linear};
        cache_                                = BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                            std::move(groups),
                                                            std::vector<Component>{},
                                                            BlockTreeCacheConfig{.eviction_thread_pool_size = 2});
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_b, BlockIdxType swa_b, BlockIdxType lin_b) {
        std::vector<std::vector<GroupSlot>> slots(keys.size(), std::vector<GroupSlot>(3));
        for (size_t i = 0; i < keys.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(full_b + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(swa_b + i)};
            slots[i][2].device_blocks = {static_cast<BlockIdxType>(lin_b + i)};
        }
        cache_->insert(nullptr, keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full reclaim cascades to BOTH SWA and Linear.
//
//   Before reclaimBlocks(1, DEVICE):
//   root → [100] F:{10} S:{20} L:{30}
//          → [200] F:{10} S:{20} L:{30} ←leaf
//   Full heap: {[200]}  SWA heap: {[200]}  Linear heap: {[200]}
//   Total: 3
//
//   After reclaimBlocks(1, DEVICE) + wait:
//   root → [100] F:{10} S:{20} L:{30}
//
//   Full[200] reclaimed → cascade: SWA[200]+Linear[200] cleared.
//   [200] all 3 groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, FullReclaimCascadesToBothSWAAndLinear) {
    insertPath({100, 200}, 10, 20, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 3u);  // 1 Full + 1 SWA + 1 Linear (leaf)

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Single node with all 3 groups — Full reclaim clears all.
//
//   Before:                               After reclaimBlocks(1, DEVICE) + wait:
//   root → [100] F:{10} S:{20} L:{30}     root (empty tree)
//
//   Full[100] reclaimed → cascade SWA[100]+Linear[100] cleared.
//   All REUSABLE groups empty → deleted → empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SingleNodeAllGroupsCleared) {
    insertPath({100}, 10, 20, 30);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Sequential reclaim with 3 groups drains all.
//
//   root → [100] → [200] → [300], all with F+S+L data.
//   Only [300] in all heaps (leaf).
//
//   Step 1: reclaim Full[300] → cascade S+L → [300] deleted
//   Step 2: reclaim Full[200] → cascade S+L → [200] deleted
//   Step 3: reclaim Full[100] → cascade S+L → [100] deleted
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SequentialReclaimDrainsAllGroups) {
    insertPath({100, 200, 300}, 10, 20, 30);
    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Heap composition — only leaf in each heap.
//
//   root → [100] → [200] → [300] F:{10} S:{20} L:{30}
//
//   Full device heap:   {[300]} (1 leaf)
//   SWA device heap:    {[300]} (1 leaf)
//   Linear device heap: {[300]} (1 leaf)
//   Total: 3
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, HeapCompositionVerification) {
    insertPath({100, 200, 300}, 10, 20, 30);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 3u);
    EXPECT_EQ(stats.device_heap_total_size, 3u);  // 1 per group (leaf only)
}

// ---------------------------------------------------------------------------
// Test: Fork with 3 groups — both branches evictable.
//
//   root → [100] → [200] F:{10} S:{20} L:{30}  ← leaf
//                → [300] F:{40} S:{50} L:{60}  ← leaf
//
//   Full heap: {[200],[300]}  SWA heap: {[200],[300]}  Linear heap: {[200],[300]}
//   Total: 6
//
//   Sequential reclaim: 2 leaves + parent = 3 reclaims total.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, ForkBothBranchesEvictable) {
    insertPath({100, 200}, 10, 20, 30);
    insertPath({100, 300}, 40, 50, 60);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 3u);
    EXPECT_EQ(stats0.device_heap_total_size, 6u);  // 2 per group (2 leaves)

    // Reclaim first leaf
    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    // Reclaim second leaf
    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives

    // Reclaim [100] (now Full leaf)
    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: SWA → LINEAR cascade (design doc scenario C).
//
//   SWA+LINEAR only (no Full). SWA(gid=0), LINEAR(gid=1).
//
//   Before:                                   After SWA reclaim + wait:
//   root → [100] S:{20} L:{30}                root → [100] S:{20} L:_{30}
//          → [200] S:{20} L:{30} ←SWA leaf           → [200] S:_{20} L:_{30}
//   SWA heap: {[200]}  LINEAR heap: {[200]}
//
//   SWA[200] reclaimed → cascadeEviction: LINEAR below SWA → clears LINEAR[200].
//   [200] all groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SWAReclaimCascadesToLinear) {
    std::unique_ptr<BlockTree> tree        = std::make_unique<BlockTree>(2);
    auto                       swa         = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 0;
    auto linear                            = std::make_shared<LinearComponentGroup>();
    linear->component_group_id             = 1;
    std::vector<ComponentGroupPtr>  groups = {swa, linear};
    std::unique_ptr<BlockTreeCache> swa_lin_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {20};
    slots[0][1].device_blocks = {30};
    slots[1][0].device_blocks = {21};
    slots[1][1].device_blocks = {31};
    swa_lin_cache->insert(nullptr, {100, 200}, slots);

    EXPECT_EQ(swa_lin_cache->getStats().tree_node_count, 2u);
    // SWA heap: {[200]}, LINEAR heap: {[200]}
    EXPECT_EQ(swa_lin_cache->getStats().device_heap_total_size, 2u);

    // Reclaim SWA[200] → cascade clears LINEAR[200] → deleted
    swa_lin_cache->reclaimBlocks(1, Tier::DEVICE);
    swa_lin_cache->waitForPendingTasks();
    EXPECT_EQ(swa_lin_cache->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Multi-node ancestor chain cleanup (design doc section 2.7).
//
//   root → [100] → [200] → [300] → [400] F:{10} S:{20} L:{30}
//   Only [400] in heaps (leaf). All nodes have data from insert.
//
//   After reclaim [400]: cascade S+L → [400] deleted → [300] promoted
//   After reclaim [300]: cascade S+L → [300] deleted → [200] promoted
//   After reclaim [200]: cascade S+L → [200] deleted → [100] promoted
//   After reclaim [100]: cascade S+L → [100] deleted → empty tree
//
//   Verifies ancestor chain cleanup: each deleted node triggers parent promotion.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, AncestorChainCleanupDeepChain) {
    insertPath({100, 200, 300, 400}, 10, 20, 30);
    EXPECT_EQ(cache_->getStats().tree_node_count, 4u);

    // Sequential reclaim drains all 4 nodes
    for (int i = 4; i >= 1; --i) {
        cache_->reclaimBlocks(1, Tier::DEVICE);
        cache_->waitForPendingTasks();
        EXPECT_EQ(cache_->getStats().tree_node_count, static_cast<size_t>(i - 1))
            << "After reclaiming node " << (5 - i);
    }
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
