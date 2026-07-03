#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

namespace rtp_llm {
namespace {

// Helper: BlockTreeCache with Full(gid=0, REUSABLE) + SWA(gid=1, REUSABLE)
//         + SWA(gid=2, NON_REUSABLE).
class FullSWANonReusableEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(3);
        auto full                             = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        full->reuse_policy                    = CacheReusePolicy::REUSABLE;
        auto swa_r                            = std::make_shared<SWAComponentGroup>(128, 64);
        swa_r->component_group_id             = 1;
        swa_r->reuse_policy                   = CacheReusePolicy::REUSABLE;
        auto swa_nr                           = std::make_shared<SWAComponentGroup>(64, 32);
        swa_nr->component_group_id            = 2;
        swa_nr->reuse_policy                  = CacheReusePolicy::NON_REUSABLE;
        std::vector<ComponentGroupPtr> groups = {full, swa_r, swa_nr};
        cache_ =
            std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, 2);
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_b, BlockIdxType swa_r_b, BlockIdxType swa_nr_b) {
        std::vector<GroupSlot> slots(3);
        slots[0].device_blocks = {full_b};
        slots[1].device_blocks = {swa_r_b};
        slots[2].device_blocks = {swa_nr_b};
        cache_->insert(keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full eviction cascades to both SWA(R) and SWA(NR).
//
//   Before evict(1, DEVICE):
//   root → [100] F:{10} SR:{20} SNR:{30}
//          → [200] F:{10} SR:{20} SNR:{30} ←leaf
//   Full heap: {[200]}  SWA(R) heap: {[200]}  SWA(NR) heap: {[200]}
//   Total: 3
//
//   After evict + wait:
//   root → [100] F:{10} SR:{20} SNR:{30}
//
//   Full[200] evicted → cascade: SWA(R)[200]+SWA(NR)[200] cleared.
//   [200] all groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, FullCascadeClearsBothReusableAndNonReusable) {
    insertPath({100, 200}, 10, 20, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 3u);  // 1 Full + 1 SWA(R) + 1 SWA(NR)

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);
}

// ---------------------------------------------------------------------------
// Test: Single node — Full eviction clears all groups, node deleted.
//
//   Before:                                After evict + wait:
//   root → [100] F:{10} SR:{20} SNR:{30}   root (empty tree)
//
//   Full[100] evicted → cascade SWA(R)+SWA(NR) cleared.
//   All REUSABLE groups (Full, SWA_R) empty → deleted.
//   SWA(NR) data ignored by shouldDeleteNode.
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, SingleNodeAllGroupsCleared) {
    insertPath({100}, 10, 20, 30);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: NON_REUSABLE SWA has no host/disk heaps.
//   tryAddToHostHeap for NON_REUSABLE group is a no-op.
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, NonReusableSWANoHostDiskHeaps) {
    auto& swa_nr = cache_->componentGroups()[2];
    EXPECT_EQ(swa_nr->reuse_policy, CacheReusePolicy::NON_REUSABLE);

    insertPath({100}, 10, 20, 30);

    auto result = cache_->tree()->findNode({100});
    ASSERT_NE(result.matched_node, nullptr);

    // tryAddToHostHeap for NON_REUSABLE group should be no-op
    swa_nr->tryAddToHostHeap(result.matched_node);
    auto& slot = result.matched_node->group_slots[2];
    EXPECT_FALSE(slot.in_host_heap);
}

// ---------------------------------------------------------------------------
// Test: Sequential eviction with mixed REUSABLE/NON_REUSABLE.
//
//   root → [100] → [200], all with F:{10} SR:{20} SNR:{30}
//
//   Step 1: evict Full[200] → cascade SR+SNR → [200] deleted
//   Step 2: evict Full[100] → cascade SR+SNR → [100] deleted
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, SequentialEvictionMixedReuse) {
    insertPath({100, 200}, 10, 20, 30);
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Heap sizes for mixed REUSABLE/NON_REUSABLE.
//
//   root → [100] → [200] F:{10} SR:{20} SNR:{30}
//
//   Full heap: {[200]}  SWA(R) heap: {[200]}  SWA(NR) heap: {[200]}
//   Total: 3 (leaf only per group)
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, HeapSizesMixedReuse) {
    insertPath({100, 200}, 10, 20, 30);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 3u);  // 1 per group (leaf only)
}

// ---------------------------------------------------------------------------
// Test: SWA(NR)-only cache — synchronous eviction drains chain.
//
//   root → [100] → [200] SNR:{30}
//   SWA(NR) heap: {[200]} (leaf only)
//
//   After Bug 1 fix: shouldDeleteNode only checks REUSABLE groups.
//   Since all groups are NON_REUSABLE, nodes are always deletable.
//   Evict [200] synchronous → deleted. [100] also deleted by removeEmptyAncestors
//   (no REUSABLE data to keep it alive).
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, SWANonReusableOnlyCacheSynchronous) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto swa_nr                           = std::make_shared<SWAComponentGroup>(64, 32);
    swa_nr->component_group_id            = 0;
    swa_nr->reuse_policy                  = CacheReusePolicy::NON_REUSABLE;
    std::vector<ComponentGroupPtr> groups = {swa_nr};
    auto                           nr_cache =
        std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, 2);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {30};
    nr_cache->insert({100, 200}, slots);

    EXPECT_EQ(nr_cache->getStats().tree_node_count, 2u);
    EXPECT_EQ(nr_cache->getStats().device_heap_total_size, 1u);  // [200]

    // Evict [200] (synchronous, NON_REUSABLE → target=NONE)
    // After Bug 1 fix: [200] deleted, [100] also deleted (no REUSABLE data)
    nr_cache->evict(1, Tier::DEVICE);
    EXPECT_EQ(nr_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: NON_REUSABLE cascade direct release (design doc scenario A).
//
//   Full(gid=0, REUSABLE) + SWA(gid=1, REUSABLE) + SWA(gid=2, NON_REUSABLE).
//   Single node: root → [100] F:{10} SR:{20} SNR:{30}
//
//   Full[100] evicted (REUSABLE, async D2H) → cascade:
//     - SWA(R)[100]: device cleared (cascaded)
//     - SWA(NR)[100]: device cleared (cascaded, NON_REUSABLE → direct release)
//   All groups empty → [100] deleted → empty tree.
//   NON_REUSABLE data is lost when Full cascade clears it.
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, NonReusableCascadeDirectRelease) {
    insertPath({100}, 10, 20, 30);

    // Verify all 3 groups have data
    auto result = cache_->tree()->findNode({100});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_TRUE(result.matched_node->group_slots[0].has_device_value());  // Full
    EXPECT_TRUE(result.matched_node->group_slots[1].has_device_value());  // SWA(R)
    EXPECT_TRUE(result.matched_node->group_slots[2].has_device_value());  // SWA(NR)

    // Full evict → cascade clears ALL lower groups on same node
    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    // Node deleted: all groups empty (including NON_REUSABLE)
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Ancestor chain cleanup with NON_REUSABLE groups.
//
//   root → [100] → [200] → [300] F:{10} SR:{20} SNR:{30}
//
//   Evict [300] → cascade → deleted. [200] promoted.
//   Evict [200] → cascade → deleted. [100] promoted.
//   Evict [100] → cascade → deleted. Empty tree.
//   Verifies ancestor cleanup works with mixed REUSABLE/NON_REUSABLE.
// ---------------------------------------------------------------------------
TEST_F(FullSWANonReusableEvictionTest, AncestorChainCleanupMixedReuse) {
    insertPath({100, 200, 300}, 10, 20, 30);
    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
