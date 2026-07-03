#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

namespace rtp_llm {
namespace {

// Helper: build BlockTreeCache with Full(REUSABLE, gid=0) + Linear(REUSABLE, gid=1).
class FullLinearEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(2);
        auto full                             = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        full->reuse_policy                    = CacheReusePolicy::REUSABLE;
        auto linear                           = std::make_shared<LinearComponentGroup>(CacheReusePolicy::REUSABLE);
        linear->component_group_id            = 1;
        std::vector<ComponentGroupPtr> groups = {full, linear};
        cache_                                = std::make_unique<BlockTreeCache>(
            std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, nullptr, 2);
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_block, BlockIdxType linear_block) {
        std::vector<GroupSlot> slots(2);
        slots[0].device_blocks = {full_block};
        slots[1].device_blocks = {linear_block};
        cache_->insert(keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full eviction cascades to Linear (Full > LINEAR priority).
//
//   Before evict(1, DEVICE):             After evict + wait:
//   root → [100] F:{10} L:{30}           root → [100] F:{10} L:{30}
//          → [200] F:{10} L:{30} ←leaf
//   Full heap: {[200]}  Linear heap: {[200]}
//   Total: 2
//
//   Evict Full[200] → cascade clears Linear[200] device.
//   [200] both groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullEvictionCascadesToLinear) {
    insertPath({100, 200}, 10, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 2u);  // 1 Full + 1 Linear (leaf only)

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Linear-only cache — sequential eviction drains chain.
//
//   Linear-only: root → [100] → [200] → [300]
//   Linear heap: {[300]} (insert-leaf only)
//
//   After evict [300]: [200] promoted (all groups promote parents).
//   After evict [200]: [100] promoted.
//   After evict [100]: empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, LinearOnlySequentialDrain) {
    auto tree                                = std::make_unique<BlockTree>(1);
    auto linear                              = std::make_shared<LinearComponentGroup>(CacheReusePolicy::REUSABLE);
    linear->component_group_id               = 0;
    std::vector<ComponentGroupPtr> groups    = {linear};
    auto                           lin_cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, nullptr, 2);

    std::vector<GroupSlot> slots(1);
    slots[0].device_blocks = {30};
    lin_cache->insert({100, 200, 300}, slots);

    EXPECT_EQ(lin_cache->getStats().device_heap_total_size, 1u);  // [300]

    lin_cache->evict(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 2u);

    lin_cache->evict(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 1u);

    lin_cache->evict(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Full+Linear NON_REUSABLE — Full eviction clears Full data,
//       Linear data survives but doesn't prevent node deletion.
//
//   Before:                               After Full evict + wait:
//   root → [100] F:{10} L:{30}            root → [100] F:{10} L:_
//          → [200] F:{10} L:{30} ←leaf           → [200] F:{10} L:_
//
//   Full heap: {[200]}, Linear heap: {[200]}
//   Evict Full[200] → cascade clears Linear[200] → both empty → deleted.
//   [100] survives, becomes Full leaf.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullEvictionWithNonReusableLinear) {
    auto tree                     = std::make_unique<BlockTree>(2);
    auto full                     = std::make_shared<FullComponentGroup>();
    full->component_group_id      = 0;
    full->reuse_policy            = CacheReusePolicy::REUSABLE;
    auto linear_nr                = std::make_shared<LinearComponentGroup>(CacheReusePolicy::NON_REUSABLE);
    linear_nr->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups   = {full, linear_nr};
    auto                           nr_cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), std::vector<Component>{}, nullptr, nullptr, 2);

    std::vector<GroupSlot> slots(2);
    slots[0].device_blocks = {10};
    slots[1].device_blocks = {30};
    nr_cache->insert({100, 200}, slots);

    // Linear NON_REUSABLE: no host/disk heaps
    EXPECT_EQ(nr_cache->componentGroups()[1]->host_heap, nullptr);
    EXPECT_EQ(nr_cache->componentGroups()[1]->disk_heap, nullptr);

    // Evict Full[200] → cascade Linear[200] → deleted
    nr_cache->evict(1, Tier::DEVICE);
    nr_cache->waitForPendingTasks();
    EXPECT_EQ(nr_cache->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Full eviction clears both Full+Linear on single node.
//
//   root → [100] F:{10} L:{30}
//   Full heap: {[100]}  Linear heap: {[100]}
//
//   Evict Full[100] → cascade Linear[100] → both empty → deleted.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullEvictionClearsBothGroupsSingleNode) {
    insertPath({100}, 10, 30);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Sequential Full eviction drains 2-node chain (Full+Linear).
//
//   Step 1: evict Full[200] → cascade Linear[200] → deleted
//   Step 2: evict Full[100] → cascade Linear[100] → deleted
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, SequentialFullEvictionDrainsChain) {
    insertPath({100, 200}, 10, 30);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
