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
        auto linear                           = std::make_shared<LinearComponentGroup>();
        linear->component_group_id            = 1;
        std::vector<ComponentGroupPtr> groups = {full, linear};
        cache_ = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_block, BlockIdxType linear_block) {
        std::vector<std::vector<GroupSlot>> slots(keys.size(), std::vector<GroupSlot>(2));
        for (size_t i = 0; i < keys.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(full_block + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(linear_block + i)};
        }
        cache_->insert(nullptr, keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full reclaim cascades to Linear (Full > LINEAR priority).
//
//   Before reclaimBlocks(1, DEVICE):             After reclaim + wait:
//   root → [100] F:{10} L:{30}           root → [100] F:{10} L:{30}
//          → [200] F:{10} L:{30} ←leaf
//   Full heap: {[200]}  Linear heap: {[200]}
//   Total: 2
//
//   Reclaim Full[200] → cascade clears Linear[200] device.
//   [200] both groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullReclaimCascadesToLinear) {
    insertPath({100, 200}, 10, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 2u);  // 1 Full + 1 Linear (leaf only)

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Linear-only cache — sequential reclaim drains chain.
//
//   Linear-only: root → [100] → [200] → [300]
//   Linear heap: {[300]} (insert-leaf only)
//
//   After reclaim [300]: [200] promoted (all groups promote parents).
//   After reclaim [200]: [100] promoted.
//   After reclaim [100]: empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, LinearOnlySequentialDrain) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto linear                           = std::make_shared<LinearComponentGroup>();
    linear->component_group_id            = 0;
    std::vector<ComponentGroupPtr> groups = {linear};
    auto lin_cache = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {30};
    slots[1][0].device_blocks = {31};
    slots[2][0].device_blocks = {32};
    lin_cache->insert(nullptr, {100, 200, 300}, slots);

    EXPECT_EQ(lin_cache->getStats().device_heap_total_size, 1u);  // [300]

    lin_cache->reclaimBlocks(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 2u);

    lin_cache->reclaimBlocks(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 1u);

    lin_cache->reclaimBlocks(1, Tier::DEVICE);
    lin_cache->waitForPendingTasks();
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Full reclaim clears both Full+Linear on single node.
//
//   root → [100] F:{10} L:{30}
//   Full heap: {[100]}  Linear heap: {[100]}
//
//   Reclaim Full[100] → cascade Linear[100] → both empty → deleted.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullReclaimClearsBothGroupsSingleNode) {
    insertPath({100}, 10, 30);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Sequential Full reclaim drains 2-node chain (Full+Linear).
//
//   Step 1: reclaim Full[200] → cascade Linear[200] → deleted
//   Step 2: reclaim Full[100] → cascade Linear[100] → deleted
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, SequentialFullReclaimDrainsChain) {
    insertPath({100, 200}, 10, 30);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
