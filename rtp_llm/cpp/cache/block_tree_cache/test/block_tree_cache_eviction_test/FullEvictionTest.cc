#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

namespace rtp_llm {
namespace {

// Helper: build a BlockTreeCache with a single Full(REUSABLE) group.
class FullEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full                             = std::make_shared<FullComponentGroup>();
        full->component_group_id              = 0;
        std::vector<ComponentGroupPtr> groups = {full};
        cache_ = std::make_unique<BlockTreeCache>(std::move(tree), std::move(groups), std::vector<Component>{}, BlockTreeCacheConfig{.eviction_thread_pool_size = 2});
    }

    // Insert a path with given device block for group 0.
    void insertPath(const CacheKeysType& keys, BlockIdxType dev_block) {
        std::vector<std::vector<GroupSlot>> slots(keys.size(), std::vector<GroupSlot>(1));
        for (size_t i = 0; i < keys.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(dev_block + i)};
        }
        cache_->insert(nullptr, keys, slots);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Only leaf nodes enter the Full device heap.
//
//   Insert: root → [100] D={10} → [200] D={10} → [300] D={10}
//
//   Only [300] is the insert-leaf → enters heap.
//   [100] and [200] are intermediate → NOT in heap.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, OnlyLeafEntersDeviceHeap) {
    insertPath({100, 200, 300}, 10);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 3u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);  // Only insert-leaf [300]
}

// ---------------------------------------------------------------------------
// Test: Evict single leaf — async, node deleted, parent becomes leaf.
//
//   Before evict(DEVICE):                After evict(1) + wait:
//   root → [100] → [200] → [300] ←heap   root → [100] → [200] ←new leaf, in heap
//
//   [300] evicted async: D cleared → empty → deleted.
//   [200] becomes leaf → tryAddToDeviceHeap → enters heap.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, EvictSingleLeafDeletesNodeAndPromotesParent) {
    insertPath({100, 200, 300}, 10);

    int evicted = cache_->evict(1, Tier::DEVICE);
    EXPECT_EQ(evicted, 1);
    cache_->waitForPendingTasks();

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);         // [100], [200] remain
    EXPECT_EQ(stats.device_heap_total_size, 1u);  // [200] is now the leaf
}

// ---------------------------------------------------------------------------
// Test: Parent becomes leaf after child eviction.
//
//   Before:                              After evict(1) + wait:
//   root → [100] → [200] → [300] ←heap   root → [100] → [200] ←heap
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, ParentBecomesLeafAfterChildEviction) {
    insertPath({100, 200, 300}, 10);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);
}

// ---------------------------------------------------------------------------
// Test: Sequential eviction drains a 3-node chain.
//
//   Step 0: root → [100] → [200] → [300]  heap: {[300]}
//   Step 1: evict [300] → deleted          heap: {[200]}
//   Step 2: evict [200] → deleted          heap: {[100]}
//   Step 3: evict [100] → deleted          heap: {}
//   Final:  empty tree
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, SequentialEvictionDrainsChain) {
    insertPath({100, 200, 300}, 10);

    // Step 1: evict [300]
    EXPECT_EQ(cache_->evict(1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    // Step 2: evict [200]
    EXPECT_EQ(cache_->evict(1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    // Step 3: evict [100]
    EXPECT_EQ(cache_->evict(1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);

    // No more to evict
    EXPECT_EQ(cache_->evict(1, Tier::DEVICE), 0);
}

// ---------------------------------------------------------------------------
// Test: Fork — two branches, both leaves in heap.
//
//   root → [100] → [200] D={10} ← leaf, in heap
//                → [300] D={20} ← leaf, in heap
//
//   Both [200] and [300] are insert-leaves → both in heap.
//   After evicting both leaves, [100] becomes leaf with data → 3rd evict needed.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, ForkBothLeavesEvictable) {
    insertPath({100, 200}, 10);
    insertPath({100, 300}, 20);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 3u);
    EXPECT_EQ(stats.device_heap_total_size, 2u);  // [200] and [300]

    // Evict first leaf
    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);  // [100] + one leaf

    // Evict second leaf
    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives (has data)

    // Evict [100] (now leaf after both children deleted)
    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Evict empty tree returns 0.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, EvictEmptyTreeReturnsZero) {
    EXPECT_EQ(cache_->evict(1, Tier::DEVICE), 0);
}

// ---------------------------------------------------------------------------
// Test: LRU ordering — oldest leaf evicted first.
//
//   Insert [100] D={10}, then [200] D={20}.
//   Both are leaves (separate roots). LRU: evict [100] first.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, LRUEvictsOldestLeafFirst) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    EXPECT_EQ(cache_->getStats().device_heap_total_size, 2u);

    cache_->evict(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    // [100] was evicted (oldest). Only [200] remains.
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    auto result = cache_->match({200});
    EXPECT_EQ(result.matched_blocks, 1u);
}

}  // namespace
}  // namespace rtp_llm
