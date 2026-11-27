#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using block_tree_cache_test::BlockTreeCacheTestPeer;
using block_tree_cache_test::makeBlockTreeCacheForTest;

// Helper: build BlockTreeCache with Full(REUSABLE, group_set_id=0) + Linear(REUSABLE, group_set_id=1).
class FullLinearEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        auto linear = std::make_shared<LinearGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups = {full, linear};
        BlockTreeCacheConfig config{};
        config.task_pool_size = 2;
        cache_ = makeBlockTreeCacheForTest(std::move(groups), config);
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_block, BlockIdxType linear_block) {
        std::vector<std::vector<GroupSetResource>> resources(keys.size(), std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < keys.size(); ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(full_block + i)};
            resources[i][1].device_blocks = {static_cast<BlockIdxType>(linear_block + i)};
        }
        cache_->insert(keys, resources, Tier::DEVICE);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full drop cascades to Linear (Full > LINEAR priority).
//
//   Before dropBlocksForTest(1, DEVICE):             After drop + wait:
//   root → [100] F:{10} L:{30}           root → [100] F:{10} L:{30}
//          → [200] F:{10} L:{30} ←leaf
//   Full heap: {[200]}  Linear heap: {[100],[200]}
//   Total: 3
//
//   Drop Full[200] → cascade clears Linear[200] device.
//   [200] both groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullDropCascadesToLinear) {
    insertPath({100, 200}, 10, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 3u);  // 1 Full + 2 Linear

    BlockTreeCacheTestPeer::dropBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Linear-only cache — sequential drop drains chain.
//
//   Linear-only: root → [100] → [200] → [300]
//   Linear heap: {[100],[200],[300]}
//
//   LRU drops [100], then [200]. Both remain as empty internal nodes.
//   Dropping [300] deletes the leaf and prunes both empty ancestors.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, LinearOnlySequentialDrain) {
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr>        groups = {linear};
    BlockTreeCacheConfig config{};
    config.task_pool_size = 2;
    std::unique_ptr<BlockTreeCache> lin_cache = makeBlockTreeCacheForTest(std::move(groups), config);

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {30};
    resources[1][0].device_blocks = {31};
    resources[2][0].device_blocks = {32};
    lin_cache->insert({100, 200, 300}, resources, Tier::DEVICE);

    EXPECT_EQ(lin_cache->getStats().device_heap_total_size, 3u);

    BlockTreeCacheTestPeer::dropBlocksForTest(*lin_cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*lin_cache);
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 3u);
    EXPECT_EQ(lin_cache->getStats().device_heap_total_size, 2u);

    BlockTreeCacheTestPeer::dropBlocksForTest(*lin_cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*lin_cache);
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 3u);
    EXPECT_EQ(lin_cache->getStats().device_heap_total_size, 1u);

    BlockTreeCacheTestPeer::dropBlocksForTest(*lin_cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*lin_cache);
    EXPECT_EQ(lin_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Full drop clears both Full+Linear on single node.
//
//   root → [100] F:{10} L:{30}
//   Full heap: {[100]}  Linear heap: {[100]}
//
//   Drop Full[100] → cascade Linear[100] → both empty → deleted.
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, FullDropClearsBothGroupsSingleNode) {
    insertPath({100}, 10, 30);

    BlockTreeCacheTestPeer::dropBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Sequential Full drop drains 2-node chain (Full+Linear).
//
//   Step 1: drop Full[200] → cascade Linear[200] → deleted
//   Step 2: drop Full[100] → cascade Linear[100] → deleted
// ---------------------------------------------------------------------------
TEST_F(FullLinearEvictionTest, SequentialFullDropDrainsChain) {
    insertPath({100, 200}, 10, 30);

    BlockTreeCacheTestPeer::dropBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    BlockTreeCacheTestPeer::dropBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
