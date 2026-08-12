#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using block_tree_cache_test::BlockTreeCacheTestPeer;
using block_tree_cache_test::makeBlockTreeCacheForTest;

// Helper: BlockTreeCache with Full(group_set_id=0) + SWA(group_set_id=1) + Linear(group_set_id=2), all REUSABLE.
class FullSWALinearEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        auto swa = std::make_shared<SWAGroupSet>(
            128,
            64,
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            nullptr,
            nullptr);
        auto linear = std::make_shared<LinearGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups = {full, swa, linear};
        cache_ = makeBlockTreeCacheForTest(std::move(groups), BlockTreeCacheConfig{.task_pool_size = 2});
    }

    void insertPath(const CacheKeysType& keys, BlockIdxType full_b, BlockIdxType swa_b, BlockIdxType lin_b) {
        std::vector<std::vector<GroupSetResource>> resources(keys.size(), std::vector<GroupSetResource>(3));
        for (size_t i = 0; i < keys.size(); ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(full_b + i)};
            resources[i][1].device_blocks = {static_cast<BlockIdxType>(swa_b + i)};
            resources[i][2].device_blocks = {static_cast<BlockIdxType>(lin_b + i)};
        }
        cache_->insert(keys, resources, Tier::DEVICE);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

// ---------------------------------------------------------------------------
// Test: Full reclaim cascades to BOTH SWA and Linear.
//
//   Before reclaimBlocksForTest(1, DEVICE):
//   root → [100] F:{10} S:{20} L:{30}
//          → [200] F:{10} S:{20} L:{30} ←leaf
//   Full heap: {[200]}  SWA heap: {[100],[200]}  Linear heap: {[100],[200]}
//   Total: 5
//
//   After reclaimBlocksForTest(1, DEVICE) + wait:
//   root → [100] F:{10} S:{20} L:{30}
//
//   Full[200] reclaimed → cascade: SWA[200]+Linear[200] cleared.
//   [200] all 3 groups empty → deleted. [100] survives.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, FullReclaimCascadesToBothSWAAndLinear) {
    insertPath({100, 200}, 10, 20, 30);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 2u);
    EXPECT_EQ(stats0.device_heap_total_size, 5u);  // 1 Full + 2 SWA + 2 Linear

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives
}

// ---------------------------------------------------------------------------
// Test: Single node with all 3 groups — Full reclaim clears all.
//
//   Before:                               After reclaimBlocksForTest(1, DEVICE) + wait:
//   root → [100] F:{10} S:{20} L:{30}     root (empty tree)
//
//   Full[100] reclaimed → cascade SWA[100]+Linear[100] cleared.
//   All REUSABLE groups empty → deleted → empty tree.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SingleNodeAllGroupsCleared) {
    insertPath({100}, 10, 20, 30);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Sequential reclaim with 3 groups drains all.
//
//   root → [100] → [200] → [300], all with F+S+L data.
//   FULL contains only [300]; SWA and LINEAR contain every node.
//
//   Step 1: reclaim Full[300] → cascade S+L → [300] deleted
//   Step 2: reclaim Full[200] → cascade S+L → [200] deleted
//   Step 3: reclaim Full[100] → cascade S+L → [100] deleted
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SequentialReclaimDrainsAllGroups) {
    insertPath({100, 200, 300}, 10, 20, 30);
    EXPECT_EQ(cache_->getStats().tree_node_count, 3u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Heap composition — FULL is leaf-only; SWA/LINEAR include interior nodes.
//
//   root → [100] → [200] → [300] F:{10} S:{20} L:{30}
//
//   Full device heap:   {[300]}
//   SWA device heap:    {[100],[200],[300]}
//   Linear device heap: {[100],[200],[300]}
//   Total: 7
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, HeapCompositionVerification) {
    insertPath({100, 200, 300}, 10, 20, 30);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 3u);
    EXPECT_EQ(stats.device_heap_total_size, 7u);  // 1 Full + 3 SWA + 3 Linear
}

// ---------------------------------------------------------------------------
// Test: Fork with 3 groups — both branches evictable.
//
//   root → [100] → [200] F:{10} S:{20} L:{30}  ← leaf
//                → [300] F:{40} S:{50} L:{60}  ← leaf
//
//   Full heap: {[200],[300]}
//   SWA heap: {[100],[200],[300]}  Linear heap: {[100],[200],[300]}
//   Total: 8
//
//   Sequential reclaim: 2 leaves + parent = 3 reclaims total.
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, ForkBothBranchesEvictable) {
    insertPath({100, 200}, 10, 20, 30);
    insertPath({100, 300}, 40, 50, 60);

    auto stats0 = cache_->getStats();
    EXPECT_EQ(stats0.tree_node_count, 3u);
    EXPECT_EQ(stats0.device_heap_total_size, 8u);  // 2 Full + 3 SWA + 3 Linear

    // Reclaim first leaf
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    // Reclaim second leaf
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives

    // Reclaim [100] (now Full leaf)
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: SWA → LINEAR cascade (design doc scenario C).
//
//   SWA+LINEAR only (no Full). SWA(group_set_id=0), LINEAR(group_set_id=1).
//
//   Before:
//   root → [100] S:{20} L:{30}
//          → [200] S:{21} L:{31}
//   SWA heap: {[100],[200]}  LINEAR heap: {[100],[200]}
//
//   LRU first reclaims SWA[100] and cascades LINEAR[100]; [100] remains because it has a child.
//   Reclaiming SWA[200] cascades LINEAR[200], deletes [200], then prunes empty [100].
// ---------------------------------------------------------------------------
TEST_F(FullSWALinearEvictionTest, SWAReclaimCascadesToLinear) {
    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr>        groups = {swa, linear};
    std::unique_ptr<BlockTreeCache> swa_lin_cache =
        makeBlockTreeCacheForTest(std::move(groups), BlockTreeCacheConfig{.task_pool_size = 2});

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {20};
    resources[0][1].device_blocks = {30};
    resources[1][0].device_blocks = {21};
    resources[1][1].device_blocks = {31};
    swa_lin_cache->insert({100, 200}, resources, Tier::DEVICE);

    EXPECT_EQ(swa_lin_cache->getStats().tree_node_count, 2u);
    EXPECT_EQ(swa_lin_cache->getStats().device_heap_total_size, 4u);

    // Reclaim SWA[100] first (LRU); the empty internal node remains.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*swa_lin_cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*swa_lin_cache);
    EXPECT_EQ(swa_lin_cache->getStats().tree_node_count, 2u);
    EXPECT_EQ(swa_lin_cache->getStats().device_heap_total_size, 2u);

    // Reclaim SWA[200], then delete the leaf and prune empty [100].
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*swa_lin_cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*swa_lin_cache);
    EXPECT_EQ(swa_lin_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: Multi-node ancestor chain cleanup (design doc section 2.7).
//
//   root → [100] → [200] → [300] → [400] F:{10} S:{20} L:{30}
//   FULL contains only [400]; SWA/LINEAR contain all four nodes.
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
        BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
        EXPECT_EQ(cache_->getStats().tree_node_count, static_cast<size_t>(i - 1))
            << "After reclaiming node " << (5 - i);
    }
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

}  // namespace
}  // namespace rtp_llm
