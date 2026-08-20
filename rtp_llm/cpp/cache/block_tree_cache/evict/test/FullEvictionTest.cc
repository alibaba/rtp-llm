#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using block_tree_cache_test::BlockTreeCacheTestPeer;
using block_tree_cache_test::makeBlockTreeCacheForTest;

// Helper: build a BlockTreeCache with a single Full(REUSABLE) group.
class FullEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups = {full};
        cache_ = makeBlockTreeCacheForTest(std::move(groups), BlockTreeCacheConfig{.task_pool_size = 2});
    }

    // Insert a path with given device block for group 0.
    void insertPath(const CacheKeysType& keys, BlockIdxType dev_block) {
        std::vector<std::vector<GroupSetResource>> resources(keys.size(), std::vector<GroupSetResource>(1));
        for (size_t i = 0; i < keys.size(); ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(dev_block + i)};
        }
        cache_->insert(keys, resources, Tier::DEVICE);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

struct TieredFullCache {
    DeviceBlockPoolPtr                      device_pool;
    std::shared_ptr<HostBlockPool>          host_pool;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool;
    std::shared_ptr<FullGroupSet>           full;
    std::unique_ptr<BlockTreeCache>         cache;
};

TieredFullCache makeTieredFullCache(bool enable_lower_tiers = true) {
    TieredFullCache result;
    result.device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    result.host_pool   = block_tree_cache_test::makeHostPool(1, 16);
    result.disk_pool =
        block_tree_cache_test::makeDiskPool(1, 16, std::make_unique<block_tree_cache_test::MemoryDiskBlockIO>());
    result.full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{result.device_pool}, result.host_pool, result.disk_pool);

    BlockTreeCacheConfig config;
    config.enable_host_cache = enable_lower_tiers;
    config.enable_disk_cache = enable_lower_tiers;
    std::vector<GroupSetPtr> group_sets{result.full};
    result.cache = makeBlockTreeCacheForTest(std::move(group_sets), config);
    return result;
}

std::vector<TreeNode*> insertFullSandwich(TieredFullCache& environment) {
    const BlockIdxType host_block = environment.full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block = environment.full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    RTP_LLM_CHECK(!isNullBlockIdx(host_block));
    RTP_LLM_CHECK(!isNullBlockIdx(disk_block));

    std::vector<std::vector<GroupSetResource>> resources(5, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = host_block;
    resources[3][0].device_blocks = {12};
    resources[4][0].disk_slot     = disk_block;
    RTP_LLM_CHECK(
        block_tree_cache_test::insertGroupSetResources(*environment.cache, {100, 200, 300, 400, 500}, resources));
    return environment.cache->tree()->findNode({100, 200, 300, 400, 500});
}

std::vector<TreeNode*> insertFullLowerTierDescendants(TieredFullCache& environment) {
    const BlockIdxType host_block = environment.full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block = environment.full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    RTP_LLM_CHECK(!isNullBlockIdx(host_block));
    RTP_LLM_CHECK(!isNullBlockIdx(disk_block));

    std::vector<std::vector<GroupSetResource>> resources(4, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = host_block;
    resources[3][0].disk_slot     = disk_block;
    RTP_LLM_CHECK(block_tree_cache_test::insertGroupSetResources(*environment.cache, {100, 200, 300, 400}, resources));
    return environment.cache->tree()->findNode({100, 200, 300, 400});
}

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

// Extending an existing FULL leaf creates only the suffix node in inserted_nodes.
// The old leaf must be refreshed separately and removed from the FULL heap.
TEST_F(FullEvictionTest, ExtendingExistingLeafRefreshesDirectParent) {
    insertPath({100, 200, 300, 400}, 10);
    ASSERT_EQ(cache_->getStats().device_heap_total_size, 1u);

    const auto before = cache_->tree()->findNode({100, 200, 300, 400});
    ASSERT_EQ(before.size(), 4u);
    std::vector<CandidateMeta> ancestor_meta_before;
    for (size_t index = 0; index + 1 < before.size(); ++index) {
        ancestor_meta_before.push_back(before[index]->group_set_resources[0].candidate_meta);
    }
    const CandidateMeta direct_parent_meta_before = before.back()->group_set_resources[0].candidate_meta;
    insertPath({100, 200, 300, 400, 500}, 20);

    EXPECT_EQ(cache_->getStats().tree_node_count, 5u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 1u);  // only the new [500] leaf
    const auto after = cache_->tree()->findNode({100, 200, 300, 400, 500});
    ASSERT_EQ(after.size(), 5u);

    EXPECT_EQ(after[3]->group_set_resources[0].candidate_meta.last_access_seq,
              direct_parent_meta_before.last_access_seq);
    EXPECT_EQ(after[3]->group_set_resources[0].candidate_meta.admission_seq, direct_parent_meta_before.admission_seq);
    EXPECT_EQ(after[3]->group_set_resources[0].candidate_meta.hit_count, direct_parent_meta_before.hit_count);
    for (size_t index = 0; index < ancestor_meta_before.size(); ++index) {
        const CandidateMeta& after_meta = after[index]->group_set_resources[0].candidate_meta;
        EXPECT_EQ(after_meta.last_access_seq, ancestor_meta_before[index].last_access_seq) << "ancestor=" << index;
        EXPECT_EQ(after_meta.admission_seq, ancestor_meta_before[index].admission_seq) << "ancestor=" << index;
        EXPECT_EQ(after_meta.hit_count, ancestor_meta_before[index].hit_count) << "ancestor=" << index;
    }
}

TEST(FullEvictionRegressionTest, ExtendingExistingLeafRefreshesOnlyFullParent) {
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(1)}, nullptr, nullptr);
    auto cache = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full, swa});
    ASSERT_NE(cache, nullptr);

    std::vector<std::vector<GroupSetResource>> initial_resources(4, std::vector<GroupSetResource>(2));
    for (size_t index = 0; index < initial_resources.size(); ++index) {
        initial_resources[index][0].device_blocks = {static_cast<BlockIdxType>(10 + index)};
        initial_resources[index][1].device_blocks = {static_cast<BlockIdxType>(20 + index)};
    }
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300, 400}, initial_resources));

    const auto before = cache->tree()->findNode({100, 200, 300, 400});
    ASSERT_EQ(before.size(), 4u);
    EvictionHeap* swa_heap = cache->evictor_.heapFor(/*group_set_id=*/1, Tier::DEVICE);
    ASSERT_NE(swa_heap, nullptr);
    for (TreeNode* node : before) {
        swa_heap->erase(node);
    }
    ASSERT_EQ(cache->evictor_.candidateCount(/*group_set_id=*/1, Tier::DEVICE), 0u);

    std::vector<std::vector<GroupSetResource>> extended_resources(5, std::vector<GroupSetResource>(2));
    for (size_t index = 0; index < extended_resources.size(); ++index) {
        extended_resources[index][0].device_blocks = {static_cast<BlockIdxType>(30 + index)};
        extended_resources[index][1].device_blocks = {static_cast<BlockIdxType>(40 + index)};
    }
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300, 400, 500}, extended_resources));

    const auto after = cache->tree()->findNode({100, 200, 300, 400, 500});
    ASSERT_EQ(after.size(), 5u);
    EXPECT_EQ(cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    EXPECT_EQ(cache->evictor_.candidateCount(/*group_set_id=*/1, Tier::DEVICE), 1u);
    for (size_t index = 0; index < 4; ++index) {
        EXPECT_FALSE(swa_heap->contains(after[index])) << "ancestor=" << index << " must not be re-scanned";
    }
    EXPECT_TRUE(swa_heap->contains(after[4]));
}

// ---------------------------------------------------------------------------
// Test: Reclaim single leaf — node deleted, parent becomes leaf.
//
//   Before reclaimBlocksForTest(DEVICE):                After reclaimBlocksForTest(1) + wait:
//   root → [100] → [200] → [300] ←heap   root → [100] → [200] ←new leaf, in heap
//
//   [300] reclaimed: D cleared → empty → deleted.
//   [200] becomes leaf -> refreshCandidate re-admits it as a device candidate.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, ReclaimSingleLeafDeletesNodeAndPromotesParent) {
    insertPath({100, 200, 300}, 10);

    int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);         // [100], [200] remain
    EXPECT_EQ(stats.device_heap_total_size, 1u);  // [200] is now the leaf
}

// ---------------------------------------------------------------------------
// Test: Parent becomes leaf after child reclaim.
//
//   Before:                              After reclaimBlocksForTest(1) + wait:
//   root → [100] → [200] → [300] ←heap   root → [100] → [200] ←heap
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, ParentBecomesLeafAfterChildEviction) {
    insertPath({100, 200, 300}, 10);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);
}

// ---------------------------------------------------------------------------
// Test: Sequential reclaim drains a 3-node chain.
//
//   Step 0: root → [100] → [200] → [300]  heap: {[300]}
//   Step 1: reclaim [300] → deleted        heap: {[200]}
//   Step 2: reclaim [200] → deleted        heap: {[100]}
//   Step 3: reclaim [100] → deleted        heap: {}
//   Final:  empty tree
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, SequentialReclaimDrainsChain) {
    insertPath({100, 200, 300}, 10);

    // Step 1: reclaim [300]
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    // Step 2: reclaim [200]
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    // Step 3: reclaim [100]
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);

    // No more to reclaim
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 0);
}

// ---------------------------------------------------------------------------
// Test: Fork — two branches, both leaves in heap.
//
//   root → [100] → [200] D={10} ← leaf, in heap
//                → [300] D={20} ← leaf, in heap
//
//   Both [200] and [300] are insert-leaves → both in heap.
//   After reclaiming both leaves, [100] becomes leaf with data → 3rd reclaim needed.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, ForkBothLeavesEvictable) {
    insertPath({100, 200}, 10);
    insertPath({100, 300}, 20);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 3u);
    EXPECT_EQ(stats.device_heap_total_size, 2u);  // [200] and [300]

    // Reclaim first leaf
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);  // [100] + one leaf

    // Reclaim second leaf
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives (has data)

    // Reclaim [100] (now leaf after both children deleted)
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Test: LRU ordering — oldest leaf reclaimed first.
//
//   Insert [100] D={10}, then [200] D={20}.
//   Both are leaves (separate roots). LRU: reclaim [100] first.
// ---------------------------------------------------------------------------
TEST_F(FullEvictionTest, LRUReclaimsOldestLeafFirst) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    EXPECT_EQ(cache_->getStats().device_heap_total_size, 2u);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    // [100] was reclaimed (oldest). Only [200] remains.
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    auto result = cache_->match({200});
    EXPECT_EQ(result.matched_device_blocks, 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
}

// A real match is the only event that advances LRU heat. Matching the oldest
// leaf makes it newer than a leaf inserted later, so the latter is reclaimed.
TEST_F(FullEvictionTest, MatchRefreshesLruOrder) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    auto match = cache_->match({100});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, match.matched_device_resources);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    auto hot_match  = cache_->match({100});
    auto cold_match = cache_->match({200});
    EXPECT_EQ(hot_match.matched_device_blocks, 1u);
    EXPECT_EQ(cold_match.matched_device_blocks, 0u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, hot_match.matched_device_resources);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, cold_match.matched_device_resources);
}

// Releasing match references must not count as another access or change the
// ordering metadata.
TEST_F(FullEvictionTest, MatchReleaseDoesNotMutateHeat) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    auto match = cache_->match({100});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    auto found = cache_->tree()->findNode({100});
    ASSERT_FALSE(found.empty());
    const CandidateMeta meta_after_match = found.back()->group_set_resources[0].candidate_meta;
    ASSERT_EQ(meta_after_match.hit_count, 1u);

    block_tree_cache_test::releaseRequestRefsForTest(*cache_, match.matched_device_resources);

    const CandidateMeta meta_after_release = found.back()->group_set_resources[0].candidate_meta;
    EXPECT_EQ(meta_after_release.last_access_seq, meta_after_match.last_access_seq);
    EXPECT_EQ(meta_after_release.admission_seq, meta_after_match.admission_seq);
    EXPECT_EQ(meta_after_release.hit_count, meta_after_match.hit_count);

    // Releasing the hotter node must not refresh it again: the untouched rival
    // remains the next LRU victim.
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    auto hot_match  = cache_->match({100});
    auto cold_match = cache_->match({200});
    EXPECT_EQ(hot_match.matched_device_blocks, 1u);
    EXPECT_EQ(cold_match.matched_device_blocks, 0u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, hot_match.matched_device_resources);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, cold_match.matched_device_resources);
}

// An insert that completely overlaps an existing path creates no inserted_nodes.
// It must neither overwrite the existing block nor make that node artificially hot.
TEST_F(FullEvictionTest, OverlappingInsertDoesNotOverwriteOrRefreshLru) {
    insertPath({100}, 10);
    insertPath({200}, 20);
    insertPath({100}, 99);

    auto before = cache_->tree()->findNode({100});
    ASSERT_FALSE(before.empty());
    ASSERT_EQ(before.back()->group_set_resources[0].device_blocks, std::vector<BlockIdxType>({10}));

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);

    auto old_match = cache_->match({100});
    auto new_match = cache_->match({200});
    EXPECT_EQ(old_match.matched_device_blocks, 0u);
    EXPECT_EQ(new_match.matched_device_blocks, 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, old_match.matched_device_resources);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, new_match.matched_device_resources);
}

TEST(FullPruneTest, PrunesDependentFullSubtreeAcrossTiers) {
    TieredFullCache environment        = makeTieredFullCache();
    const size_t    device_free_before = environment.device_pool->freeBlocksNum();
    const size_t    host_free_before   = environment.host_pool->freeBlocksNum();
    const size_t    disk_free_before   = environment.disk_pool->freeBlocksNum();
    const auto      path               = insertFullSandwich(environment);
    ASSERT_EQ(path.size(), 5u);

    const BlockIdxType host_block = path[2]->group_set_resources[0].host_block;
    const BlockIdxType disk_block = path[4]->group_set_resources[0].disk_slot;
    ASSERT_EQ(environment.device_pool->freeBlocksNum(), device_free_before);
    ASSERT_EQ(environment.host_pool->freeBlocksNum(), host_free_before - 1);
    ASSERT_EQ(environment.disk_pool->freeBlocksNum(), disk_free_before - 1);

    EXPECT_EQ(environment.cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 2);

    const auto remaining_path = environment.cache->tree()->findNode({100, 200, 300, 400, 500});
    ASSERT_EQ(remaining_path.size(), 1u);
    EXPECT_EQ(remaining_path.front(), path.front());
    EXPECT_EQ(environment.cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(environment.cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment.cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(environment.cache->getStats().disk_heap_total_size, 0u);
    EXPECT_EQ(environment.device_pool->freeBlocksNum(), device_free_before + 2);
    EXPECT_EQ(environment.host_pool->freeBlocksNum(), host_free_before);
    EXPECT_EQ(environment.disk_pool->freeBlocksNum(), disk_free_before);
    EXPECT_FALSE(environment.host_pool->isAllocated(host_block));
    EXPECT_FALSE(environment.disk_pool->isAllocated(disk_block));
}

TEST(FullPruneTest, PrunesBranchedFullSubtreeBottomUp) {
    TieredFullCache    environment      = makeTieredFullCache();
    const size_t       host_free_before = environment.host_pool->freeBlocksNum();
    const BlockIdxType left_host_block  = environment.full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType right_host_block = environment.full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(left_host_block));
    ASSERT_FALSE(isNullBlockIdx(right_host_block));

    std::vector<std::vector<GroupSetResource>> left_resources(3, std::vector<GroupSetResource>(1));
    left_resources[0][0].device_blocks = {10};
    left_resources[1][0].device_blocks = {11};
    left_resources[2][0].host_block    = left_host_block;
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*environment.cache, {100, 200, 300}, left_resources));

    std::vector<std::vector<GroupSetResource>> right_resources(3, std::vector<GroupSetResource>(1));
    right_resources[0][0].device_blocks = {10};
    right_resources[1][0].device_blocks = {11};
    right_resources[2][0].host_block    = right_host_block;
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*environment.cache, {100, 200, 400}, right_resources));

    EXPECT_EQ(environment.cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);

    EXPECT_EQ(environment.cache->tree()->findNode({100, 200, 300}).size(), 1u);
    EXPECT_EQ(environment.cache->tree()->findNode({100, 200, 400}).size(), 1u);
    EXPECT_EQ(environment.cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(environment.host_pool->freeBlocksNum(), host_free_before);
    EXPECT_FALSE(environment.host_pool->isAllocated(left_host_block));
    EXPECT_FALSE(environment.host_pool->isAllocated(right_host_block));
}

TEST(FullPruneTest, DetachesBusyClosureAndKeepsTransferSourceAlive) {
    TieredFullCache environment = makeTieredFullCache();
    const auto      path        = insertFullLowerTierDescendants(environment);
    ASSERT_EQ(path.size(), 4u);
    TreeNode* const    busy_descendant = path[2];
    GroupSetResource&  busy_resource   = busy_descendant->group_set_resources[0];
    const BlockIdxType host_block      = busy_resource.host_block;
    busy_resource.transfer_state       = GroupSetTransferState::LOAD_PENDING;

    std::vector<std::vector<GroupSetResource>> alternative(1, std::vector<GroupSetResource>(1));
    alternative[0][0].device_blocks = {20};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*environment.cache, {600}, alternative));

    EXPECT_EQ(environment.cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);

    EXPECT_EQ(environment.cache->tree()->findNode({100, 200, 300, 400}).size(), 3u);
    EXPECT_EQ(environment.cache->tree()->findNode({600}).size(), 1u);
    EXPECT_TRUE(path[1]->group_set_resources[0].is_empty());
    EXPECT_TRUE(busy_resource.hasTier(Tier::HOST));
    EXPECT_EQ(busy_resource.host_block, host_block);
    EXPECT_EQ(busy_resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    EXPECT_TRUE(busy_resource.transfer_detached);
    EXPECT_TRUE(environment.host_pool->isAllocated(host_block));
    EXPECT_EQ(environment.cache->getStats().device_heap_total_size, 2u);

    // This structural test has no real load task to settle the detached source.
    busy_resource.transfer_state    = GroupSetTransferState::IDLE;
    busy_resource.transfer_detached = false;
}

TEST(FullPruneTest, PrunesCascadedDescendantGroupResourcesAndTopology) {
    DeviceBlockPoolPtr full_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr swa_device_pool  = block_tree_cache_test::makeStructuralDevicePool(1);
    auto               host_pool        = block_tree_cache_test::makeHostPool(1, 4);
    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, host_pool, nullptr);
    auto swa =
        std::make_shared<SWAGroupSet>(128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, nullptr, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    std::vector<GroupSetPtr> group_sets{full, swa};
    auto                     cache = makeBlockTreeCacheForTest(std::move(group_sets), config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = host_block;
    resources[0][1].device_blocks = {20};
    resources[1][1].device_blocks = {21};
    resources[2][1].device_blocks = {22};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300}, resources));

    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);

    const auto path = cache->tree()->findNode({100, 200, 300});
    ASSERT_EQ(path.size(), 1u);
    EXPECT_TRUE(path[0]->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_EQ(path[0]->group_set_resources[1].device_blocks, (BlockIndicesType{20}));
    EXPECT_TRUE(swa_device_pool->isAllocated(20));
    EXPECT_FALSE(swa_device_pool->isAllocated(21));
    EXPECT_FALSE(swa_device_pool->isAllocated(22));
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
}

TEST(FullPruneTest, RefreshesSurvivingFullResourceAfterDescendantPrune) {
    DeviceBlockPoolPtr full0_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr full1_device_pool = block_tree_cache_test::makeStructuralDevicePool(1);
    auto               full0_host_pool   = block_tree_cache_test::makeHostPool(1, 2);
    auto               full0 =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full0_device_pool}, full0_host_pool, nullptr);
    auto full1 = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full1_device_pool}, nullptr, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full0, full1}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType full0_host_block = full0->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(full0_host_block));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(2));
    resources[0][0].host_block    = full0_host_block;
    resources[0][1].device_blocks = {20};
    resources[1][0].device_blocks = {10};
    resources[1][1].device_blocks = {21};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200}, resources));

    // FULL group 0 is complete across tiers. Dropping its HOST root prunes
    // the descendant closure, so group 1 at [100] becomes a DEVICE leaf.
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::HOST), 1);
    ASSERT_EQ(cache->tree()->findNode({100, 200}).size(), 1u);
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST(FullPruneTest, DetachesBusyDescendantGroupResource) {
    DeviceBlockPoolPtr full_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr swa_device_pool  = block_tree_cache_test::makeStructuralDevicePool(1);
    auto               full_host_pool   = block_tree_cache_test::makeHostPool(1, 4);
    auto               swa_host_pool    = block_tree_cache_test::makeHostPool(1, 4);
    auto               full =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, full_host_pool, nullptr);
    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, swa_host_pool, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full, swa}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType full_host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType swa_host_block  = swa->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(full_host_block));
    ASSERT_FALSE(isNullBlockIdx(swa_host_block));
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = full_host_block;
    resources[0][1].device_blocks = {20};
    resources[1][1].device_blocks = {21};
    resources[2][1].host_block    = swa_host_block;
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300}, resources));

    auto path = cache->tree()->findNode({100, 200, 300});
    ASSERT_EQ(path.size(), 3u);
    GroupSetResource& busy_resource = path[2]->group_set_resources[1];
    busy_resource.transfer_state    = GroupSetTransferState::LOAD_PENDING;

    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);

    path = cache->tree()->findNode({100, 200, 300});
    ASSERT_EQ(path.size(), 3u);
    EXPECT_TRUE(path[2]->group_set_resources[0].is_empty());
    EXPECT_EQ(busy_resource.host_block, swa_host_block);
    EXPECT_TRUE(busy_resource.transfer_detached);
    EXPECT_TRUE(swa_host_pool->isAllocated(swa_host_block));
    EXPECT_FALSE(full_host_pool->isAllocated(full_host_block));

    // This structural test has no real load task to settle the detached source.
    busy_resource.transfer_state    = GroupSetTransferState::IDLE;
    busy_resource.transfer_detached = false;
}

TEST(FullPruneTest, DetachesBusyClosureRootGroupResource) {
    DeviceBlockPoolPtr full_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr swa_device_pool  = block_tree_cache_test::makeStructuralDevicePool(1);
    auto               swa_host_pool    = block_tree_cache_test::makeHostPool(1, 4);
    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, nullptr, nullptr);
    auto swa  = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, swa_host_pool, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full, swa}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType swa_host_block = swa->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(swa_host_block));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[0][1].device_blocks = {20};
    resources[1][1].host_block    = swa_host_block;
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200}, resources));

    auto path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    GroupSetResource& busy_resource = path[1]->group_set_resources[1];
    busy_resource.transfer_state    = GroupSetTransferState::LOAD_PENDING;

    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);

    path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_TRUE(path[1]->group_set_resources[0].is_empty());
    EXPECT_EQ(busy_resource.host_block, swa_host_block);
    EXPECT_TRUE(busy_resource.transfer_detached);
    EXPECT_TRUE(swa_host_pool->isAllocated(swa_host_block));

    // This structural test has no real load task to settle the detached source.
    busy_resource.transfer_state    = GroupSetTransferState::IDLE;
    busy_resource.transfer_detached = false;
}

TEST(FullPruneTest, ReverseDirectDropPrunesCascadedFullSubtree) {
    DeviceBlockPoolPtr full_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr swa_device_pool  = block_tree_cache_test::makeStructuralDevicePool(1);
    auto               full_host_pool   = block_tree_cache_test::makeHostPool(1, 4);
    auto               full =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, full_host_pool, nullptr);
    auto swa =
        std::make_shared<SWAGroupSet>(128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, nullptr, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full, swa}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType host_descendant = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_descendant));
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = host_descendant;
    resources[1][1].device_blocks = {21};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300}, resources));

    EXPECT_EQ(cache->evictForGroup(/*group_id=*/1, /*num_blocks=*/1), 1);

    const auto remaining_path = cache->tree()->findNode({100, 200, 300});
    EXPECT_TRUE(remaining_path.empty());
    EXPECT_FALSE(full_host_pool->isAllocated(host_descendant));
}

TEST(FullPruneTest, ReverseDirectDropAttachesOneClosureForMultipleFullGroups) {
    DeviceBlockPoolPtr full0_device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    DeviceBlockPoolPtr full1_device_pool = block_tree_cache_test::makeStructuralDevicePool(1);
    DeviceBlockPoolPtr swa_device_pool   = block_tree_cache_test::makeStructuralDevicePool(2);
    auto               full0_host_pool   = block_tree_cache_test::makeHostPool(1, 4);
    auto               full1_host_pool   = block_tree_cache_test::makeHostPool(1, 4);
    auto               full0 =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full0_device_pool}, full0_host_pool, nullptr);
    auto full1 =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full1_device_pool}, full1_host_pool, nullptr);
    auto swa =
        std::make_shared<SWAGroupSet>(128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, nullptr, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full0, full1, swa}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType full0_host_descendant = full0->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType full1_host_descendant = full1->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(full0_host_descendant));
    ASSERT_FALSE(isNullBlockIdx(full1_host_descendant));
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(3));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].host_block    = full0_host_descendant;
    resources[0][1].device_blocks = {20};
    resources[1][1].device_blocks = {21};
    resources[2][1].host_block    = full1_host_descendant;
    resources[1][2].device_blocks = {31};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300}, resources));

    EXPECT_EQ(cache->evictForGroup(/*group_id=*/2, /*num_blocks=*/1), 1);

    const auto remaining_path = cache->tree()->findNode({100, 200, 300});
    EXPECT_TRUE(remaining_path.empty());
    EXPECT_FALSE(full0_host_pool->isAllocated(full0_host_descendant));
    EXPECT_FALSE(full1_host_pool->isAllocated(full1_host_descendant));
}

TEST(FullPruneTest, WatermarkSelectsAndCommitsEachClosureSequentially) {
    TieredFullCache environment = makeTieredFullCache(/*enable_lower_tiers=*/false);
    const auto      path        = insertFullSandwich(environment);
    ASSERT_EQ(path.size(), 5u);

    const size_t used_blocks = environment.device_pool->usedBlocksNum();
    ASSERT_GE(used_blocks, 2u);
    const double watermark_ratio =
        (static_cast<double>(used_blocks - 2) + 0.5) / environment.device_pool->totalBlocksNum();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment.cache, Tier::DEVICE, watermark_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*environment.cache);

    const auto remaining_path = environment.cache->tree()->findNode({100, 200, 300, 400, 500});
    ASSERT_EQ(remaining_path.size(), 1u);
    EXPECT_EQ(remaining_path.front(), path.front());
    EXPECT_EQ(environment.cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(environment.cache->getStats().device_heap_total_size, 1u);
}

TEST(FullPruneTest, HostPruneKeepsRequestReferencedDeviceDescendantUntilRelease) {
    DeviceBlockPoolPtr             device_pool = block_tree_cache_test::makeStructuralDevicePool(0);
    std::shared_ptr<HostBlockPool> host_pool   = block_tree_cache_test::makeHostPool(1, 16);
    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    std::vector<GroupSetPtr> group_sets{full};
    auto                     cache = makeBlockTreeCacheForTest(std::move(group_sets), config);
    ASSERT_NE(cache, nullptr);

    const size_t       device_free_before = device_pool->freeBlocksNum();
    const size_t       host_free_before   = host_pool->freeBlocksNum();
    const BlockIdxType host_block         = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].host_block    = host_block;
    resources[2][0].device_blocks = {11};
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, {100, 200, 300}, resources));
    device_pool->incRef({11}, BlockRefType::REQUEST);
    ASSERT_EQ(device_pool->refCount(11), 2u);

    const double watermark_ratio = 0.5 / static_cast<double>(host_pool->totalBlocksNum());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, watermark_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);

    EXPECT_EQ(cache->tree()->findNode({100, 200, 300}).size(), 1u);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_TRUE(device_pool->isAllocated(11));
    EXPECT_EQ(device_pool->refCount(11), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
    EXPECT_FALSE(host_pool->isAllocated(host_block));

    block_tree_cache_test::releaseDeviceBlocks(*cache, device_pool, {11}, BlockRefType::REQUEST);
    EXPECT_FALSE(device_pool->isAllocated(11));
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before + 1);
}

}  // namespace
}  // namespace rtp_llm
