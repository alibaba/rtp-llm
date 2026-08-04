#include <gtest/gtest.h>

#include <algorithm>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using block_tree_cache_test::BlockTreeCacheTestPeer;
using block_tree_cache_test::makeBlockTreeCacheForTest;

// Test double that records which resources FullGroupSet::isEvictable is
// asked about while recording is enabled, then forwards to the base class.
// During a no-eviction insert, refreshCandidate is the only caller of
// isEvictable, so recording these evaluations lets a test verify exactly
// which resources the evictor re-evaluates -- without any production-side hook.
class CountingFullGroupSet: public FullGroupSet {
public:
    using FullGroupSet::FullGroupSet;
    bool isEvictable(const GroupSetResource& resource, Tier tier) const override {
        if (recording_) {
            checked_resources_.push_back(&resource);
        }
        return FullGroupSet::isEvictable(resource, tier);
    }

    mutable bool                                 recording_{false};
    mutable std::vector<const GroupSetResource*> checked_resources_;
};

// Helper: build a BlockTreeCache with a single Full(REUSABLE) group.
class FullEvictionTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto full = std::make_shared<CountingFullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        counting_full_                  = full.get();
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
    CountingFullGroupSet*           counting_full_{nullptr};
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
    const std::vector<TreeNode*> path_before               = before;
    const TreeNode* const        direct_parent             = before.back();
    const CandidateMeta          direct_parent_meta_before = direct_parent->group_set_resources[0].candidate_meta;

    // Record every isEvictable evaluation during the extending insert.
    // Metadata alone cannot distinguish "refresh only the direct parent" from
    // "scan every ancestor": refreshCandidate does not mutate CandidateMeta and
    // interior FULL nodes are heap-ineligible anyway, so a regression that
    // re-walks the whole prefix would keep the other assertions green. With no
    // eviction pressure, refreshCandidate is the sole caller of isEvictable,
    // so the counting double captures exactly the re-evaluated nodes.
    counting_full_->checked_resources_.clear();
    counting_full_->recording_ = true;
    insertPath({100, 200, 300, 400, 500}, 20);
    counting_full_->recording_ = false;

    EXPECT_EQ(cache_->getStats().tree_node_count, 5u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 1u);  // only the new [500] leaf
    const auto after = cache_->tree()->findNode({100, 200, 300, 400, 500});
    ASSERT_EQ(after.size(), 5u);

    const auto refresh_count = [this](const TreeNode* node) {
        return std::count(counting_full_->checked_resources_.begin(),
                          counting_full_->checked_resources_.end(),
                          &node->group_set_resources[0]);
    };
    EXPECT_EQ(refresh_count(direct_parent), 1) << "direct parent must be re-evaluated exactly once";
    EXPECT_EQ(refresh_count(after.back()), 1) << "new leaf is admitted exactly once";
    for (size_t index = 0; index + 1 < path_before.size(); ++index) {
        EXPECT_EQ(refresh_count(path_before[index]), 0) << "ancestor=" << index << " must not be re-scanned";
    }

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
    cache_->waitForPendingTasks();

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
    cache_->waitForPendingTasks();

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
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);

    // Step 2: reclaim [200]
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    // Step 3: reclaim [100]
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
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
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 2u);  // [100] + one leaf

    // Reclaim second leaf
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);  // [100] survives (has data)

    // Reclaim [100] (now leaf after both children deleted)
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE);
    cache_->waitForPendingTasks();
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
    cache_->waitForPendingTasks();

    // [100] was reclaimed (oldest). Only [200] remains.
    EXPECT_EQ(cache_->getStats().tree_node_count, 1u);

    auto result = cache_->match({200});
    EXPECT_EQ(result.matched_device_blocks, 1u);
}

// A real match is the only event that advances LRU heat. Matching the oldest
// leaf makes it newer than a leaf inserted later, so the latter is reclaimed.
TEST_F(FullEvictionTest, MatchRefreshesLruOrder) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    auto match = cache_->match({100});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    cache_->releaseMatchedResources(match.matched_device_resources);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->match({100}).matched_device_blocks, 1u);
    EXPECT_EQ(cache_->match({200}).matched_device_blocks, 0u);
}

// Releasing match-protection references only restores candidate eligibility;
// it must not count as another access or change the ordering metadata.
TEST_F(FullEvictionTest, MatchReleaseDoesNotMutateHeat) {
    insertPath({100}, 10);
    insertPath({200}, 20);

    auto match = cache_->match({100});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    auto found = cache_->tree()->findNode({100});
    ASSERT_FALSE(found.empty());
    const CandidateMeta meta_after_match = found.back()->group_set_resources[0].candidate_meta;
    ASSERT_EQ(meta_after_match.hit_count, 1u);

    cache_->releaseMatchedResources(match.matched_device_resources);

    const CandidateMeta meta_after_release = found.back()->group_set_resources[0].candidate_meta;
    EXPECT_EQ(meta_after_release.last_access_seq, meta_after_match.last_access_seq);
    EXPECT_EQ(meta_after_release.admission_seq, meta_after_match.admission_seq);
    EXPECT_EQ(meta_after_release.hit_count, meta_after_match.hit_count);

    // Releasing the hotter node must not refresh it again: the untouched rival
    // remains the next LRU victim.
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    auto hot_match  = cache_->match({100});
    auto cold_match = cache_->match({200});
    EXPECT_EQ(hot_match.matched_device_blocks, 1u);
    EXPECT_EQ(cold_match.matched_device_blocks, 0u);
    cache_->releaseMatchedResources(hot_match.matched_device_resources);
    cache_->releaseMatchedResources(cold_match.matched_device_resources);
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
    cache_->waitForPendingTasks();

    EXPECT_EQ(cache_->match({100}).matched_device_blocks, 0u);
    EXPECT_EQ(cache_->match({200}).matched_device_blocks, 1u);
}

}  // namespace
}  // namespace rtp_llm
