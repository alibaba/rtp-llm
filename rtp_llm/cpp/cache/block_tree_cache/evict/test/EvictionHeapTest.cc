#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"

namespace rtp_llm {
namespace {

// Helper: create a TreeNode on the heap (caller owns memory).
TreeNode* makeNode(CacheKeyType key) {
    auto* node      = new TreeNode();
    node->cache_key = key;
    return node;
}

// Helper: build a CandidateMeta from explicit fields.
CandidateMeta makeMeta(uint64_t last_access_seq, uint64_t admission_seq, uint64_t hit_count) {
    CandidateMeta meta;
    meta.last_access_seq = last_access_seq;
    meta.admission_seq   = admission_seq;
    meta.hit_count       = hit_count;
    return meta;
}

TEST(EvictionHeapTest, LRUBestOrder) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // last_access_seq ascending -> n1 oldest, n3 newest.
    heap.upsert(n1, makeMeta(/*last_access=*/10, /*admission=*/1, 0));
    heap.upsert(n2, makeMeta(/*last_access=*/20, /*admission=*/2, 0));
    heap.upsert(n3, makeMeta(/*last_access=*/30, /*admission=*/3, 0));

    auto r1 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n1);
    heap.erase(r1->node);

    auto r2 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);
    heap.erase(r2->node);

    auto r3 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n3);
    heap.erase(r3->node);

    EXPECT_EQ(heap.size(), 0u);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, LFUBestOrder) {
    EvictionHeap heap(EvictionPolicy::LFU);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // hit_count ascending -> least frequently used first.
    heap.upsert(n1, makeMeta(/*last_access=*/1, /*admission=*/1, /*hit=*/3));
    heap.upsert(n2, makeMeta(/*last_access=*/2, /*admission=*/2, /*hit=*/1));
    heap.upsert(n3, makeMeta(/*last_access=*/3, /*admission=*/3, /*hit=*/0));

    auto r1 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n3);
    heap.erase(r1->node);

    auto r2 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);
    heap.erase(r2->node);

    auto r3 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n1);
    heap.erase(r3->node);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, FIFOBestOrder) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // admission_seq ascending -> first admitted first out.
    heap.upsert(n1, makeMeta(0, /*admission=*/1, 0));
    heap.upsert(n2, makeMeta(0, /*admission=*/2, 0));
    heap.upsert(n3, makeMeta(0, /*admission=*/3, 0));

    auto r1 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n1);
    heap.erase(r1->node);

    auto r2 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);
    heap.erase(r2->node);

    auto r3 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n3);
    heap.erase(r3->node);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, RepeatedUpsertKeepsSingleEntry) {
    // A single hot node upserted many times keeps exactly one physical entry.
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1 = makeNode(1);

    constexpr uint64_t kUpdateCount = 100000;
    for (uint64_t seq = 1; seq <= kUpdateCount; ++seq) {
        heap.upsert(n1, makeMeta(seq, 1, 0));
    }

    EXPECT_EQ(heap.size(), 1u);
    EXPECT_TRUE(heap.contains(n1));

    // The retained ordering key reflects the most recent upsert.
    auto r = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node, n1);
    EXPECT_EQ(r->primary_key, kUpdateCount);
    heap.erase(r->node);
    EXPECT_EQ(heap.size(), 0u);

    delete n1;
}

TEST(EvictionHeapTest, UpdateIfPresentReordersExistingEntryWithoutAdmittingMissingNode) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* first   = makeNode(1);
    TreeNode* second  = makeNode(2);
    TreeNode* missing = makeNode(3);
    heap.upsert(first, makeMeta(/*last_access=*/10, /*admission=*/1, 0));
    heap.upsert(second, makeMeta(/*last_access=*/20, /*admission=*/2, 0));

    EXPECT_FALSE(heap.updateIfPresent(missing, makeMeta(/*last_access=*/5, /*admission=*/3, 0)));
    EXPECT_FALSE(heap.contains(missing));
    EXPECT_EQ(heap.size(), 2u);

    EXPECT_TRUE(heap.updateIfPresent(first, makeMeta(/*last_access=*/30, /*admission=*/1, 0)));
    ASSERT_TRUE(heap.best([](TreeNode*) { return true; }).has_value());
    EXPECT_EQ(heap.best([](TreeNode*) { return true; })->node, second);
    EXPECT_EQ(heap.size(), 2u);

    delete first;
    delete second;
    delete missing;
}

TEST(EvictionHeapTest, EraseSyncsBothContainers) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    heap.upsert(n1, makeMeta(0, 1, 0));
    heap.upsert(n2, makeMeta(0, 2, 0));
    heap.upsert(n3, makeMeta(0, 3, 0));
    EXPECT_EQ(heap.size(), 3u);

    // Erase the current best; size and membership stay consistent.
    heap.erase(n1);
    EXPECT_FALSE(heap.contains(n1));
    EXPECT_EQ(heap.size(), 2u);

    // best now returns n2 (next by admission_seq).
    auto r = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node, n2);
    heap.erase(r->node);
    EXPECT_EQ(heap.size(), 1u);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, EraseIsIdempotent) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1 = makeNode(1);
    heap.upsert(n1, makeMeta(10, 1, 0));

    heap.erase(n1);
    heap.erase(n1);  // second erase is a no-op
    EXPECT_FALSE(heap.contains(n1));
    EXPECT_EQ(heap.size(), 0u);

    delete n1;
}

TEST(EvictionHeapTest, CacheKeyTieBreak) {
    // Equal ordering keys fall back to cache_key ascending.
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* low  = makeNode(/*cache_key=*/5);
    TreeNode* high = makeNode(/*cache_key=*/9);

    // Same last_access_seq and admission_seq -> tie-break by cache_key.
    heap.upsert(high, makeMeta(100, 1, 0));
    heap.upsert(low, makeMeta(100, 1, 0));

    auto r1 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, low);
    heap.erase(r1->node);

    auto r2 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, high);
    heap.erase(r2->node);

    delete low;
    delete high;
}

TEST(EvictionHeapTest, PointerTieBreakKeepsOtherwiseIdenticalEntries) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1   = makeNode(/*cache_key=*/7);
    TreeNode* n2   = makeNode(/*cache_key=*/7);
    auto      meta = makeMeta(/*last_access=*/100, /*admission=*/1, /*hit=*/0);

    heap.upsert(n1, meta);
    heap.upsert(n2, meta);
    EXPECT_EQ(heap.size(), 2u);
    EXPECT_TRUE(heap.contains(n1));
    EXPECT_TRUE(heap.contains(n2));

    auto r1 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r1.has_value());
    heap.erase(r1->node);
    auto r2 = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_NE(r1->node, r2->node);
    EXPECT_TRUE((r1->node == n1 && r2->node == n2) || (r1->node == n2 && r2->node == n1));
    heap.erase(r2->node);

    delete n1;
    delete n2;
}

TEST(EvictionHeapTest, EmptyBestReturnsNullopt) {
    EvictionHeap heap(EvictionPolicy::LRU);
    EXPECT_FALSE(heap.best([](TreeNode*) { return true; }).has_value());
}

TEST(EvictionHeapTest, BestReturnsVictimWithoutRemovingIt) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* first  = makeNode(1);
    TreeNode* second = makeNode(2);
    heap.upsert(first, makeMeta(0, 1, 0));
    heap.upsert(second, makeMeta(0, 2, 0));

    auto best = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(best.has_value());
    EXPECT_EQ(best->node, first);
    EXPECT_EQ(heap.size(), 2u);
    EXPECT_TRUE(heap.contains(first));
    EXPECT_TRUE(heap.contains(second));

    heap.erase(best->node);
    EXPECT_FALSE(heap.contains(first));
    EXPECT_EQ(heap.size(), 1u);

    delete first;
    delete second;
}

TEST(EvictionHeapTest, BestSkipsEntriesWithoutChangingMembership) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* first  = makeNode(1);
    TreeNode* second = makeNode(2);
    TreeNode* third  = makeNode(3);
    heap.upsert(first, makeMeta(/*last_access=*/10, /*admission=*/1, 0));
    heap.upsert(second, makeMeta(/*last_access=*/20, /*admission=*/2, 0));
    heap.upsert(third, makeMeta(/*last_access=*/30, /*admission=*/3, 0));

    const auto matching = heap.best([first](TreeNode* node) { return node != first; });
    ASSERT_TRUE(matching.has_value());
    EXPECT_EQ(matching->node, second);
    EXPECT_FALSE(heap.best([](TreeNode*) { return false; }).has_value());
    ASSERT_TRUE(heap.best([](TreeNode*) { return true; }).has_value());
    EXPECT_EQ(heap.best([](TreeNode*) { return true; })->node, first);
    EXPECT_EQ(heap.size(), 3u);
    EXPECT_TRUE(heap.contains(first));
    EXPECT_TRUE(heap.contains(second));
    EXPECT_TRUE(heap.contains(third));

    delete first;
    delete second;
    delete third;
}

TEST(EvictionHeapTest, ContainsAfterErase) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* n1 = makeNode(1);
    heap.upsert(n1, makeMeta(0, 1, 0));
    EXPECT_TRUE(heap.contains(n1));

    auto best = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(best.has_value());
    heap.erase(best->node);
    EXPECT_FALSE(heap.contains(n1));

    delete n1;
}

TEST(EvictionHeapTest, UpsertNullNodeIgnored) {
    EvictionHeap heap(EvictionPolicy::LRU);
    heap.upsert(nullptr, makeMeta(1, 1, 0));
    EXPECT_EQ(heap.size(), 0u);
}

class FrozenEvictionHeapTest: public ::testing::TestWithParam<EvictionPolicy> {};

TEST_P(FrozenEvictionHeapTest, FrozenEntriesFollowOrdinaryEntriesAndStopSelection) {
    EvictionHeap heap(GetParam());
    TreeNode     oldest;
    TreeNode     middle;
    TreeNode     newest;
    oldest.cache_key = 1;
    middle.cache_key = 2;
    newest.cache_key = 3;
    heap.upsert(&oldest, makeMeta(10, 10, 10));
    heap.upsert(&middle, makeMeta(20, 20, 20));
    heap.upsert(&newest, makeMeta(30, 30, 30));

    ASSERT_TRUE(heap.setFrozenIfPresent(&oldest, true));
    const auto victim = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, &middle);
    EXPECT_FALSE(victim->is_frozen);

    std::vector<TreeNode*> visited;
    const auto             rejected = heap.best([&](TreeNode* node) {
        visited.push_back(node);
        return false;
    });
    EXPECT_FALSE(rejected.has_value());
    EXPECT_EQ(visited, (std::vector<TreeNode*>{&middle, &newest}));
    EXPECT_EQ(heap.size(), 3u);

    ASSERT_TRUE(heap.setFrozenIfPresent(&middle, true));
    ASSERT_TRUE(heap.setFrozenIfPresent(&newest, true));
    size_t     predicate_calls = 0;
    const auto all_frozen      = heap.best([&](TreeNode*) {
        ++predicate_calls;
        return true;
    });
    EXPECT_FALSE(all_frozen.has_value());
    EXPECT_EQ(predicate_calls, 0u);

    ASSERT_TRUE(heap.setFrozenIfPresent(&oldest, false));
    const auto thawed = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(thawed.has_value());
    EXPECT_EQ(thawed->node, &oldest);
    EXPECT_EQ(thawed->primary_key, 10u);
    EXPECT_EQ(heap.size(), 3u);
}

TEST_P(FrozenEvictionHeapTest, MetadataUpdatePreservesFrozenStateAndRestoresLatestKeysOnThaw) {
    EvictionHeap heap(GetParam());
    TreeNode     frozen;
    TreeNode     ordinary;
    frozen.cache_key   = 1;
    ordinary.cache_key = 2;
    heap.upsert(&frozen, makeMeta(10, 10, 10), /*is_frozen=*/true);
    heap.upsert(&ordinary, makeMeta(100, 100, 100));

    ASSERT_TRUE(heap.updateIfPresent(&frozen, makeMeta(30, 40, 50)));
    const auto victim = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, &ordinary);

    ASSERT_TRUE(heap.setFrozenIfPresent(&frozen, false));
    const auto thawed = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(thawed.has_value());
    EXPECT_EQ(thawed->node, &frozen);
    switch (GetParam()) {
        case EvictionPolicy::LRU:
            EXPECT_EQ(thawed->primary_key, 30u);
            EXPECT_EQ(thawed->secondary_key, 40u);
            break;
        case EvictionPolicy::LFU:
            EXPECT_EQ(thawed->primary_key, 50u);
            EXPECT_EQ(thawed->secondary_key, 30u);
            break;
        case EvictionPolicy::FIFO:
            EXPECT_EQ(thawed->primary_key, 40u);
            EXPECT_EQ(thawed->secondary_key, 0u);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(AllPolicies,
                         FrozenEvictionHeapTest,
                         ::testing::Values(EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO));

TEST(EvictionHeapTest, FrozenUpdatesAreIdempotentAndNeverAdmitMissingNodes) {
    EvictionHeap heap(EvictionPolicy::LRU);
    TreeNode     node;
    EXPECT_FALSE(heap.setFrozenIfPresent(&node, true));
    EXPECT_FALSE(heap.setFrozenIfPresent(&node, false));
    EXPECT_EQ(heap.size(), 0u);

    heap.upsert(&node, makeMeta(10, 20, 30));
    EXPECT_TRUE(heap.setFrozenIfPresent(&node, true));
    EXPECT_TRUE(heap.setFrozenIfPresent(&node, true));
    EXPECT_FALSE(heap.best([](TreeNode*) { return true; }).has_value());
    EXPECT_EQ(heap.size(), 1u);

    EXPECT_TRUE(heap.setFrozenIfPresent(&node, false));
    EXPECT_TRUE(heap.setFrozenIfPresent(&node, false));
    const auto victim = heap.best([](TreeNode*) { return true; });
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, &node);
    EXPECT_EQ(victim->primary_key, 10u);
    EXPECT_EQ(victim->secondary_key, 20u);
    heap.erase(&node);
    EXPECT_FALSE(heap.setFrozenIfPresent(&node, true));
    EXPECT_EQ(heap.size(), 0u);
}

}  // namespace
}  // namespace rtp_llm
