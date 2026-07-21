#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"

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

TEST(EvictionHeapTest, LRUTakeBestOrder) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // last_access_seq ascending -> n1 oldest, n3 newest.
    heap.upsert(n1, makeMeta(/*last_access=*/10, /*admission=*/1, 0));
    heap.upsert(n2, makeMeta(/*last_access=*/20, /*admission=*/2, 0));
    heap.upsert(n3, makeMeta(/*last_access=*/30, /*admission=*/3, 0));

    auto r1 = heap.takeBest();
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n1);

    auto r2 = heap.takeBest();
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);

    auto r3 = heap.takeBest();
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n3);

    EXPECT_EQ(heap.size(), 0u);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, LFUTakeBestOrder) {
    EvictionHeap heap(EvictionPolicy::LFU);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // hit_count ascending -> least frequently used first.
    heap.upsert(n1, makeMeta(/*last_access=*/1, /*admission=*/1, /*hit=*/3));
    heap.upsert(n2, makeMeta(/*last_access=*/2, /*admission=*/2, /*hit=*/1));
    heap.upsert(n3, makeMeta(/*last_access=*/3, /*admission=*/3, /*hit=*/0));

    auto r1 = heap.takeBest();
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n3);

    auto r2 = heap.takeBest();
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);

    auto r3 = heap.takeBest();
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n1);

    delete n1;
    delete n2;
    delete n3;
}

TEST(EvictionHeapTest, FIFOTakeBestOrder) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* n1 = makeNode(1);
    TreeNode* n2 = makeNode(2);
    TreeNode* n3 = makeNode(3);

    // admission_seq ascending -> first admitted first out.
    heap.upsert(n1, makeMeta(0, /*admission=*/1, 0));
    heap.upsert(n2, makeMeta(0, /*admission=*/2, 0));
    heap.upsert(n3, makeMeta(0, /*admission=*/3, 0));

    auto r1 = heap.takeBest();
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, n1);

    auto r2 = heap.takeBest();
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, n2);

    auto r3 = heap.takeBest();
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3->node, n3);

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
    auto r = heap.takeBest();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node, n1);
    EXPECT_EQ(r->primary_key, kUpdateCount);
    EXPECT_EQ(heap.size(), 0u);

    delete n1;
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

    // takeBest now returns n2 (next by admission_seq).
    auto r = heap.takeBest();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node, n2);
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

    auto r1 = heap.takeBest();
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->node, low);

    auto r2 = heap.takeBest();
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->node, high);

    delete low;
    delete high;
}

TEST(EvictionHeapTest, PointerTieBreakKeepsOtherwiseIdenticalEntries) {
    EvictionHeap heap(EvictionPolicy::LRU);

    TreeNode* n1 = makeNode(/*cache_key=*/7);
    TreeNode* n2 = makeNode(/*cache_key=*/7);
    auto      meta = makeMeta(/*last_access=*/100, /*admission=*/1, /*hit=*/0);

    heap.upsert(n1, meta);
    heap.upsert(n2, meta);
    EXPECT_EQ(heap.size(), 2u);
    EXPECT_TRUE(heap.contains(n1));
    EXPECT_TRUE(heap.contains(n2));

    auto r1 = heap.takeBest();
    auto r2 = heap.takeBest();
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_NE(r1->node, r2->node);
    EXPECT_TRUE((r1->node == n1 && r2->node == n2) || (r1->node == n2 && r2->node == n1));

    delete n1;
    delete n2;
}

TEST(EvictionHeapTest, EmptyTakeBestReturnsNullopt) {
    EvictionHeap heap(EvictionPolicy::LRU);
    EXPECT_EQ(heap.size(), 0u);
    auto result = heap.takeBest();
    EXPECT_FALSE(result.has_value());
}

TEST(EvictionHeapTest, ContainsAfterTakeBest) {
    EvictionHeap heap(EvictionPolicy::FIFO);

    TreeNode* n1 = makeNode(1);
    heap.upsert(n1, makeMeta(0, 1, 0));
    EXPECT_TRUE(heap.contains(n1));

    heap.takeBest();
    EXPECT_FALSE(heap.contains(n1));

    delete n1;
}

TEST(EvictionHeapTest, UpsertNullNodeIgnored) {
    EvictionHeap heap(EvictionPolicy::LRU);
    heap.upsert(nullptr, makeMeta(1, 1, 0));
    EXPECT_EQ(heap.size(), 0u);
}

}  // namespace
}  // namespace rtp_llm
