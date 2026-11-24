#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
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

TEST(EvictionHeapTest, OrderedTraversalPreservesPolicyAndTieBreakOrderWhileErasingCurrentEntries) {
    for (const auto policy : {EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO}) {
        SCOPED_TRACE(static_cast<int>(policy));
        EvictionHeap                           expected(policy);
        EvictionHeap                           actual(policy);
        std::vector<std::unique_ptr<TreeNode>> nodes;
        for (const CacheKeyType key : {9, 5, 5, 1}) {
            nodes.emplace_back(makeNode(key));
        }
        const std::vector<CandidateMeta> metadata = {
            makeMeta(30, 1, 2), makeMeta(10, 3, 1), makeMeta(10, 3, 1), makeMeta(10, 3, 1)};
        for (size_t i = 0; i < nodes.size(); ++i) {
            expected.upsert(nodes[i].get(), metadata[i]);
            actual.upsert(nodes[i].get(), metadata[i]);
        }

        const auto& ordered  = actual;
        auto        position = ordered.begin();
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto old_entry = expected.best([](TreeNode*) { return true; });
            ASSERT_TRUE(old_entry.has_value());
            ASSERT_NE(position, ordered.end());
            const auto new_entry = *position++;
            EXPECT_EQ(new_entry.node, old_entry->node);
            EXPECT_EQ(new_entry.primary_key, old_entry->primary_key);
            EXPECT_EQ(new_entry.secondary_key, old_entry->secondary_key);
            EXPECT_EQ(new_entry.cache_key, old_entry->cache_key);
            expected.erase(old_entry->node);
            actual.erase(new_entry.node);
        }
        EXPECT_EQ(position, ordered.end());
        EXPECT_EQ(actual.size(), 0u);
    }
}

TEST(EvictionHeapTest, OrderedTraversalVisitsRejectedPrefixOnceAndNewPassRetriesIt) {
    EvictionHeap                           heap(EvictionPolicy::FIFO);
    std::vector<std::unique_ptr<TreeNode>> nodes;
    for (CacheKeyType key = 1; key <= 5; ++key) {
        nodes.emplace_back(makeNode(key));
        heap.upsert(nodes.back().get(), makeMeta(0, key, 0));
    }
    size_t visits = 0;
    {
        const auto& ordered  = heap;
        auto        position = ordered.begin();
        size_t      selected = 0;
        while (position != ordered.end()) {
            const auto entry = *position++;
            ++visits;
            if (entry.node->cache_key <= 2) {
                continue;
            }
            EXPECT_EQ(entry.node, nodes[selected + 2].get());
            ++selected;
            heap.erase(entry.node);
        }
        EXPECT_EQ(selected, 3u);
        EXPECT_EQ(position, ordered.end());
    }
    EXPECT_EQ(visits, 5u);
    EXPECT_EQ(heap.size(), 2u);
    const auto& ordered = heap;
    const auto  retry   = ordered.begin();
    ASSERT_NE(retry, ordered.end());
    EXPECT_EQ(retry->node, nodes.front().get());
}

TEST(EvictionHeapTest, OrderedTraversalErasesRejectedCurrentEntries) {
    EvictionHeap                           heap(EvictionPolicy::FIFO);
    std::vector<std::unique_ptr<TreeNode>> nodes;
    for (CacheKeyType key = 1; key <= 3; ++key) {
        nodes.emplace_back(makeNode(key));
        heap.upsert(nodes.back().get(), makeMeta(0, key, 0));
    }
    const auto&                  ordered  = heap;
    auto                         position = ordered.begin();
    std::optional<EvictionEntry> selected;
    while (position != ordered.end()) {
        const auto entry = *position++;
        if (entry.node->cache_key < 3) {
            heap.erase(entry.node);
            continue;
        }
        selected = entry;
        break;
    }
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->node, nodes.back().get());
    EXPECT_EQ(heap.size(), 1u);
    heap.erase(selected->node);
    EXPECT_EQ(position, ordered.end());
}

TEST(EvictionHeapTest, EmptyOrderedTraversalVisitsNoEntries) {
    EvictionHeap heap(EvictionPolicy::LRU);
    const auto&  ordered = heap;
    EXPECT_EQ(ordered.begin(), ordered.end());
    size_t visits = 0;
    for (const auto& entry : ordered) {
        (void)entry;
        ++visits;
    }
    EXPECT_EQ(visits, 0u);
}

static_assert(!std::is_copy_constructible_v<EvictionHeap::RejectedPrefixScope>);
static_assert(!std::is_move_constructible_v<EvictionHeap::RejectedPrefixScope>);
static_assert(std::is_nothrow_destructible_v<EvictionHeap::RejectedPrefixScope>);

TEST(EvictionHeapTest, RejectedPrefixVisitsMixedRejectionsOnceIncludingTerminalLookup) {
    EvictionHeap                           heap(EvictionPolicy::LRU);
    std::vector<std::unique_ptr<TreeNode>> nodes;
    for (uint64_t rank = 1; rank <= 5; ++rank) {
        nodes.emplace_back(makeNode(rank));
        heap.upsert(nodes.back().get(), makeMeta(rank, rank, rank));
    }
    const std::unordered_set<TreeNode*> held{nodes[0].get(), nodes[2].get(), nodes[4].get()};
    std::unordered_set<TreeNode*>       rejected;
    size_t                              visits = 0;
    {
        std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
        const auto                                       qualify = [&](TreeNode* node) {
            ++visits;
            if (held.count(node) != 0) {
                rejected.insert(node);
                return false;
            }
            return true;
        };
        auto selected = heap.bestForDrop(qualify, rejected, prefix);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->node, nodes[1].get());
        heap.erase(selected->node);
        selected = heap.bestForDrop(qualify, rejected, prefix);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->node, nodes[3].get());
        heap.erase(selected->node);
        EXPECT_FALSE(heap.bestForDrop(qualify, rejected, prefix).has_value());
        EXPECT_EQ(visits, 5u);
        EXPECT_FALSE(heap.bestForDrop(qualify, rejected, prefix).has_value());
        EXPECT_EQ(visits, 5u);
        EXPECT_EQ(heap.size(), 3u);
        EXPECT_EQ(heap.best([](TreeNode*) { return true; })->node, nodes[0].get());
    }
    EXPECT_FALSE(heap.isSkippingRejectedPrefix());
    rejected.clear();
    std::optional<EvictionHeap::RejectedPrefixScope> next(std::in_place, heap, rejected);
    EXPECT_EQ(heap.bestForDrop([](TreeNode*) { return true; }, rejected, next)->node, nodes[0].get());
}

TEST(EvictionHeapTest, RejectedPrefixShrinksForPromotedAndReorderedCandidates) {
    for (const auto policy : {EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO}) {
        for (const bool update_only : {false, true}) {
            SCOPED_TRACE(static_cast<int>(policy));
            SCOPED_TRACE(update_only);
            EvictionHeap heap(policy);
            auto         first    = std::unique_ptr<TreeNode>(makeNode(10));
            auto         boundary = std::unique_ptr<TreeNode>(makeNode(20));
            auto         selected = std::unique_ptr<TreeNode>(makeNode(30));
            auto         later    = std::unique_ptr<TreeNode>(makeNode(40));
            auto         promoted = std::unique_ptr<TreeNode>(makeNode(5));
            for (auto* node : {first.get(), boundary.get(), selected.get(), later.get()}) {
                const auto rank = static_cast<uint64_t>(node->cache_key);
                heap.upsert(node, makeMeta(rank, rank, rank));
            }
            std::unordered_set<TreeNode*>                    rejected{first.get(), boundary.get()};
            std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
            size_t                                           visits  = 0;
            const auto                                       qualify = [&](TreeNode* node) {
                ++visits;
                return rejected.count(node) == 0;
            };
            EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, selected.get());
            heap.erase(selected.get());
            heap.upsert(promoted.get(), makeMeta(5, 5, 5));
            visits = 0;
            EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, promoted.get());
            EXPECT_EQ(visits, 1u);
            heap.erase(promoted.get());
            EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, later.get());
            // The previous lookup rebuilt a fence at rank 20. A live candidate
            // entering that prefix must outrank the remaining rank-40 suffix.
            if (update_only) {
                EXPECT_TRUE(heap.updateIfPresent(later.get(), makeMeta(15, 15, 15)));
            } else {
                heap.upsert(later.get(), makeMeta(15, 15, 15));
            }
            visits = 0;
            EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, later.get());
            // Rank 10 remains proved rejected; resume at the newly ranked node.
            EXPECT_EQ(visits, 1u);
        }
    }
}

TEST(EvictionHeapTest, RejectedPrefixKeepsEarliestOfSequentialUpdates) {
    for (const auto policy : {EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO}) {
        for (const bool earlier_first : {false, true}) {
            for (const bool update_only : {false, true}) {
                SCOPED_TRACE(static_cast<int>(policy));
                SCOPED_TRACE(earlier_first);
                SCOPED_TRACE(update_only);
                EvictionHeap                           heap(policy);
                std::vector<std::unique_ptr<TreeNode>> nodes;
                for (const uint64_t rank : {10u, 20u, 30u, 100u, 200u, 300u}) {
                    nodes.emplace_back(makeNode(rank));
                    heap.upsert(nodes.back().get(), makeMeta(rank, rank, rank));
                }
                std::unordered_set<TreeNode*> rejected{nodes[0].get(), nodes[1].get(), nodes[2].get()};
                std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
                size_t                                           visits  = 0;
                const auto                                       qualify = [&](TreeNode* node) {
                    ++visits;
                    return rejected.count(node) == 0;
                };
                EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[3].get());
                const auto change_rank = [&](size_t index, uint64_t rank) {
                    if (update_only) {
                        EXPECT_TRUE(heap.updateIfPresent(nodes[index].get(), makeMeta(rank, rank, rank)));
                    } else {
                        heap.upsert(nodes[index].get(), makeMeta(rank, rank, rank));
                    }
                };
                if (earlier_first) {
                    change_rank(4, 15);
                    change_rank(3, 25);
                } else {
                    change_rank(3, 25);
                    change_rank(4, 15);
                }
                change_rank(5, 27);
                visits = 0;
                EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[4].get());
                EXPECT_EQ(visits, 1u);
                heap.erase(nodes[4].get());
                visits = 0;
                EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[3].get());
                EXPECT_EQ(visits, 2u);  // rank 20, then the rank-25 candidate.
                heap.erase(nodes[3].get());
                visits = 0;
                EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[5].get());
                EXPECT_EQ(visits, 1u);
                heap.erase(nodes[5].get());
                visits = 0;
                EXPECT_FALSE(heap.bestForDrop(qualify, rejected, prefix).has_value());
                EXPECT_EQ(visits, 1u);  // Only rank 30 remains beyond the proof.
            }
        }
    }
}

TEST(EvictionHeapTest, RejectedPrefixRechecksHeldAdmissionsAndBoundaryChanges) {
    EvictionHeap heap(EvictionPolicy::LRU);
    auto         first          = std::unique_ptr<TreeNode>(makeNode(10));
    auto         middle         = std::unique_ptr<TreeNode>(makeNode(20));
    auto         boundary       = std::unique_ptr<TreeNode>(makeNode(30));
    auto         selected       = std::unique_ptr<TreeNode>(makeNode(50));
    auto         later          = std::unique_ptr<TreeNode>(makeNode(60));
    auto         held_admission = std::unique_ptr<TreeNode>(makeNode(25));
    auto         earlier        = std::unique_ptr<TreeNode>(makeNode(15));
    auto         next           = std::unique_ptr<TreeNode>(makeNode(7));
    auto         head           = std::unique_ptr<TreeNode>(makeNode(4));
    for (auto* node : {first.get(), middle.get(), boundary.get(), selected.get(), later.get()}) {
        const auto rank = static_cast<uint64_t>(node->cache_key);
        heap.upsert(node, makeMeta(rank, rank, rank));
    }
    std::unordered_set<TreeNode*> held{first.get(), middle.get(), boundary.get(), held_admission.get()};
    std::unordered_set<TreeNode*> rejected;
    size_t                        visits = 0;
    {
        std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
        const auto                                       qualify = [&](TreeNode* node) {
            ++visits;
            if (rejected.count(node) != 0) {
                return false;
            }
            if (held.count(node) != 0) {
                rejected.insert(node);
                return false;
            }
            return true;
        };
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, selected.get());
        heap.erase(selected.get());
        heap.upsert(held_admission.get(), makeMeta(25, 25, 25));
        heap.upsert(earlier.get(), makeMeta(15, 15, 15));
        visits = 0;
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, earlier.get());
        EXPECT_EQ(visits, 1u);
        heap.erase(earlier.get());
        visits = 0;
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, later.get());
        EXPECT_EQ(visits, 4u);  // rank 20, newly held 25, rank 30, and rank 60.
        EXPECT_EQ(rejected.count(held_admission.get()), 1u);
        heap.erase(boundary.get());
        held.erase(boundary.get());
        boundary.reset();
        heap.erase(later.get());
        EXPECT_TRUE(heap.updateIfPresent(held_admission.get(), makeMeta(5, 5, 5)));
        heap.upsert(next.get(), makeMeta(7, 7, 7));
        visits = 0;
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, next.get());
        EXPECT_EQ(visits, 2u);  // The deleted boundary requires rebuilding once.
        heap.erase(next.get());
        // A request release does not revoke this batch's rejection identity.
        held.erase(held_admission.get());
        EXPECT_TRUE(heap.updateIfPresent(held_admission.get(), makeMeta(15, 15, 15)));
        heap.upsert(head.get(), makeMeta(4, 4, 4));
        visits = 0;
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, head.get());
        EXPECT_EQ(visits, 1u);
        heap.erase(head.get());
        visits = 0;
        EXPECT_FALSE(heap.bestForDrop(qualify, rejected, prefix).has_value());
        EXPECT_EQ(visits, 3u);
    }
    rejected.clear();
    std::optional<EvictionHeap::RejectedPrefixScope> next_batch(std::in_place, heap, rejected);
    const auto                                       retry = [&](TreeNode* node) {
        if (held.count(node) != 0) {
            rejected.insert(node);
            return false;
        }
        return true;
    };
    EXPECT_EQ(heap.bestForDrop(retry, rejected, next_batch)->node, held_admission.get());
}

TEST(EvictionHeapTest, RejectedPrefixShrinksUsingCacheKeyWithinPolicyTies) {
    for (const auto policy : {EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO}) {
        SCOPED_TRACE(static_cast<int>(policy));
        EvictionHeap                           heap(policy);
        std::vector<std::unique_ptr<TreeNode>> nodes;
        for (const CacheKeyType key : {10, 20, 30, 40}) {
            nodes.emplace_back(makeNode(key));
            heap.upsert(nodes.back().get(), makeMeta(7, 7, 7));
        }
        std::unordered_set<TreeNode*>                    rejected{nodes[0].get(), nodes[1].get(), nodes[2].get()};
        std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
        size_t                                           visits  = 0;
        const auto                                       qualify = [&](TreeNode* node) {
            ++visits;
            return rejected.count(node) == 0;
        };
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[3].get());
        nodes[3]->cache_key = 25;
        EXPECT_TRUE(heap.updateIfPresent(nodes[3].get(), makeMeta(7, 7, 7)));
        visits = 0;
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, nodes[3].get());
        EXPECT_EQ(visits, 1u);
    }
}

TEST(EvictionHeapTest, RejectedPrefixRebuildsAfterRemovedBoundaryAndRejectedReordering) {
    EvictionHeap heap(EvictionPolicy::LRU);
    auto         first    = std::unique_ptr<TreeNode>(makeNode(1));
    auto         boundary = std::unique_ptr<TreeNode>(makeNode(2));
    auto         selected = std::unique_ptr<TreeNode>(makeNode(3));
    auto         later    = std::unique_ptr<TreeNode>(makeNode(4));
    for (auto* node : {first.get(), boundary.get(), selected.get(), later.get()}) {
        const auto rank = static_cast<uint64_t>(node->cache_key);
        heap.upsert(node, makeMeta(rank, rank, rank));
    }
    std::unordered_set<TreeNode*>                    rejected{first.get(), boundary.get()};
    std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
    size_t                                           visits  = 0;
    const auto                                       qualify = [&](TreeNode* node) {
        ++visits;
        return rejected.count(node) == 0;
    };
    EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, selected.get());
    heap.erase(boundary.get());
    boundary.reset();
    heap.erase(selected.get());
    selected.reset();
    // Deleting the boundary invalidates its pointer tie-breaker before destroy.
    // The surviving rejected node is rechecked while rebuilding the certificate.
    EXPECT_TRUE(heap.updateIfPresent(first.get(), makeMeta(0, 0, 0)));
    visits = 0;
    EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, later.get());
    EXPECT_EQ(visits, 2u);
}

TEST(EvictionHeapTest, RejectedPrefixUsesFullComparatorForPointerTies) {
    EvictionHeap                           heap(EvictionPolicy::LRU);
    std::vector<std::unique_ptr<TreeNode>> nodes;
    for (size_t i = 0; i < 4; ++i) {
        nodes.emplace_back(makeNode(7));
        heap.upsert(nodes.back().get(), makeMeta(10, 10, 10));
    }
    std::vector<TreeNode*> order;
    for (const auto& entry : heap) {
        order.push_back(entry.node);
    }
    std::unordered_set<TreeNode*>                    rejected{order[0], order[1]};
    std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
    size_t                                           visits  = 0;
    const auto                                       qualify = [&](TreeNode* node) {
        ++visits;
        return rejected.count(node) == 0;
    };
    EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, order[2]);
    heap.erase(order[2]);
    EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, order[3]);
    EXPECT_EQ(visits, 4u);
}

TEST(EvictionHeapTest, RejectedPrefixKeepsBatchPointerIdentityOnAddressReuse) {
    EvictionHeap            heap(EvictionPolicy::LRU);
    std::optional<TreeNode> storage;
    storage.emplace();
    TreeNode* original  = &*storage;
    original->cache_key = 1;
    auto other          = std::unique_ptr<TreeNode>(makeNode(2));
    heap.upsert(original, makeMeta(1, 1, 1));
    heap.upsert(other.get(), makeMeta(2, 2, 2));
    std::unordered_set<TreeNode*> rejected{original};
    {
        std::optional<EvictionHeap::RejectedPrefixScope> prefix(std::in_place, heap, rejected);
        const auto qualify = [&](TreeNode* node) { return rejected.count(node) == 0; };
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, other.get());
        heap.erase(original);
        storage.reset();
        storage.emplace();
        TreeNode* replacement = &*storage;
        ASSERT_EQ(replacement, original);
        replacement->cache_key = 3;
        heap.upsert(replacement, makeMeta(0, 0, 0));
        // Ordinary cached selection would reject the same reused address too.
        EXPECT_EQ(heap.best(qualify)->node, other.get());
        EXPECT_EQ(heap.bestForDrop(qualify, rejected, prefix)->node, other.get());
        heap.erase(replacement);
    }
    EXPECT_FALSE(heap.isSkippingRejectedPrefix());
}

TEST(EvictionHeapTest, RejectedPrefixRejectsOverlapButAllowsIndependentHeaps) {
    EvictionHeap first(EvictionPolicy::LRU);
    EvictionHeap second(EvictionPolicy::LRU);
    auto         node = std::unique_ptr<TreeNode>(makeNode(1));
    first.upsert(node.get(), makeMeta(1, 1, 1));
    second.upsert(node.get(), makeMeta(1, 1, 1));
    std::unordered_set<TreeNode*> rejected{node.get()};
    const auto                    independent_rejections = rejected;
    {
        std::optional<EvictionHeap::RejectedPrefixScope> outer(std::in_place, first, rejected);
        EXPECT_ANY_THROW({ EvictionHeap::RejectedPrefixScope nested(first, rejected); });
        std::optional<EvictionHeap::RejectedPrefixScope> independent(std::in_place, second, independent_rejections);

        const auto qualify               = [&](TreeNode* candidate) { return rejected.count(candidate) == 0; };
        const auto qualify_independently = [&](TreeNode* candidate) {
            return independent_rejections.count(candidate) == 0;
        };
        EXPECT_FALSE(first.bestForDrop(qualify, rejected, outer).has_value());
        EXPECT_FALSE(second.bestForDrop(qualify_independently, independent_rejections, independent).has_value());
        EXPECT_ANY_THROW(second.bestForDrop(qualify_independently, independent_rejections, outer));
        EXPECT_EQ(first.best([](TreeNode*) { return true; })->node, node.get());
        EXPECT_EQ(second.best([](TreeNode*) { return true; })->node, node.get());
    }
    EXPECT_FALSE(first.isSkippingRejectedPrefix());
    EXPECT_FALSE(second.isSkippingRejectedPrefix());
}

TEST(EvictionHeapTest, RejectedPrefixRejectsDifferentBatchWithIdenticalRejections) {
    EvictionHeap heap(EvictionPolicy::LRU);
    EvictionHeap other_heap(EvictionPolicy::LRU);
    auto         oldest = std::unique_ptr<TreeNode>(makeNode(1));
    auto         next   = std::unique_ptr<TreeNode>(makeNode(2));
    heap.upsert(oldest.get(), makeMeta(1, 1, 1));
    heap.upsert(next.get(), makeMeta(2, 2, 2));
    std::unordered_set<TreeNode*> rejected{oldest.get()};
    const auto                    other_batch = rejected;

    {
        std::optional<EvictionHeap::RejectedPrefixScope> scope(std::in_place, heap, rejected);

        const auto qualify = [&](TreeNode* candidate) { return rejected.count(candidate) == 0; };
        const auto first   = heap.bestForDrop(qualify, rejected, scope);
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->node, next.get());

        EXPECT_ANY_THROW(heap.bestForDrop(qualify, other_batch, scope));
        EXPECT_ANY_THROW(heap.bestForDrop(qualify, other_batch, std::nullopt));
        EXPECT_ANY_THROW(other_heap.bestForDrop(qualify, rejected, scope));

        std::vector<TreeNode*> visited;
        const auto             resumed = heap.bestForDrop(
            [&](TreeNode* candidate) {
                visited.push_back(candidate);
                return qualify(candidate);
            },
            rejected,
            scope);
        ASSERT_TRUE(resumed.has_value());
        EXPECT_EQ(resumed->node, next.get());
        EXPECT_EQ(visited, std::vector<TreeNode*>{next.get()});
    }
    const auto fresh = heap.bestForDrop([](TreeNode*) { return true; }, other_batch, std::nullopt);
    ASSERT_TRUE(fresh.has_value());
    EXPECT_EQ(fresh->node, oldest.get());
}

}  // namespace
}  // namespace rtp_llm
