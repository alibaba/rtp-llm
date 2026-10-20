#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <random>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"

namespace rtp_llm::test {
namespace {

BlockDependency rootDep(uint32_t ordinal = 0) {
    BlockDependency dep;
    dep.ordinal = ordinal;
    return dep;
}

BlockDependency childDep(CacheKeyType parent, uint32_t ordinal) {
    BlockDependency dep;
    dep.has_parent = true;
    dep.parent_key = parent;
    dep.ordinal    = ordinal;
    return dep;
}

void putOne(SharedBlockCache& cache,
            CacheKeyType      key,
            BlockIdxType      block,
            const BlockDependency& dep,
            SharedBlockCache::NamespaceId namespace_id = SharedBlockCache::kGpuLogicalNamespace,
            bool resident = false) {
    cache.put(key, std::vector<BlockIdxType>{block}, resident, namespace_id, dep);
}

}  // namespace

TEST(SharedBlockCacheTest, CacheKeysSnapshotKeepsVersionConsistentDuringConcurrentPuts) {
    SharedBlockCache  cache;
    const auto        initial_version = cache.version();
    std::atomic<bool> start{false};
    std::atomic<bool> done{false};
    constexpr int     kKeyCount = 2000;
    std::thread       writer([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int key = 1; key <= kKeyCount; ++key) {
            putOne(cache, key, key, rootDep());
            std::this_thread::yield();
        }
        done.store(true);
    });

    start.store(true);
    bool consistent = true;
    do {
        const auto snapshot = cache.snapshotCacheKeys();
        if (snapshot.version - initial_version != static_cast<int64_t>(snapshot.keys.size())) {
            consistent = false;
            break;
        }
        std::this_thread::yield();
    } while (!done.load());
    writer.join();

    EXPECT_TRUE(consistent);
    const auto snapshot = cache.snapshotCacheKeys();
    EXPECT_EQ(snapshot.keys.size(), kKeyCount);
    EXPECT_EQ(snapshot.version, initial_version + kKeyCount);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsCollectedChainInParentFirstOrderWithDependencies) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2, 3}));
    ASSERT_EQ(evicted.evicted_slots.at(1), (std::vector<BlockIdxType>{101}));
    ASSERT_FALSE(evicted.evicted_dependencies.at(1).has_parent);
    ASSERT_TRUE(evicted.evicted_dependencies.at(2).has_parent);
    ASSERT_EQ(evicted.evicted_dependencies.at(2).parent_key, 1);
    ASSERT_TRUE(evicted.evicted_dependencies.at(3).has_parent);
    ASSERT_EQ(evicted.evicted_dependencies.at(3).parent_key, 2);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeStopsAtBranchPoint) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(1, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    EXPECT_FALSE(cache.contains(2));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, PrefixTreeLinksChildInsertedBeforeParent) {
    SharedBlockCache cache;
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 1, 101, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, 0), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsOrphanLeafWithMissingParentDependency) {
    SharedBlockCache cache;
    putOne(cache, 2, 102, childDep(1, 1));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(2));
    EXPECT_TRUE(evicted.evicted_dependencies.at(2).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(2).parent_key, 1);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeAttachesMultiplePendingChildrenAndStopsAtBranch) {
    SharedBlockCache cache;
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(1, 2));
    putOne(cache, 1, 101, rootDep(0));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    EXPECT_FALSE(cache.contains(2));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(3));

    evicted = cache.selectAndEvict(/*min_blocks=*/1);
    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 3}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeStopsAtResidentParent) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/true);
    putOne(cache, 2, 102, childDep(1, 1));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(2));
    EXPECT_TRUE(evicted.evicted_dependencies.at(2).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(2).parent_key, 1);
    EXPECT_TRUE(cache.contains(1));
    EXPECT_FALSE(cache.contains(2));
}

TEST(SharedBlockCacheTest, MatchGroupTouchesPrefixTreeLeafLru) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, 0), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{3}));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_FALSE(cache.contains(3));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossPuts) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/true);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossNamespaceAliases) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace, /*resident=*/true);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionReportsNamespace) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    ASSERT_TRUE(evicted.evicted_namespaces.count(1));
    EXPECT_EQ(evicted.evicted_namespaces.at(1), SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionKeepsCanonicalDependencyWhenLogicalAliasUpdatesSameKey) {
    SharedBlockCache cache;
    putOne(cache, 8, 108, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 8, NULL_BLOCK_IDX, childDep(7, 7), SharedBlockCache::kGpuLogicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{8}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(8));
    EXPECT_FALSE(evicted.evicted_dependencies.at(8).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(8).ordinal, 0u);
    ASSERT_TRUE(evicted.evicted_namespaces.count(8));
    EXPECT_EQ(evicted.evicted_namespaces.at(8), SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, CanonicalAliasOwnsEvictionWhenLogicalAliasIsOlder) {
    SharedBlockCache cache;
    putOne(cache, 100, 1000, rootDep(0), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 101, 1010, childDep(100, 1), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 102, 1020, childDep(101, 2), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 103, 1030, childDep(102, 3), SharedBlockCache::kGpuLogicalNamespace);

    putOne(cache, 101, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 103, NULL_BLOCK_IDX, childDep(101, 1), SharedBlockCache::kGpuCpCanonicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{101, 103}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(101));
    EXPECT_FALSE(evicted.evicted_dependencies.at(101).has_parent);
    ASSERT_TRUE(evicted.evicted_dependencies.count(103));
    EXPECT_TRUE(evicted.evicted_dependencies.at(103).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(103).parent_key, 101);
    EXPECT_EQ(evicted.evicted_namespaces.at(101), SharedBlockCache::kGpuCpCanonicalNamespace);
    EXPECT_EQ(evicted.evicted_namespaces.at(103), SharedBlockCache::kGpuCpCanonicalNamespace);
    EXPECT_TRUE(cache.contains(100));
    EXPECT_TRUE(cache.contains(102));
}

TEST(SharedBlockCacheTest, FlatFallbackKeepsCanonicalDependencyWhenLogicalAliasUpdatesSameKey) {
    SharedBlockCache cache;
    cache.setPrefixTreeEnabled(false);

    putOne(cache, 8, 108, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 8, NULL_BLOCK_IDX, childDep(7, 7), SharedBlockCache::kGpuLogicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{8}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(8));
    EXPECT_FALSE(evicted.evicted_dependencies.at(8).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(8).ordinal, 0u);
    ASSERT_TRUE(evicted.evicted_namespaces.count(8));
    EXPECT_EQ(evicted.evicted_namespaces.at(8), SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, NonMatchableSlotStillEvictsButDoesNotMatchGroup) {
    SharedBlockCache cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0),
              std::vector<bool>{true, false});

    EXPECT_EQ(cache.matchGroup(1, 0), 101);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(1, 1)));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/2);
    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    ASSERT_EQ(evicted.evicted_slots.at(1), (std::vector<BlockIdxType>{101, 201}));
}

TEST(SharedBlockCacheTest, StateIndependentEvictionDropsDeepestNonLeafStateFirst) {
    SharedBlockCache cache;
    cache.setStateBlockIndependentEviction(/*enabled=*/true, {3});

    cache.put(1, std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(3, std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(2, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_EQ(evicted.evicted_slots.at(2),
              (std::vector<BlockIdxType>{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302}));
    ASSERT_TRUE(evicted.evicted_state_only_group.count(2));
    EXPECT_EQ(evicted.evicted_state_only_group.at(2), 3);
    EXPECT_EQ(cache.matchGroup(2, 0), 102);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionScansMultipleLeavesSafely) {
    SharedBlockCache cache;
    cache.setStateBlockIndependentEviction(/*enabled=*/true, {3});

    cache.put(1, std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(3, std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(2, 2));
    cache.put(10, std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 310}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(11, std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(10, 1));
    cache.put(12, std::vector<BlockIdxType>{112, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 312}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(11, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/2);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 11}));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(11, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
    EXPECT_EQ(cache.matchGroup(12, 3), 312);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionFallsBackToWholeChainWhenOnlyLeafStateRemains) {
    SharedBlockCache cache;
    cache.setStateBlockIndependentEviction(/*enabled=*/true, {3});

    cache.put(1, std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    ASSERT_FALSE(evicted.evicted_state_only_group.count(2));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupSkipsChainsWithoutTargetSlot) {
    SharedBlockCache cache;
    cache.setStateBlockIndependentEviction(/*enabled=*/true, {3});

    cache.put(1, std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(10, std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX}, false,
              SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(11, std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311}, false,
              SharedBlockCache::kGpuLogicalNamespace, childDep(10, 1));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{10, 11}));
    EXPECT_FALSE(cache.contains(10));
    EXPECT_FALSE(cache.contains(11));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupPrunesBranchUntilTargetAncestorIsEvictable) {
    SharedBlockCache cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 1, 3}));
    ASSERT_EQ(evicted.evicted_slots.at(1), (std::vector<BlockIdxType>{101, 201}));
    EXPECT_TRUE(isNullBlockIdx(evicted.evicted_slots.at(2)[1]));
    EXPECT_TRUE(isNullBlockIdx(evicted.evicted_slots.at(3)[1]));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentSibling) {
    SharedBlockCache cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentDescendant) {
    SharedBlockCache cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));
    cache.put(4,
              std::vector<BlockIdxType>{104, NULL_BLOCK_IDX},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(3, 3));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
    EXPECT_TRUE(cache.contains(4));
}

TEST(SharedBlockCacheTest, GroupEvictionMatchesGlobalAcrossBranchAndAliasMutations) {
    SharedBlockCache grouped;
    SharedBlockCache global;
    for (auto* cache : {&grouped, &global}) {
        for (int root = 1; root < 200; root += 4) {
            putOne(*cache, root, root, rootDep());
            putOne(*cache, root + 1, root + 1, childDep(root, 1));
            putOne(*cache, root + 2, root + 2, childDep(root, 1));
            putOne(*cache, root + 3, root + 3, childDep(root + 2, 2));
            // Removing one key may invalidate another entry in the leaf set.
            putOne(*cache, root + 1, root + 1, rootDep(), SharedBlockCache::kDefaultNamespace);
        }
    }
    while (!global.empty()) {
        const auto expected = global.selectAndEvict(3);
        const auto actual   = grouped.selectAndEvictForGroup(0, 3);
        ASSERT_EQ(actual.evicted_keys, expected.evicted_keys);
        ASSERT_EQ(actual.evicted_slots, expected.evicted_slots);
    }
    EXPECT_TRUE(grouped.empty());
}

TEST(SharedBlockCacheTest, FlatGlobalEvictionCountsSlotsAndKeepsResidents) {
    SharedBlockCache cache;
    cache.setPrefixTreeEnabled(false);
    putOne(cache, 1, 101, rootDep(), SharedBlockCache::kGpuLogicalNamespace, true);
    cache.put(2, {NULL_BLOCK_IDX, NULL_BLOCK_IDX}, false);
    cache.put(3, {103, 203}, false);
    cache.put(4, {104, NULL_BLOCK_IDX}, false);
    ASSERT_TRUE(cache.match(3).found);

    EXPECT_TRUE(cache.selectAndEvict(0).evicted_keys.empty());
    EXPECT_EQ(cache.selectAndEvict(1).evicted_keys, (CacheKeysType{2, 4}));
    EXPECT_EQ(cache.selectAndEvict(2).evicted_keys, (CacheKeysType{3}));
    EXPECT_TRUE(cache.selectAndEvict(100).evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, FlatGroupEvictionSkipsMissingSlotsAndCountsNonMatchableSlots) {
    SharedBlockCache cache;
    cache.setPrefixTreeEnabled(false);
    cache.put(1, {101, 201}, true);
    cache.put(2, {102}, false);
    cache.put(3, {103, NULL_BLOCK_IDX}, false);
    cache.put(4, {104, 204}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(3, 1), {true, false});
    cache.put(5, {105, 205}, false);

    EXPECT_TRUE(cache.selectAndEvictForGroup(-1, 1).evicted_keys.empty());
    EXPECT_TRUE(cache.selectAndEvictForGroup(5, 1).evicted_keys.empty());
    EXPECT_TRUE(cache.selectAndEvictForGroup(1, 0).evicted_keys.empty());
    const auto evicted = cache.selectAndEvictForGroup(1, 1);
    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{4}));
    EXPECT_EQ(evicted.evicted_slots.at(4), (std::vector<BlockIdxType>{104, 204}));
    EXPECT_EQ(evicted.evicted_dependencies.at(4).parent_key, 3);
    EXPECT_EQ(cache.selectAndEvictForGroup(1, 100).evicted_keys, (CacheKeysType{5}));
    EXPECT_EQ(cache.size(), 3);
}

TEST(SharedBlockCacheTest, FlatEvictionMatchesReferenceLruAfterTouchesAndResidentAliases) {
    struct Item {
        CacheKeyType              key;
        std::vector<BlockIdxType> slots;
        bool                      resident;
    };
    SharedBlockCache cache;
    cache.setPrefixTreeEnabled(false);
    std::vector<Item> oldest_first;
    std::mt19937      random(20261020);
    for (CacheKeyType key = 1; key <= 1000; ++key) {
        Item item{key, {static_cast<BlockIdxType>(key), random() % 3 == 0 ? NULL_BLOCK_IDX :
                                                                 static_cast<BlockIdxType>(key + 1000)},
                  key % 11 == 0};
        cache.put(key, item.slots, item.resident);
        oldest_first.push_back(item);
    }
    for (int i = 0; i < 300; ++i) {
        const size_t index = random() % oldest_first.size();
        auto         item  = oldest_first[index];
        if (i % 3 == 0) {
            // A resident alias protects the single flat entry for this key.
            item.resident = true;
            cache.put(item.key, item.slots, true, SharedBlockCache::kGpuCpCanonicalNamespace, rootDep());
            cache.put(item.key, item.slots, false, SharedBlockCache::kGpuLogicalNamespace, rootDep());
        } else {
            ASSERT_TRUE(cache.match(item.key).found);
        }
        oldest_first.erase(oldest_first.begin() + index);
        oldest_first.push_back(item);
    }
    for (int step = 0; step < 100; ++step) {
        const bool   global = step % 3 == 0;
        const int    group  = step % 2;
        const size_t target = random() % 30;
        size_t       count  = 0;
        CacheKeysType expected;
        std::unordered_map<CacheKeyType, std::vector<BlockIdxType>> expected_slots;
        for (auto it = oldest_first.begin(); it != oldest_first.end() && count < target;) {
            if (it->resident || (!global && isNullBlockIdx(it->slots[group]))) {
                ++it;
                continue;
            }
            expected.push_back(it->key);
            expected_slots[it->key] = it->slots;
            count += global ? std::count_if(it->slots.begin(), it->slots.end(),
                                            [](BlockIdxType slot) { return !isNullBlockIdx(slot); }) : 1;
            it = oldest_first.erase(it);
        }
        const auto actual = global ? cache.selectAndEvict(target) : cache.selectAndEvictForGroup(group, target);
        ASSERT_EQ(actual.evicted_keys, expected) << "step=" << step;
        ASSERT_EQ(actual.evicted_slots, expected_slots) << "step=" << step;
        ASSERT_EQ(cache.size(), oldest_first.size());
    }
}

// Run in separate processes with RTP_LLM_CACHE_EVICTION_INCREMENTAL=0 and 1.
// Only selection/deletion is timed; table creation and teardown are excluded.
TEST(SharedBlockCacheTest, DISABLED_FlatEvictionScalingBenchmark) {
    const char* size_env   = std::getenv("RTP_LLM_EVICT_BENCH_SIZE");
    const char* target_env = std::getenv("RTP_LLM_EVICT_BENCH_TARGETS");
    const bool shuffled = std::getenv("RTP_LLM_EVICT_BENCH_SHUFFLE") != nullptr;
    const size_t nodes    = size_env ? std::stoull(size_env) : 150000;
    const size_t target   = target_env ? std::stoull(target_env) : 32;
    constexpr size_t calls = 32;
    ASSERT_GT(nodes, calls * target);
    for (bool global : {true, false}) {
        SharedBlockCache cache;
        cache.setPrefixTreeEnabled(false);
        CacheKeysType oldest_first;
        oldest_first.reserve(nodes);
        for (size_t key = 1; key <= nodes; ++key) {
            cache.put(key, {static_cast<BlockIdxType>(key)}, false);
            oldest_first.push_back(key);
        }
        if (shuffled) {
            // Model reuse scrambles list order relative to heap allocation.
            // Keep this pointer-chasing case outside the timed region too.
            std::mt19937 random(20261020);
            std::shuffle(oldest_first.begin(), oldest_first.end(), random);
            for (const auto key : oldest_first) {
                ASSERT_TRUE(cache.match(key).found);
            }
        }
        double elapsed_us = 0;
        std::vector<double> samples;
        samples.reserve(calls);
        for (size_t i = 0; i < calls; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = global ? cache.selectAndEvict(target) : cache.selectAndEvictForGroup(0, target);
            const double duration_us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
            elapsed_us += duration_us;
            samples.push_back(duration_us);
            ASSERT_EQ(result.evicted_keys.size(), target);
            EXPECT_EQ(result.evicted_keys.front(), oldest_first[i * target]);
            EXPECT_EQ(result.evicted_keys.back(), oldest_first[(i + 1) * target - 1]);
        }
        std::sort(samples.begin(), samples.end());
        std::cerr << "FLAT_EVICTION_BENCH global=" << global << " nodes=" << nodes << " target=" << target
                  << " shuffled=" << shuffled
                  << " calls=" << calls << " mean_us=" << elapsed_us / calls
                  << " median_us=" << (samples[calls / 2 - 1] + samples[calls / 2]) / 2
                  << " p90_us=" << samples[(calls * 9 + 9) / 10 - 1]
                  << " min_us=" << samples.front() << " max_us=" << samples.back() << std::endl;
    }
}

// Reproduce small, repeated admissions against a large cache. Cache creation
// is excluded from timing; use the same binary/test on each side of an A/B.
TEST(SharedBlockCacheTest, DISABLED_RepeatedSmallEvictionsBenchmark) {
    const char*      size_env   = std::getenv("RTP_LLM_EVICT_BENCH_SIZE");
    const size_t     node_count = size_env ? std::stoull(size_env) : 100000;
    constexpr size_t calls      = 256;
    ASSERT_GT(node_count, calls);
    SharedBlockCache cache;
    for (size_t key = 1; key <= node_count; ++key) {
        putOne(cache, key, static_cast<BlockIdxType>(key), rootDep());
    }
    const auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < calls; ++i) {
        const auto evicted = cache.selectAndEvictForGroup(0, 1);
        ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{i + 1}));
    }
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    std::cerr << "EVICTION_BENCH nodes=" << node_count << " calls=" << calls << " total_us=" << elapsed_us
              << " mean_us=" << elapsed_us / calls << std::endl;
}

// Manual performance regression benchmark. Keep disabled because it builds a
// large synthetic prefix forest and is intended for local scaling checks rather
// than every CI invocation.
//
// Example: set RTP_LLM_EVICT_BENCH_SIZE=10000, then run the test binary with
// --gtest_also_run_disabled_tests and the filter below.
TEST(SharedBlockCacheTest, DISABLED_SelectAndEvictForGroupLongChainBenchmark) {
    const char* size_env = std::getenv("RTP_LLM_EVICT_BENCH_SIZE");
    const size_t node_count = size_env == nullptr ? 10000 : std::stoull(size_env);
    ASSERT_GT(node_count, 0);

    const char* target_count_env = std::getenv("RTP_LLM_EVICT_BENCH_TARGETS");
    const size_t requested_target_count =
        target_count_env == nullptr ? size_t{137} : static_cast<size_t>(std::stoull(target_count_env));
    const size_t target_count = std::min(node_count, requested_target_count);
    ASSERT_GT(target_count, 0);

    const char* chain_length_env = std::getenv("RTP_LLM_EVICT_BENCH_CHAIN_LENGTH");
    const size_t requested_chain_length =
        chain_length_env == nullptr ? node_count : static_cast<size_t>(std::stoull(chain_length_env));
    const size_t chain_length = std::min(node_count, requested_chain_length);
    ASSERT_GT(chain_length, 0);

    SharedBlockCache cache;
    const auto build_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < node_count; ++i) {
        const auto key = static_cast<CacheKeyType>(i + 1);
        const bool has_target_slot = i + target_count >= node_count;
        const bool is_chain_root = i % chain_length == 0;
        cache.put(key,
                  std::vector<BlockIdxType>{static_cast<BlockIdxType>(key),
                                            has_target_slot ? static_cast<BlockIdxType>(key + node_count) :
                                                              NULL_BLOCK_IDX},
                  /*is_resident=*/false,
                  SharedBlockCache::kGpuLogicalNamespace,
                  is_chain_root ? rootDep(0) : childDep(key - 1, static_cast<uint32_t>(i % chain_length)));
    }
    const auto build_end = std::chrono::steady_clock::now();

    const auto evict_begin = std::chrono::steady_clock::now();
    const auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, target_count);
    const auto evict_end = std::chrono::steady_clock::now();

    const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_begin).count();
    const auto evict_ms = std::chrono::duration_cast<std::chrono::milliseconds>(evict_end - evict_begin).count();
    std::cerr << "SharedBlockCache prefix-tree benchmark: nodes=" << node_count << ", chain_length=" << chain_length
              << ", targets=" << target_count << ", build_ms=" << build_ms << ", evict_ms=" << evict_ms
              << std::endl;

    EXPECT_GE(evicted.evicted_keys.size(), target_count);
    EXPECT_EQ(cache.size() + evicted.evicted_keys.size(), node_count);
}

}  // namespace rtp_llm::test
