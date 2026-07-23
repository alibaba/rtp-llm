#include "gtest/gtest.h"

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

void putOne(SharedBlockCache&             cache,
            CacheKeyType                  key,
            BlockIdxType                  block,
            const BlockDependency&        dep,
            SharedBlockCache::NamespaceId namespace_id = SharedBlockCache::kGpuLogicalNamespace,
            bool                          resident     = false) {
    cache.put(key, {{"full", block}}, resident, namespace_id, dep);
}

}  // namespace

TEST(SharedBlockCacheTest, EmptyCacheKeepsLegacyVersion) {
    SharedBlockCache cache;
    EXPECT_EQ(cache.version(), -1);
}

TEST(SharedBlockCacheTest, IndexedStoragePreservesTopologyOrder) {
    SharedBlockCache cache;
    cache.init({{"z_group", nullptr}, {"a_group", nullptr}});

    cache.putIndexed(7, {70, 71}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0), {1, 1});

    const auto matched = cache.match(7);
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.group_block_ids, (SharedBlockCache::TaggedBlockIds{{"a_group", 71}, {"z_group", 70}}));
    EXPECT_EQ(cache.matchGroup(7, "z_group"), 70);
    EXPECT_EQ(cache.matchGroup(7, "a_group"), 71);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsCollectedChainInParentFirstOrderWithDependencies) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2, 3}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1), (SharedBlockCache::TaggedBlockIds{{"full", 101}}));
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

    ASSERT_EQ(cache.matchGroup(2, "full"), 102);

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

    ASSERT_EQ(cache.matchGroup(2, "full"), 102);

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
              {{"full", 101}, {"linear", 201}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0),
              {{"full", true}, {"linear", false}});

    EXPECT_EQ(cache.matchGroup(1, "full"), 101);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(1, "linear")));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/2);
    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1),
              (SharedBlockCache::TaggedBlockIds{{"full", 101}, {"linear", 201}}));
}

TEST(SharedBlockCacheTest, StateIndependentEvictionDropsDeepestNonLeafStateFirst) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {"state"});

    cache.put(1, {{"full", 101}, {"state", 301}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, {{"full", 102}, {"state", 302}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(3, {{"full", 103}, {"state", 303}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(2, 2));

    auto evicted = cache.selectAndEvictForGroup("state", /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(2), (SharedBlockCache::TaggedBlockIds{{"state", 302}}));
    ASSERT_TRUE(evicted.evicted_independent_group.count(2));
    EXPECT_EQ(evicted.evicted_independent_group.at(2), "state");
    EXPECT_EQ(cache.matchGroup(2, "full"), 102);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, "state")));
    EXPECT_EQ(cache.matchGroup(3, "state"), 303);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionScansMultipleLeavesSafely) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {"state"});

    cache.put(1, {{"full", 101}, {"state", 301}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, {{"full", 102}, {"state", 302}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(3, {{"full", 103}, {"state", 303}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(2, 2));
    cache.put(10, {{"full", 110}, {"state", 310}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(11, {{"full", 111}, {"state", 311}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(10, 1));
    cache.put(12, {{"full", 112}, {"state", 312}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(11, 2));

    auto evicted = cache.selectAndEvictForGroup("state", /*min_blocks=*/2);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 11}));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, "state")));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(11, "state")));
    EXPECT_EQ(cache.matchGroup(3, "state"), 303);
    EXPECT_EQ(cache.matchGroup(12, "state"), 312);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionFallsBackToWholeChainWhenOnlyLeafStateRemains) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {"state"});

    cache.put(1, {{"full", 101}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, {{"full", 102}, {"state", 302}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));

    auto evicted = cache.selectAndEvictForGroup("state", /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    ASSERT_FALSE(evicted.evicted_independent_group.count(2));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupSkipsChainsWithoutTargetSlot) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {"state"});

    cache.put(1, {{"full", 101}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, {{"full", 102}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(10, {{"full", 110}}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(11, {{"full", 111}, {"state", 311}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(10, 1));

    auto evicted = cache.selectAndEvictForGroup("state", /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{10, 11}));
    EXPECT_FALSE(cache.contains(10));
    EXPECT_FALSE(cache.contains(11));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupPrunesBranchUntilTargetAncestorIsEvictable) {
    SharedBlockCache cache;
    cache.put(1,
              {{"full", 101}, {"linear", 201}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {{"full", 102}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {{"full", 103}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup("linear", /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 1, 3}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1),
              (SharedBlockCache::TaggedBlockIds{{"full", 101}, {"linear", 201}}));
    EXPECT_FALSE(evicted.evicted_group_block_ids.at(2).count("linear"));
    EXPECT_FALSE(evicted.evicted_group_block_ids.at(3).count("linear"));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentSibling) {
    SharedBlockCache cache;
    cache.put(1,
              {{"full", 101}, {"linear", 201}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {{"full", 102}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {{"full", 103}},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup("linear", /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentDescendant) {
    SharedBlockCache cache;
    cache.put(1,
              {{"full", 101}, {"linear", 201}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {{"full", 102}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {{"full", 103}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));
    cache.put(4,
              {{"full", 104}},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(3, 3));

    auto evicted = cache.selectAndEvictForGroup("linear", /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
    EXPECT_TRUE(cache.contains(4));
}

}  // namespace rtp_llm::test
