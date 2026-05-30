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

void putOne(SharedBlockCache& cache,
            CacheKeyType      key,
            BlockIdxType      block,
            const BlockDependency& dep,
            SharedBlockCache::NamespaceId namespace_id = SharedBlockCache::kGpuLogicalNamespace,
            bool resident = false) {
    cache.put(key, std::vector<BlockIdxType>{block}, resident, namespace_id, dep);
}

}  // namespace

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

}  // namespace rtp_llm::test
