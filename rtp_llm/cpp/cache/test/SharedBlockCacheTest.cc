#include "gtest/gtest.h"

#include <algorithm>
#include <string_view>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"

namespace rtp_llm::test {
namespace {

constexpr std::string_view kFullTag        = "group_0";
constexpr std::string_view kLinearTag      = "group_1";
constexpr std::string_view kIndependentTag = "group_3";

using TaggedBlockId = std::pair<std::string, BlockIdxType>;

class TestSharedBlockCache: public SharedBlockCache {
public:
    TestSharedBlockCache() {
        constexpr uint32_t kBlockNum = 2048;

        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = kBlockNum;
        layout.dtype                    = TYPE_UINT8;
        layout.kv_block_stride_bytes    = 1;
        layout.k_block_stride_bytes     = 1;
        layout.kv_block_pool_size_bytes = kBlockNum;
        layout.total_size_bytes         = kBlockNum;

        BlockPoolConfig config;
        config.pool_name        = "shared_block_cache_test";
        config.block_num        = kBlockNum;
        config.total_size_bytes = kBlockNum;
        config.memory_layouts   = {layout};

        pool_ = std::make_shared<BlockPool>(config, AllocationType::HOST);
        RTP_LLM_CHECK(pool_->init());
        SharedBlockCache::init({
            {std::string(kFullTag), pool_},
            {std::string(kLinearTag), pool_},
            {std::string(kIndependentTag), pool_},
        });
    }

private:
    BlockPoolPtr pool_;
};

TaggedBlockId groupBlock(std::string_view tag, BlockIdxType block_id) {
    return {std::string(tag), block_id};
}

BlockIdxType evictedBlockForTag(const SharedBlockCache::EvictResult& result, CacheKeyType key, std::string_view tag) {
    const auto& group_block_ids = result.evicted_group_block_ids.at(key);
    const auto  it              = group_block_ids.find(std::string(tag));
    return it == group_block_ids.end() ? NULL_BLOCK_IDX : it->second;
}

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
    cache.put(key, {groupBlock(kFullTag, block)}, resident, namespace_id, dep);
}

}  // namespace

TEST(SharedBlockCacheTest, EmptyCacheKeepsLegacyVersion) {
    TestSharedBlockCache cache;
    EXPECT_EQ(cache.version(), -1);
}

TEST(SharedBlockCacheTest, SelectAndEvictOnEmptyCacheEvictsNothing) {
    TestSharedBlockCache cache;

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictWithZeroMinBlocksEvictsNothing) {
    TestSharedBlockCache direct_cache;
    TestSharedBlockCache zero_then_evict_cache;
    const auto           populate = [](SharedBlockCache& cache) {
        putOne(cache, 1, 101, rootDep(0));
        putOne(cache, 2, 102, childDep(1, 1));
        putOne(cache, 3, 103, childDep(2, 2));
    };
    populate(direct_cache);
    populate(zero_then_evict_cache);

    auto zero_evicted = zero_then_evict_cache.selectAndEvict(/*min_blocks=*/0);
    auto direct       = direct_cache.selectAndEvict(/*min_blocks=*/1);
    auto after_zero   = zero_then_evict_cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(zero_evicted.evicted_keys.empty());
    EXPECT_EQ(after_zero.evicted_keys, direct.evicted_keys);
    EXPECT_EQ(after_zero.evicted_keys, (CacheKeysType{1, 2, 3}));
}

TEST(SharedBlockCacheTest, SelectAndEvictRequestExceedingAvailableEvictsEverything) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, rootDep(1));
    putOne(cache, 3, 103, rootDep(2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1000);

    EXPECT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2, 3}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsCollectedChainInParentFirstOrderWithDependencies) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2, 3}));
    ASSERT_EQ(evictedBlockForTag(evicted, 1, kFullTag), 101);
    ASSERT_FALSE(evicted.evicted_dependencies.at(1).has_parent);
    ASSERT_TRUE(evicted.evicted_dependencies.at(2).has_parent);
    ASSERT_EQ(evicted.evicted_dependencies.at(2).parent_key, 1);
    ASSERT_TRUE(evicted.evicted_dependencies.at(3).has_parent);
    ASSERT_EQ(evicted.evicted_dependencies.at(3).parent_key, 2);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeStopsAtBranchPoint) {
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 1, 101, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, kFullTag), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsOrphanLeafWithMissingParentDependency) {
    TestSharedBlockCache cache;
    putOne(cache, 2, 102, childDep(1, 1));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(2));
    EXPECT_TRUE(evicted.evicted_dependencies.at(2).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(2).parent_key, 1);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeAttachesMultiplePendingChildrenAndStopsAtBranch) {
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, kFullTag), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{3}));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_FALSE(cache.contains(3));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossPuts) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/true);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossNamespaceAliases) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace, /*resident=*/true);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionReportsNamespace) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    ASSERT_TRUE(evicted.evicted_namespaces.count(1));
    EXPECT_EQ(evicted.evicted_namespaces.at(1), SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionKeepsCanonicalDependencyWhenLogicalAliasUpdatesSameKey) {
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
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
    TestSharedBlockCache cache;
    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kLinearTag, 201)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0),
              {{std::string(kLinearTag), false}});

    EXPECT_EQ(cache.matchGroup(1, kFullTag), 101);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(1, kLinearTag)));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/2);
    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1}));
    EXPECT_EQ(evictedBlockForTag(evicted, 1, kFullTag), 101);
    EXPECT_EQ(evictedBlockForTag(evicted, 1, kLinearTag), 201);
}

TEST(SharedBlockCacheTest, PutsForDifferentTagsShareOneRequestIdentityAndDependency) {
    TestSharedBlockCache cache;
    cache.put(7, {groupBlock(kFullTag, 107)}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(6, 1));
    cache.put(7, {groupBlock(kLinearTag, 207)}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(6, 1));

    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.matchGroup(7, kFullTag), 107);
    EXPECT_EQ(cache.matchGroup(7, kLinearTag), 207);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);
    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{7}));
    EXPECT_EQ(evictedBlockForTag(evicted, 7, kFullTag), 107);
    EXPECT_EQ(evictedBlockForTag(evicted, 7, kLinearTag), 207);
    ASSERT_EQ(evicted.evicted_dependencies.size(), 1u);
    EXPECT_EQ(evicted.evicted_dependencies.at(7).parent_key, 6);
}

TEST(SharedBlockCacheTest, RejectsDuplicateEmptyAndUnknownInputTagsBeforeUpdatingExistingItem) {
    TestSharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));

    EXPECT_THROW((cache.put(1, {groupBlock(kLinearTag, 201), groupBlock(kLinearTag, 202)}, false)), std::exception);
    EXPECT_THROW(cache.put(1, {groupBlock("", NULL_BLOCK_IDX)}, false), std::exception);
    EXPECT_THROW(cache.put(1, {groupBlock("unknown", NULL_BLOCK_IDX)}, false), std::exception);

    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.matchGroup(1, kFullTag), 101);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(1, kLinearTag)));
}

TEST(SharedBlockCacheTest, RejectsUnknownTagsForMatchAndGroupEviction) {
    TestSharedBlockCache cache;

    EXPECT_THROW(cache.matchGroup(1, ""), std::exception);
    EXPECT_THROW(cache.matchGroup(1, "unknown"), std::exception);
    EXPECT_THROW(cache.selectAndEvictForGroup("unknown", /*min_blocks=*/0), std::exception);
    EXPECT_THROW(cache.evictAndFreeForGroup("unknown", /*min_blocks=*/1), std::exception);
    EXPECT_THROW(cache.setIndependentGroupEviction(/*enabled=*/true, {"unknown"}), std::exception);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionDropsDeepestNonLeafStateFirst) {
    TestSharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {std::string(kIndependentTag)});

    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kIndependentTag, 301)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102), groupBlock(kIndependentTag, 302)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {groupBlock(kFullTag, 103), groupBlock(kIndependentTag, 303)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));

    auto evicted = cache.selectAndEvictForGroup(kIndependentTag, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_EQ(evictedBlockForTag(evicted, 2, kIndependentTag), 302);
    ASSERT_TRUE(evicted.evicted_independent_tag.count(2));
    EXPECT_EQ(evicted.evicted_independent_tag.at(2), kIndependentTag);
    EXPECT_EQ(cache.matchGroup(2, kFullTag), 102);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, kIndependentTag)));
    EXPECT_EQ(cache.matchGroup(3, kIndependentTag), 303);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionScansMultipleLeavesSafely) {
    TestSharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {std::string(kIndependentTag)});

    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kIndependentTag, 301)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102), groupBlock(kIndependentTag, 302)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {groupBlock(kFullTag, 103), groupBlock(kIndependentTag, 303)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));
    cache.put(10,
              {groupBlock(kFullTag, 110), groupBlock(kIndependentTag, 310)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(11,
              {groupBlock(kFullTag, 111), groupBlock(kIndependentTag, 311)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));
    cache.put(12,
              {groupBlock(kFullTag, 112), groupBlock(kIndependentTag, 312)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(11, 2));

    auto evicted = cache.selectAndEvictForGroup(kIndependentTag, /*min_blocks=*/2);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 11}));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, kIndependentTag)));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(11, kIndependentTag)));
    EXPECT_EQ(cache.matchGroup(3, kIndependentTag), 303);
    EXPECT_EQ(cache.matchGroup(12, kIndependentTag), 312);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionFallsBackToWholeChainWhenOnlyLeafStateRemains) {
    TestSharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {std::string(kIndependentTag)});

    cache.put(1, {groupBlock(kFullTag, 101)}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102), groupBlock(kIndependentTag, 302)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));

    auto evicted = cache.selectAndEvictForGroup(kIndependentTag, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    ASSERT_FALSE(evicted.evicted_independent_tag.count(2));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupSkipsChainsWithoutTargetSlot) {
    TestSharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {std::string(kIndependentTag)});

    cache.put(1, {groupBlock(kFullTag, 101)}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, {groupBlock(kFullTag, 102)}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(10, {groupBlock(kFullTag, 110)}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(11,
              {groupBlock(kFullTag, 111), groupBlock(kIndependentTag, 311)},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));

    auto evicted = cache.selectAndEvictForGroup(kIndependentTag, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{10, 11}));
    EXPECT_FALSE(cache.contains(10));
    EXPECT_FALSE(cache.contains(11));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupPrunesBranchUntilTargetAncestorIsEvictable) {
    TestSharedBlockCache cache;
    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kLinearTag, 201)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {groupBlock(kFullTag, 103)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(kLinearTag, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 1, 3}));
    EXPECT_EQ(evictedBlockForTag(evicted, 1, kFullTag), 101);
    EXPECT_EQ(evictedBlockForTag(evicted, 1, kLinearTag), 201);
    EXPECT_TRUE(isNullBlockIdx(evictedBlockForTag(evicted, 2, kLinearTag)));
    EXPECT_TRUE(isNullBlockIdx(evictedBlockForTag(evicted, 3, kLinearTag)));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentSibling) {
    TestSharedBlockCache cache;
    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kLinearTag, 201)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {groupBlock(kFullTag, 103)},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(kLinearTag, /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentDescendant) {
    TestSharedBlockCache cache;
    cache.put(1,
              {groupBlock(kFullTag, 101), groupBlock(kLinearTag, 201)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              {groupBlock(kFullTag, 102)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              {groupBlock(kFullTag, 103)},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));
    cache.put(4,
              {groupBlock(kFullTag, 104)},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(3, 3));

    auto evicted = cache.selectAndEvictForGroup(kLinearTag, /*min_blocks=*/1);

    EXPECT_TRUE(evicted.evicted_keys.empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
    EXPECT_TRUE(cache.contains(4));
}

}  // namespace rtp_llm::test
