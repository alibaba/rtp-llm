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
    cache.put(key, std::vector<BlockIdxType>{block}, resident, namespace_id, dep);
}

class RecordingPublisher final: public KVCacheEventPublisher {
public:
    bool start() noexcept override {
        running_ = true;
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        if (!running_) {
            return PublishResult::NOT_RUNNING;
        }
        events.push_back(std::move(event));
        return PublishResult::ACCEPTED;
    }

    void stop() noexcept override {
        running_ = false;
    }

    PublisherStatus status() const noexcept override {
        return {};
    }

    bool enabled() const noexcept override {
        return true;
    }

    std::vector<KVCacheEvent> events;

private:
    bool running_{false};
};

}  // namespace

TEST(SharedBlockCacheTest, EmptyCacheKeepsLegacyVersion) {
    SharedBlockCache cache;
    EXPECT_EQ(cache.version(), -1);
}

TEST(SharedBlockCacheTest, PublisherTracksCompleteLogicalKeysOnly) {
    SharedBlockCache cache;
    auto             publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/2);

    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(),
              std::vector<bool>{true, false});
    EXPECT_TRUE(publisher->events.empty());
    EXPECT_TRUE(cache.logicalCacheSnapshot().cache_keys.empty());

    cache.put(1,
              std::vector<BlockIdxType>{NULL_BLOCK_IDX, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(),
              std::vector<bool>{true, true});
    ASSERT_EQ(publisher->events.size(), 1u);
    EXPECT_EQ(publisher->events[0].type, KVCacheEventType::BLOCK_ADD);
    EXPECT_EQ(publisher->events[0].block_key, 1);
    EXPECT_EQ(cache.logicalCacheSnapshot().cache_keys, (std::vector<CacheKeyType>{1}));

    cache.put(1,
              std::vector<BlockIdxType>{NULL_BLOCK_IDX, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(),
              std::vector<bool>{true, true});
    EXPECT_EQ(publisher->events.size(), 1u);

    ASSERT_TRUE(cache.remove(1).has_value());
    ASSERT_EQ(publisher->events.size(), 2u);
    EXPECT_EQ(publisher->events[1].type, KVCacheEventType::BLOCK_DELETE);
    EXPECT_EQ(publisher->events[1].block_key, 1);
    EXPECT_TRUE(cache.logicalCacheSnapshot().cache_keys.empty());
}

TEST(SharedBlockCacheTest, PublisherIsSeededFromExistingCompleteKeysWithoutDuplicateAdds) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep());

    auto publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/1);

    EXPECT_TRUE(publisher->events.empty());
    ASSERT_TRUE(cache.remove(1).has_value());
    ASSERT_EQ(1u, publisher->events.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events[0].type);
    EXPECT_EQ(1, publisher->events[0].block_key);
}

TEST(SharedBlockCacheTest, PublisherReportsWholeChainEvictionDeletes) {
    SharedBlockCache cache;
    auto             publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/1);

    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));
    ASSERT_EQ(3u, publisher->events.size());

    const auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ((CacheKeysType{1, 2, 3}), evicted.evicted_keys);
    ASSERT_EQ(6u, publisher->events.size());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events[i + 3].type);
        EXPECT_EQ(static_cast<CacheKeyType>(i + 1), publisher->events[i + 3].block_key);
    }
}

TEST(SharedBlockCacheTest, PublisherReportsDeleteWhenIndependentGroupEvictionMakesKeyIncomplete) {
    SharedBlockCache cache;
    auto             publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/2);
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));
    ASSERT_EQ(3u, publisher->events.size());

    const auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ((CacheKeysType{2}), evicted.evicted_keys);
    ASSERT_EQ(4u, publisher->events.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events.back().type);
    EXPECT_EQ(2, publisher->events.back().block_key);
}

TEST(SharedBlockCacheTest, PublisherDoesNotDeleteKeyThatWasNeverComplete) {
    SharedBlockCache cache;
    auto             publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/2);

    putOne(cache, 1, 101, rootDep());
    EXPECT_TRUE(publisher->events.empty());

    const auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ((CacheKeysType{1}), evicted.evicted_keys);
    EXPECT_TRUE(publisher->events.empty());
}

TEST(SharedBlockCacheTest, CapacityReplacementPublishesDeleteBeforeReplacementAdd) {
    SharedBlockCache cache(/*max_capacity=*/1);
    auto             publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/1);

    putOne(cache, 1, 101, rootDep());
    putOne(cache, 2, 102, rootDep());
    putOne(cache, 1, 103, rootDep());

    ASSERT_EQ((std::vector<CacheKeyType>{1}), cache.allCacheKeys());
    ASSERT_EQ(5u, publisher->events.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, publisher->events[0].type);
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events[1].type);
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, publisher->events[2].type);
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events[3].type);
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, publisher->events[4].type);
    EXPECT_EQ((std::vector<CacheKeyType>{1, 1, 2, 2, 1}),
              (std::vector<CacheKeyType>{publisher->events[0].block_key,
                                         publisher->events[1].block_key,
                                         publisher->events[2].block_key,
                                         publisher->events[3].block_key,
                                         publisher->events[4].block_key}));
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsCollectedChainInParentFirstOrderWithDependencies) {
    SharedBlockCache cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2, 3}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1), (std::vector<BlockIdxType>{101}));
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
    auto publisher = std::make_shared<RecordingPublisher>();
    ASSERT_TRUE(publisher->start());
    cache.setEventPublisher(publisher, /*required_group_count=*/1);

    putOne(cache, 8, 108, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 8, NULL_BLOCK_IDX, childDep(7, 7), SharedBlockCache::kGpuLogicalNamespace);
    ASSERT_EQ(1u, publisher->events.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, publisher->events.front().type);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{8}));
    ASSERT_TRUE(evicted.evicted_dependencies.count(8));
    EXPECT_FALSE(evicted.evicted_dependencies.at(8).has_parent);
    EXPECT_EQ(evicted.evicted_dependencies.at(8).ordinal, 0u);
    ASSERT_TRUE(evicted.evicted_namespaces.count(8));
    EXPECT_EQ(evicted.evicted_namespaces.at(8), SharedBlockCache::kGpuCpCanonicalNamespace);
    ASSERT_EQ(2u, publisher->events.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, publisher->events.back().type);
    EXPECT_EQ(8, publisher->events.back().block_key);
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
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1), (std::vector<BlockIdxType>{101, 201}));
}

TEST(SharedBlockCacheTest, StateIndependentEvictionDropsDeepestNonLeafStateFirst) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2}));
    ASSERT_EQ(evicted.evicted_group_block_ids.at(2),
              (std::vector<BlockIdxType>{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302}));
    ASSERT_TRUE(evicted.evicted_independent_group.count(2));
    EXPECT_EQ(evicted.evicted_independent_group.at(2), 3);
    EXPECT_EQ(cache.matchGroup(2, 0), 102);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionScansMultipleLeavesSafely) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));
    cache.put(10,
              std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 310},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(11,
              std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));
    cache.put(12,
              std::vector<BlockIdxType>{112, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 312},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(11, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/2);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{2, 11}));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(11, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
    EXPECT_EQ(cache.matchGroup(12, 3), 312);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionFallsBackToWholeChainWhenOnlyLeafStateRemains) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));

    auto evicted = cache.selectAndEvictForGroup(/*group_id=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evicted.evicted_keys, (CacheKeysType{1, 2}));
    ASSERT_FALSE(evicted.evicted_independent_group.count(2));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupSkipsChainsWithoutTargetSlot) {
    SharedBlockCache cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(10,
              std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(11,
              std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));

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
    ASSERT_EQ(evicted.evicted_group_block_ids.at(1), (std::vector<BlockIdxType>{101, 201}));
    EXPECT_TRUE(isNullBlockIdx(evicted.evicted_group_block_ids.at(2)[1]));
    EXPECT_TRUE(isNullBlockIdx(evicted.evicted_group_block_ids.at(3)[1]));
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

}  // namespace rtp_llm::test
