#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/connector/memory/PrefixTreeMemoryBlockCache.h"

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

PrefixTreeMemoryBlockCache::CacheItem item(CacheKeyType key, CacheBlockKind kind, BlockIdxType block) {
    PrefixTreeMemoryBlockCache::CacheItem item;
    item.cache_key    = key;
    item.kind         = kind;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.block_size   = 1024;
    return item;
}

}  // namespace

TEST(PrefixTreeMemoryBlockCacheTest, ContainsAndMatchAreKindAware) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12)).first);

    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::STATE_SWA_KV));

    auto compressed = cache.match(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(compressed.found);
    EXPECT_EQ(compressed.block_index, 11);

    auto state = cache.match(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(state.found);
    EXPECT_EQ(state.block_index, 12);
}

TEST(PrefixTreeMemoryBlockCacheTest, DuplicateKindDoesNotBlockMissingOtherKind) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto duplicate = cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 13));
    EXPECT_FALSE(duplicate.first);
    EXPECT_FALSE(duplicate.second.has_value());

    auto missing_kind = cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12));
    EXPECT_TRUE(missing_kind.first);
    EXPECT_FALSE(missing_kind.second.has_value());
    EXPECT_EQ(cache.match(1, CacheBlockKind::COMPRESSED_KV).block_index, 11);
    EXPECT_EQ(cache.match(1, CacheBlockKind::STATE_SWA_KV).block_index, 12);
}

TEST(PrefixTreeMemoryBlockCacheTest, EvictionIsPerKindAndStopsAtBranchPoint) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(1, 2), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_FALSE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(3, CacheBlockKind::COMPRESSED_KV));
}

TEST(PrefixTreeMemoryBlockCacheTest, KindLeafAccountingIsIndependent) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::STATE_SWA_KV, 21)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::STATE_SWA_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
}

TEST(PrefixTreeMemoryBlockCacheTest, DetachThenReplaceDoesNotReturnDetachedBackingAgain) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(in_flight.found);
    ASSERT_EQ(in_flight.block_index, 11);

    auto detached = cache.detachIfMatch(1,
                                        CacheBlockKind::COMPRESSED_KV,
                                        CacheBackingType::MEMORY,
                                        in_flight.block_index,
                                        in_flight.disk_slot,
                                        in_flight.generation);
    ASSERT_TRUE(detached.has_value());

    auto replacement = cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 12));
    EXPECT_TRUE(replacement.first);
    EXPECT_FALSE(replacement.second.has_value());

    cache.releaseInFlight(1,
                          CacheBlockKind::COMPRESSED_KV,
                          CacheBackingType::MEMORY,
                          in_flight.block_index,
                          in_flight.disk_slot,
                          in_flight.generation);
    auto matched = cache.match(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.block_index, 12);
}

TEST(PrefixTreeMemoryBlockCacheTest, DetachPrunesEmptyLeafButKeepsStructuralParent) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);

    auto child = cache.matchAndMarkInFlight(2, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(child.found);
    auto detached_child = cache.detachIfMatch(2,
                                              CacheBlockKind::COMPRESSED_KV,
                                              CacheBackingType::MEMORY,
                                              child.block_index,
                                              child.disk_slot,
                                              child.generation);
    ASSERT_TRUE(detached_child.has_value());
    cache.releaseInFlight(2,
                          CacheBlockKind::COMPRESSED_KV,
                          CacheBackingType::MEMORY,
                          child.block_index,
                          child.disk_slot,
                          child.generation);

    EXPECT_FALSE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_EQ(cache.cacheKeys(), (CacheKeysType{1}));
}

}  // namespace rtp_llm::test
