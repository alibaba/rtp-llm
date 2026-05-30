#include "gtest/gtest.h"

#include <utility>
#include <vector>

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

PrefixTreeMemoryBlockCache::CacheItem
item(CacheKeyType key, CacheBlockKind kind, BlockIdxType block, std::vector<uint8_t> slot_valid_mask = {}) {
    PrefixTreeMemoryBlockCache::CacheItem item;
    item.cache_key    = key;
    item.kind         = kind;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.block_size   = 1024;
    item.slot_valid_mask = std::move(slot_valid_mask);
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

TEST(PrefixTreeMemoryBlockCacheTest, SlotMaskMustCoverRequestedSlots) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    EXPECT_TRUE(cache.contains(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0}));
    EXPECT_TRUE(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0}).found);
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0}));
    EXPECT_FALSE(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0}).found);
}

TEST(PrefixTreeMemoryBlockCacheTest, WiderSlotMaskReplacesNarrowerBacking) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto replacement = cache.putCommitted(1,
                                          rootDep(),
                                          item(1,
                                               CacheBlockKind::STATE_SWA_KV,
                                               12,
                                               /*slot_valid_mask=*/std::vector<uint8_t>{1, 1, 0}));
    ASSERT_TRUE(replacement.first);
    ASSERT_TRUE(replacement.second.has_value());
    EXPECT_EQ(replacement.second->block_index, 11);

    auto matched = cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0});
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.block_index, 12);
}

TEST(PrefixTreeMemoryBlockCacheTest, NonCoveringSlotMaskDoesNotReplaceBacking) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto replacement = cache.putCommitted(1,
                                          rootDep(),
                                          item(1,
                                               CacheBlockKind::STATE_SWA_KV,
                                               12,
                                               /*slot_valid_mask=*/std::vector<uint8_t>{1, 0, 0}));
    EXPECT_FALSE(replacement.first);
    EXPECT_FALSE(replacement.second.has_value());

    EXPECT_TRUE(cache.contains(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0}));
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0}));
    EXPECT_EQ(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0}).block_index, 11);
}

TEST(PrefixTreeMemoryBlockCacheTest, InFlightCanBeReplacedByCoveringBacking) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0});
    ASSERT_TRUE(in_flight.found);

    auto replacement = cache.putCommitted(1,
                                          rootDep(),
                                          item(1,
                                               CacheBlockKind::STATE_SWA_KV,
                                               12,
                                               /*slot_valid_mask=*/std::vector<uint8_t>{1, 1, 0}));
    ASSERT_TRUE(replacement.first);
    EXPECT_FALSE(replacement.second.has_value());

    auto matched = cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0});
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.block_index, 12);

    auto retired = cache.releaseInFlight(1,
                                         CacheBlockKind::STATE_SWA_KV,
                                         CacheBackingType::MEMORY,
                                         in_flight.block_index,
                                         in_flight.disk_slot,
                                         in_flight.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 11);
    matched = cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0});
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.block_index, 12);
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

TEST(PrefixTreeMemoryBlockCacheTest, PrefixTreeLinksChildInsertedBeforeParent) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_FALSE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
}

TEST(PrefixTreeMemoryBlockCacheTest, ParentDetachPreservesChildLeafAccounting) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);

    auto parent = cache.matchAndMarkInFlight(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(parent.found);
    EXPECT_FALSE(cache.detachIfMatch(1,
                                     CacheBlockKind::COMPRESSED_KV,
                                     CacheBackingType::MEMORY,
                                     parent.block_index,
                                     parent.disk_slot,
                                     parent.generation)
                     .has_value());
    auto retired_parent = cache.releaseInFlight(1,
                                                CacheBlockKind::COMPRESSED_KV,
                                                CacheBackingType::MEMORY,
                                                parent.block_index,
                                                parent.disk_slot,
                                                parent.generation);
    ASSERT_TRUE(retired_parent.has_value());

    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 13)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_FALSE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
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
    EXPECT_FALSE(detached.has_value());

    auto replacement = cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 12));
    EXPECT_TRUE(replacement.first);
    EXPECT_FALSE(replacement.second.has_value());

    auto retired = cache.releaseInFlight(1,
                                         CacheBlockKind::COMPRESSED_KV,
                                         CacheBackingType::MEMORY,
                                         in_flight.block_index,
                                         in_flight.disk_slot,
                                         in_flight.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 11);
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
    EXPECT_FALSE(detached_child.has_value());
    detached_child = cache.releaseInFlight(2,
                                           CacheBlockKind::COMPRESSED_KV,
                                           CacheBackingType::MEMORY,
                                           child.block_index,
                                           child.disk_slot,
                                           child.generation);
    ASSERT_TRUE(detached_child.has_value());

    EXPECT_FALSE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_EQ(cache.cacheKeys(), (CacheKeysType{1}));
}

TEST(PrefixTreeMemoryBlockCacheTest, ParentBecomesEvictableAfterChildDetachEvenAfterTouchWhileNonLeaf) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);

    ASSERT_TRUE(cache.match(1, CacheBlockKind::COMPRESSED_KV).found);

    auto child = cache.matchAndMarkInFlight(2, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(child.found);
    auto detached_child = cache.detachIfMatch(2,
                                              CacheBlockKind::COMPRESSED_KV,
                                              CacheBackingType::MEMORY,
                                              child.block_index,
                                              child.disk_slot,
                                              child.generation);
    EXPECT_FALSE(detached_child.has_value());
    detached_child = cache.releaseInFlight(2,
                                           CacheBlockKind::COMPRESSED_KV,
                                           CacheBackingType::MEMORY,
                                           child.block_index,
                                           child.disk_slot,
                                           child.generation);
    ASSERT_TRUE(detached_child.has_value());

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(PrefixTreeMemoryBlockCacheTest, InFlightReleaseRestoresEvictability) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(in_flight.found);
    EXPECT_FALSE(cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV).has_value());

    cache.releaseInFlight(1,
                          CacheBlockKind::COMPRESSED_KV,
                          CacheBackingType::MEMORY,
                          in_flight.block_index,
                          in_flight.disk_slot,
                          in_flight.generation);
    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

}  // namespace rtp_llm::test
