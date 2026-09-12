#include "gtest/gtest.h"

#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/DSV41CacheState.h"
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

PrefixTreeMemoryBlockCache::CacheItem item(CacheKeyType           key,
                                           CacheBlockKind         kind,
                                           BlockIdxType           block,
                                           std::vector<uint8_t>   slot_valid_mask = {},
                                           bool                   is_resident = false) {
    PrefixTreeMemoryBlockCache::CacheItem item;
    item.cache_key    = key;
    item.kind         = kind;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.block_size   = 1024;
    item.is_resident  = is_resident;
    item.slot_valid_mask = std::move(slot_valid_mask);
    return item;
}

PrefixTreeMemoryBlockCache::CacheItem diskItem(CacheKeyType         key,
                                               CacheBlockKind       kind,
                                               int32_t              disk_slot,
                                               std::vector<uint8_t> slot_valid_mask = {}) {
    auto result          = item(key, kind, NULL_BLOCK_IDX, std::move(slot_valid_mask));
    result.backing_type  = CacheBackingType::DISK;
    result.block_index   = NULL_BLOCK_IDX;
    result.disk_slot     = disk_slot;
    return result;
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

TEST(PrefixTreeMemoryBlockCacheTest, SameSlotMaskDuplicateDoesNotReplaceBacking) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto duplicate = cache.putCommitted(1,
                                        rootDep(),
                                        item(1,
                                             CacheBlockKind::STATE_SWA_KV,
                                             12,
                                             /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}));
    EXPECT_FALSE(duplicate.first);
    EXPECT_FALSE(duplicate.second.has_value());
    EXPECT_EQ(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0}).block_index, 11);
}

TEST(PrefixTreeMemoryBlockCacheTest, MarkInFlightRejectsNonCoveringMask) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0});
    EXPECT_FALSE(in_flight.found);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->block_index, 11);
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

TEST(PrefixTreeMemoryBlockCacheTest, RetiredItemRequiresAllInFlightReleases) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{0, 1, 0}))
                    .first);

    auto first = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0});
    auto second = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{0, 1, 0});
    ASSERT_TRUE(first.found);
    ASSERT_TRUE(second.found);

    auto replacement = cache.putCommitted(1,
                                          rootDep(),
                                          item(1,
                                               CacheBlockKind::STATE_SWA_KV,
                                               12,
                                               /*slot_valid_mask=*/std::vector<uint8_t>{1, 1, 0}));
    ASSERT_TRUE(replacement.first);
    EXPECT_FALSE(replacement.second.has_value());

    auto retired = cache.releaseInFlight(1,
                                         CacheBlockKind::STATE_SWA_KV,
                                         CacheBackingType::MEMORY,
                                         first.block_index,
                                         first.disk_slot,
                                         first.generation);
    EXPECT_FALSE(retired.has_value());

    retired = cache.releaseInFlight(1,
                                    CacheBlockKind::STATE_SWA_KV,
                                    CacheBackingType::MEMORY,
                                    second.block_index,
                                    second.disk_slot,
                                    second.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 11);
    EXPECT_EQ(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0}).block_index, 12);
}

TEST(PrefixTreeMemoryBlockCacheTest, MultipleRetiredItemsReleaseOutOfOrder) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        11,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{1, 0, 0}))
                    .first);
    auto old_in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 0, 0});
    ASSERT_TRUE(old_in_flight.found);

    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        12,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{1, 1, 0}))
                    .first);
    auto middle_in_flight = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 0});
    ASSERT_TRUE(middle_in_flight.found);

    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(),
                                   item(1,
                                        CacheBlockKind::STATE_SWA_KV,
                                        13,
                                        /*slot_valid_mask=*/std::vector<uint8_t>{1, 1, 1}))
                    .first);

    auto retired = cache.releaseInFlight(1,
                                         CacheBlockKind::STATE_SWA_KV,
                                         CacheBackingType::MEMORY,
                                         middle_in_flight.block_index,
                                         middle_in_flight.disk_slot,
                                         middle_in_flight.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 12);

    retired = cache.releaseInFlight(1,
                                    CacheBlockKind::STATE_SWA_KV,
                                    CacheBackingType::MEMORY,
                                    old_in_flight.block_index,
                                    old_in_flight.disk_slot,
                                    old_in_flight.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 11);
    EXPECT_EQ(cache.match(1, CacheBlockKind::STATE_SWA_KV, std::vector<uint8_t>{1, 1, 1}).block_index, 13);
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

TEST(PrefixTreeMemoryBlockCacheTest, ReparentMovesSubtreeRefFromOldToNewParent) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(3, rootDep(0), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);

    auto reparent = cache.putCommitted(2, childDep(3, 1), item(2, CacheBlockKind::COMPRESSED_KV, 14));
    EXPECT_FALSE(reparent.first);
    EXPECT_FALSE(reparent.second.has_value());

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);
}

TEST(PrefixTreeMemoryBlockCacheTest, MultipleOrphanChildrenAttachOnParentInsert) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(1, 2), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(cache.contains(3, CacheBlockKind::COMPRESSED_KV));

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(PrefixTreeMemoryBlockCacheTest, ReparentPendingOrphanMovesPendingEntry) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    auto reparent = cache.putCommitted(2, childDep(3, 1), item(2, CacheBlockKind::COMPRESSED_KV, 14));
    EXPECT_FALSE(reparent.first);
    ASSERT_TRUE(cache.putCommitted(3, rootDep(0), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(PrefixTreeMemoryBlockCacheTest, BranchParentBecomesEvictableAfterAllChildrenGone) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(1, 2), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);

    evicted = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(PrefixTreeMemoryBlockCacheTest, ResidentItemIsMatchableButNeverEvictable) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1,
                                   rootDep(0),
                                   item(1,
                                        CacheBlockKind::COMPRESSED_KV,
                                        11,
                                        /*slot_valid_mask=*/{},
                                        /*is_resident=*/true))
                    .first);

    EXPECT_TRUE(cache.match(1, CacheBlockKind::COMPRESSED_KV).found);
    EXPECT_FALSE(cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV).has_value());
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
    auto status_keys = cache.cacheKeysUnorderedForStatus();
    std::sort(status_keys.begin(), status_keys.end());
    EXPECT_EQ(status_keys, (CacheKeysType{1}));
}

TEST(PrefixTreeMemoryBlockCacheTest, StatusCacheKeysAreUnorderedAndDeduplicated) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(0), item(1, CacheBlockKind::STATE_SWA_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 21)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(2, 2), item(3, CacheBlockKind::STATE_SWA_KV, 31)).first);

    auto status_keys = cache.cacheKeysUnorderedForStatus();
    std::sort(status_keys.begin(), status_keys.end());

    EXPECT_EQ(status_keys, (CacheKeysType{1, 2, 3}));
    EXPECT_EQ(status_keys.size(), cache.cacheKeys().size());
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

TEST(PrefixTreeMemoryBlockCacheTest, DiskBackingMatchesAndEvictsByBacking) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), diskItem(1, CacheBlockKind::COMPRESSED_KV, 7)).first);
    ASSERT_TRUE(cache.putCommitted(2, rootDep(), item(2, CacheBlockKind::COMPRESSED_KV, 22)).first);

    auto matched = cache.matchAndMarkInFlight(1, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.backing_type, CacheBackingType::DISK);
    EXPECT_EQ(matched.disk_slot, 7);
    EXPECT_FALSE(cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::DISK).has_value());

    auto released = cache.releaseInFlight(1,
                                          CacheBlockKind::COMPRESSED_KV,
                                          CacheBackingType::DISK,
                                          matched.block_index,
                                          matched.disk_slot,
                                          matched.generation);
    EXPECT_FALSE(released.has_value());

    auto evicted_disk = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::DISK);
    ASSERT_TRUE(evicted_disk.has_value());
    EXPECT_EQ(evicted_disk->cache_key, 1);
    EXPECT_EQ(evicted_disk->disk_slot, 7);

    auto evicted_mem = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_TRUE(evicted_mem.has_value());
    EXPECT_EQ(evicted_mem->cache_key, 2);
    EXPECT_EQ(evicted_mem->block_index, 22);
}

TEST(PrefixTreeMemoryBlockCacheTest, StateIndependentEvictionDropsDeepestNonTailState) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 101)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::STATE_SWA_KV, 102)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(2, 2), item(3, CacheBlockKind::COMPRESSED_KV, 13)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(2, 2), item(3, CacheBlockKind::STATE_SWA_KV, 103)).first);

    auto evicted = cache.popOldestStateOrChainEvictable(CacheBackingType::MEMORY);

    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted[0].cache_key, 2);
    EXPECT_EQ(evicted[0].kind, CacheBlockKind::STATE_SWA_KV);
    EXPECT_EQ(evicted[0].block_index, 102);
    EXPECT_TRUE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
    EXPECT_FALSE(cache.contains(2, CacheBlockKind::STATE_SWA_KV));
    EXPECT_TRUE(cache.contains(3, CacheBlockKind::STATE_SWA_KV));
}

TEST(PrefixTreeMemoryBlockCacheTest, StateIndependentEvictionFallsBackToWholeChain) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::STATE_SWA_KV, 102)).first);

    auto evicted = cache.popOldestStateOrChainEvictable(CacheBackingType::MEMORY);

    ASSERT_EQ(evicted.size(), 3u);
    EXPECT_EQ(evicted[0].cache_key, 2);
    EXPECT_EQ(evicted[0].kind, CacheBlockKind::COMPRESSED_KV);
    EXPECT_EQ(evicted[1].cache_key, 2);
    EXPECT_EQ(evicted[1].kind, CacheBlockKind::STATE_SWA_KV);
    EXPECT_EQ(evicted[2].cache_key, 1);
    EXPECT_EQ(evicted[2].kind, CacheBlockKind::COMPRESSED_KV);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionDoesNotRemoveSiblingBranchesOrSharedParent) {
    PrefixTreeMemoryBlockCache cache;
    for (auto key : {1, 2, 3}) {
        const auto dep = key == 1 ? rootDep() : childDep(1, 1);
        ASSERT_TRUE(cache.putCommitted(key, dep, item(key, CacheBlockKind::COMPRESSED_KV, key + 10)).first);
        ASSERT_TRUE(cache.putCommitted(key, dep, item(key, CacheBlockKind::STATE_SWA_KV, key + 100)).first);
    }

    const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 2u);
    EXPECT_EQ(evicted[0].cache_key, 2);
    EXPECT_EQ(evicted[0].kind, CacheBlockKind::COMPRESSED_KV);
    EXPECT_EQ(evicted[1].cache_key, 2);
    EXPECT_EQ(evicted[1].kind, CacheBlockKind::STATE_SWA_KV);
    for (auto kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
        EXPECT_FALSE(cache.contains(2, kind));
        EXPECT_TRUE(cache.contains(1, kind));
        EXPECT_TRUE(cache.contains(3, kind));
    }

    const auto sibling = cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(sibling.size(), 2u);
    EXPECT_EQ(sibling[0].cache_key, 3);
    EXPECT_EQ(sibling[1].cache_key, 3);
    const auto parent = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(parent.size(), 2u);
    EXPECT_EQ(parent[0].cache_key, 1);
    EXPECT_EQ(parent[1].cache_key, 1);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionProtectsEitherResidentKind) {
    for (auto resident_kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
        PrefixTreeMemoryBlockCache cache;
        for (auto kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
            ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, kind, 11, {}, kind == resident_kind)).first);
        }
        EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY).empty());
        EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY).empty());
        EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
        EXPECT_TRUE(cache.contains(1, CacheBlockKind::STATE_SWA_KV));
    }
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionProtectsConcurrentReaderWithoutPinningSibling) {
    for (auto read_kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
        PrefixTreeMemoryBlockCache cache;
        for (auto key : {1, 2, 3}) {
            const auto dep = key == 1 ? rootDep() : childDep(1, 1);
            ASSERT_TRUE(cache.putCommitted(key, dep, item(key, CacheBlockKind::COMPRESSED_KV, key + 10)).first);
            ASSERT_TRUE(cache.putCommitted(key, dep, item(key, CacheBlockKind::STATE_SWA_KV, key + 100)).first);
        }
        std::promise<PrefixTreeMemoryBlockCache::MatchResult> pinned;
        std::promise<void>                                    release;
        auto                                                  allow_release = release.get_future();
        std::thread                                           reader([&] {
            const auto held = cache.matchAndMarkInFlight(2, read_kind);
            pinned.set_value(held);
            allow_release.wait();
            cache.releaseInFlight(2, read_kind, held.backing_type, held.block_index, held.disk_slot, held.generation);
        });
        const auto                                            held = pinned.get_future().get();
        EXPECT_TRUE(held.found);
        const auto sibling = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
        EXPECT_EQ(sibling.size(), 2u);
        for (const auto& evicted : sibling) {
            EXPECT_EQ(evicted.cache_key, 3);
        }
        EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY).empty());
        EXPECT_TRUE(cache.contains(2, CacheBlockKind::COMPRESSED_KV));
        EXPECT_TRUE(cache.contains(2, CacheBlockKind::STATE_SWA_KV));
        release.set_value();
        reader.join();

        const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
        ASSERT_EQ(evicted.size(), 2u);
        EXPECT_EQ(evicted[0].cache_key, 2);
        EXPECT_EQ(evicted[1].cache_key, 2);
        EXPECT_TRUE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    }
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionWaitsForEveryReaderOfRetiredGeneration) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12, {1, 0})).first);
    const auto first  = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV);
    const auto second = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(first.found);
    ASSERT_TRUE(second.found);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 13, {1, 1})).first);
    EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY).empty());

    EXPECT_FALSE(cache
                     .releaseInFlight(1,
                                      CacheBlockKind::STATE_SWA_KV,
                                      first.backing_type,
                                      first.block_index,
                                      first.disk_slot,
                                      first.generation)
                     .has_value());
    EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY).empty());
    const auto retired = cache.releaseInFlight(
        1, CacheBlockKind::STATE_SWA_KV, second.backing_type, second.block_index, second.disk_slot, second.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 12);
    const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 2u);
    EXPECT_EQ(evicted[0].block_index, 11);
    EXPECT_EQ(evicted[1].block_index, 13);
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionProtectsDetachedRetiredKind) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12)).first);
    const auto held = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(held.found);
    EXPECT_FALSE(
        cache
            .detachIfMatch(
                1, CacheBlockKind::STATE_SWA_KV, held.backing_type, held.block_index, held.disk_slot, held.generation)
            .has_value());
    EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY).empty());
    ASSERT_TRUE(
        cache
            .releaseInFlight(
                1, CacheBlockKind::STATE_SWA_KV, held.backing_type, held.block_index, held.disk_slot, held.generation)
            .has_value());
    const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted[0].block_index, 11);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(PrefixTreeMemoryBlockCacheTest, StatePressureEvictsBoundaryPairAndPreservesLaterKvBlocks) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::COMPRESSED_KV, 11)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12)).first);
    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 22)).first);
    ASSERT_TRUE(cache.putCommitted(3, childDep(1, 1), item(3, CacheBlockKind::COMPRESSED_KV, 33)).first);

    const auto boundary = cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(boundary.size(), 2u);
    EXPECT_EQ(boundary[0].cache_key, 1);
    EXPECT_EQ(boundary[0].kind, CacheBlockKind::COMPRESSED_KV);
    EXPECT_EQ(boundary[1].cache_key, 1);
    EXPECT_EQ(boundary[1].kind, CacheBlockKind::STATE_SWA_KV);
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_FALSE(cache.contains(1, CacheBlockKind::STATE_SWA_KV));
    EXPECT_TRUE(cache.contains(3, CacheBlockKind::COMPRESSED_KV));

    const auto later = cache.matchAndMarkInFlight(2, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(later.found);
    EXPECT_EQ(later.block_index, 22);
    EXPECT_EQ(later.recovery_metadata, nullptr);
    const auto sibling = cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(sibling.size(), 1u);
    EXPECT_EQ(sibling[0].cache_key, 3);
    EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY).empty());
    EXPECT_FALSE(cache
                     .releaseInFlight(2,
                                      CacheBlockKind::COMPRESSED_KV,
                                      later.backing_type,
                                      later.block_index,
                                      later.disk_slot,
                                      later.generation)
                     .has_value());
    const auto final_block = cache.popOldestEvictable(CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(final_block.has_value());
    EXPECT_EQ(final_block->cache_key, 2);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(PrefixTreeMemoryBlockCacheTest, JointEvictionFiltersRequestedBackingAndKeepsPairsTogether) {
    PrefixTreeMemoryBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), diskItem(1, CacheBlockKind::COMPRESSED_KV, 7)).first);
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), item(1, CacheBlockKind::STATE_SWA_KV, 12)).first);
    EXPECT_TRUE(cache.popOldestJointEvictable(CacheBlockKind::COMPRESSED_KV, CacheBackingType::MEMORY).empty());
    const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 2u);
    EXPECT_EQ(evicted[0].cache_key, 1);
    EXPECT_EQ(evicted[0].backing_type, CacheBackingType::DISK);
    EXPECT_EQ(evicted[0].disk_slot, 7);
    EXPECT_EQ(evicted[1].cache_key, 1);
    EXPECT_EQ(evicted[1].backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(PrefixTreeMemoryBlockCacheTest, RecoveryMetadataFollowsBackingGenerationAndRetiredReaders) {
    PrefixTreeMemoryBlockCache cache;
    auto                       old_metadata = std::make_shared<DSV41CheckpointMetadata>();
    old_metadata->materialized_end          = 128;
    auto old_item                           = item(1, CacheBlockKind::STATE_SWA_KV, 11, {1, 0});
    old_item.recovery_metadata              = old_metadata;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), old_item).first);
    const auto held = cache.matchAndMarkInFlight(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(held.found);
    EXPECT_EQ(held.recovery_metadata, old_metadata);

    auto new_metadata              = std::make_shared<DSV41CheckpointMetadata>();
    new_metadata->materialized_end = 256;
    auto new_item                  = item(1, CacheBlockKind::STATE_SWA_KV, 12, {1, 1});
    new_item.recovery_metadata     = new_metadata;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), new_item).first);
    const auto current = cache.match(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(current.found);
    EXPECT_NE(current.generation, held.generation);
    EXPECT_EQ(current.recovery_metadata, new_metadata);
    EXPECT_EQ(held.recovery_metadata->materialized_end, 128);

    const auto retired = cache.releaseInFlight(
        1, CacheBlockKind::STATE_SWA_KV, held.backing_type, held.block_index, held.disk_slot, held.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->recovery_metadata, old_metadata);
    const auto evicted = cache.popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted[0].recovery_metadata, new_metadata);
}

TEST(PrefixTreeMemoryBlockCacheTest, DuplicateBackingCannotRelabelRecoveryMetadata) {
    PrefixTreeMemoryBlockCache cache;
    auto                       metadata = std::make_shared<DSV41CheckpointMetadata>();
    auto                       original = item(1, CacheBlockKind::STATE_SWA_KV, 11, {1});
    original.recovery_metadata          = metadata;
    ASSERT_TRUE(cache.putCommitted(1, rootDep(), original).first);
    auto duplicate              = item(1, CacheBlockKind::STATE_SWA_KV, 12, {1});
    duplicate.recovery_metadata = std::make_shared<DSV41CheckpointMetadata>();
    EXPECT_FALSE(cache.putCommitted(1, rootDep(), duplicate).first);
    const auto matched = cache.match(1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(matched.found);
    EXPECT_EQ(matched.block_index, 11);
    EXPECT_EQ(matched.recovery_metadata, metadata);

    ASSERT_TRUE(cache.putCommitted(2, childDep(1, 1), item(2, CacheBlockKind::COMPRESSED_KV, 22)).first);
    const auto kv_only = cache.match(2, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(kv_only.found);
    EXPECT_EQ(kv_only.recovery_metadata, nullptr);
}

}  // namespace rtp_llm::test
