#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

namespace rtp_llm::test {
namespace {

MemoryDiskBlockCache::CacheItem memoryItem(CacheKeyType key, BlockIdxType block, bool complete = true) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.is_complete  = complete;
    return item;
}

MemoryDiskBlockCache::CacheItem diskItem(CacheKeyType key, int32_t slot, bool complete = true) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_type = CacheBackingType::DISK;
    item.block_index  = NULL_BLOCK_IDX;
    item.disk_slot    = slot;
    item.is_complete  = complete;
    return item;
}

}  // namespace

TEST(MemoryDiskBlockCacheTest, ContainsAndMatchMemoryAndDisk) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));

    auto mem = cache.match(1);
    EXPECT_EQ(mem.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(mem.matched_index, 10);

    auto disk = cache.match(2);
    EXPECT_EQ(disk.backing_type, CacheBackingType::DISK);
    EXPECT_EQ(disk.disk_slot, 20);
}

TEST(MemoryDiskBlockCacheTest, SharedAccessSeqEvictsOldestAcrossBackings) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    ASSERT_FALSE(isNullBlockIdx(cache.match(1).matched_index));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_EQ(evicted->backing_type, CacheBackingType::DISK);
}

TEST(MemoryDiskBlockCacheTest, KindAwareEvictionOnlyPopsRequestedKind) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20, false)).first);
    ASSERT_TRUE(cache.putCommitted(memoryItem(3, 30, true)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);
    EXPECT_TRUE(evicted->is_complete);

    evicted = cache.popOldestEvictable(CacheBlockKind::INCOMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
    EXPECT_FALSE(evicted->is_complete);
}

TEST(MemoryDiskBlockCacheTest, KindAwareEvictionChoosesOldestAcrossMemoryAndDiskForSameKind) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, true)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20, true)).first);
    ASSERT_FALSE(isNullBlockIdx(cache.match(1).matched_index));

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_EQ(evicted->backing_type, CacheBackingType::DISK);
}

TEST(MemoryDiskBlockCacheTest, ContainsDoesNotUpdateRecency) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);
    ASSERT_TRUE(cache.contains(1));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, PartialToCompleteCanUpgradeAcrossBacking) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);

    auto [ok, popped] = cache.putCommitted(diskItem(1, 20, true));
    ASSERT_TRUE(ok);
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(popped->block_index, 10);

    auto match = cache.match(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::DISK);
    EXPECT_EQ(match.disk_slot, 20);
    EXPECT_TRUE(match.is_complete);
}

TEST(MemoryDiskBlockCacheTest, PartialToCompleteDoesNotReplaceInFlightItem) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);

    auto in_flight = cache.matchAndMarkInFlight(1);
    EXPECT_EQ(in_flight.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(in_flight.matched_index, 10);

    auto [ok, popped] = cache.putCommitted(diskItem(1, 20, true));
    EXPECT_FALSE(ok);
    EXPECT_FALSE(popped.has_value());

    auto match = cache.match(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(match.matched_index, 10);
    EXPECT_FALSE(match.is_complete);
}

TEST(MemoryDiskBlockCacheTest, InFlightEntryIsNotEvictable) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);
    ASSERT_TRUE(cache.markInFlight(1, CacheBackingType::MEMORY, 10, -1));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    EXPECT_FALSE(cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1, cache.match(1).generation).has_value());
    evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, MatchAndMarkInFlightPreventsEviction) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    auto match = cache.matchAndMarkInFlight(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(match.matched_index, 10);

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    EXPECT_FALSE(cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1, match.generation).has_value());
    evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, RemoveIfMatchChecksBackingAndSlot) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    EXPECT_FALSE(cache.removeIfMatch(2, CacheBackingType::DISK, NULL_BLOCK_IDX, 21).has_value());
    auto removed = cache.removeIfMatch(2, CacheBackingType::DISK, NULL_BLOCK_IDX, 20);
    ASSERT_TRUE(removed.has_value());
    EXPECT_FALSE(cache.contains(2));
}

TEST(MemoryDiskBlockCacheTest, RemovedBackingIsRetainedUntilEveryReaderReleases) {
    for (const auto& item : {memoryItem(1, 10), diskItem(1, 20)}) {
        MemoryDiskBlockCache cache;
        ASSERT_TRUE(cache.putCommitted(item).first);
        auto reader_a = cache.matchAndMarkInFlight(1);
        auto reader_b = cache.matchAndMarkInFlight(1);
        ASSERT_EQ(reader_a.generation, reader_b.generation);

        // B can be paused before acquiring its physical pool request ref when A completes.
        EXPECT_FALSE(cache.removeIfMatch(1, item.backing_type, item.block_index, item.disk_slot, reader_a.generation)
                         .has_value());
        EXPECT_FALSE(cache.contains(1));
        EXPECT_TRUE(cache.empty());
        EXPECT_EQ(cache.size(), 0u);
        EXPECT_TRUE(cache.cacheKeys().empty());
        EXPECT_TRUE(isNullBlockIdx(cache.matchAndMarkInFlight(1).matched_index));
        EXPECT_FALSE(cache.popOldestEvictable().has_value());
        EXPECT_FALSE(cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot, reader_a.generation)
                         .has_value());

        auto retired =
            cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot, reader_b.generation);
        ASSERT_TRUE(retired.has_value());
        EXPECT_EQ(retired->backing_type, item.backing_type);
        EXPECT_EQ(retired->block_index, item.block_index);
        EXPECT_EQ(retired->disk_slot, item.disk_slot);
        EXPECT_EQ(retired->in_flight_ref, 0u);
        // Physical reclamation must be returned exactly once.
        EXPECT_FALSE(cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot, reader_b.generation)
                         .has_value());
    }
}

TEST(MemoryDiskBlockCacheTest, OldReaderDoesNotRemoveOrReleaseNewGenerationInAnotherPool) {
    for (const auto backing : {CacheBackingType::MEMORY, CacheBackingType::DISK}) {
        MemoryDiskBlockCache cache;
        // Complete and incomplete pools can have the same numeric block/slot ID.
        auto old_item = backing == CacheBackingType::MEMORY ? memoryItem(1, 10, false) : diskItem(1, 10, false);
        ASSERT_TRUE(cache.putCommitted(old_item).first);
        auto old_reader = cache.matchAndMarkInFlight(1);
        ASSERT_FALSE(cache.removeIfMatch(1, backing, old_item.block_index, old_item.disk_slot, old_reader.generation)
                         .has_value());

        auto new_item        = old_item;
        new_item.is_complete = true;
        ASSERT_TRUE(cache.putCommitted(new_item).first);
        auto new_reader = cache.matchAndMarkInFlight(1);
        ASSERT_NE(old_reader.generation, new_reader.generation);
        EXPECT_FALSE(cache.removeIfMatch(1, backing, old_item.block_index, old_item.disk_slot, old_reader.generation)
                         .has_value());
        auto retired =
            cache.releaseInFlight(1, backing, old_item.block_index, old_item.disk_slot, old_reader.generation);
        ASSERT_TRUE(retired.has_value());
        EXPECT_FALSE(retired->is_complete);
        EXPECT_TRUE(cache.contains(1));
        EXPECT_FALSE(cache.popOldestEvictable().has_value());
        EXPECT_FALSE(cache.releaseInFlight(1, backing, new_item.block_index, new_item.disk_slot, new_reader.generation)
                         .has_value());
        auto evicted = cache.popOldestEvictable();
        ASSERT_TRUE(evicted.has_value());
        EXPECT_TRUE(evicted->is_complete);
    }
}

TEST(MemoryDiskBlockCacheTest, LegacyRemoveAlsoRetainsPinnedMemoryBacking) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    auto reader = cache.matchAndMarkInFlight(1);
    EXPECT_FALSE(cache.remove(1).has_value());
    EXPECT_FALSE(cache.contains(1));
    auto retired = cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1, reader.generation);
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->block_index, 10);
}

}  // namespace rtp_llm::test
