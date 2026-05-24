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

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1);
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

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1);
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

}  // namespace rtp_llm::test
