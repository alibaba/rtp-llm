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

MemoryDiskBlockCache::CacheItem splitMemoryItem(CacheKeyType key, BlockIdxType block, CacheBackingRole role) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_role = role;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.is_complete  = role == CacheBackingRole::TAIL;
    return item;
}

MemoryDiskBlockCache::CacheItem splitDiskItem(CacheKeyType key, int32_t slot, CacheBackingRole role) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_role = role;
    item.backing_type = CacheBackingType::DISK;
    item.block_index  = NULL_BLOCK_IDX;
    item.disk_slot    = slot;
    item.is_complete  = role == CacheBackingRole::TAIL;
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

TEST(MemoryDiskBlockCacheTest, SplitKvOnlyThenTailUpgrade) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);

    auto match = cache.match(1);
    EXPECT_TRUE(match.has_kv);
    EXPECT_FALSE(match.has_tail);
    EXPECT_FALSE(match.is_complete);
    EXPECT_EQ(match.matched_index, 10);

    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);
    match = cache.match(1);
    EXPECT_TRUE(match.has_kv);
    EXPECT_TRUE(match.has_tail);
    EXPECT_TRUE(match.is_complete);
    EXPECT_EQ(match.matched_index, 10);
    EXPECT_EQ(match.tail_matched_index, 11);
}

TEST(MemoryDiskBlockCacheTest, SplitTailEvictionKeepsKv) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);

    auto evicted = cache.popOldestEvictableBackings(CacheBlockKind::COMPLETE);
    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted[0].backing_role, CacheBackingRole::TAIL);
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::KV));
    EXPECT_FALSE(cache.contains(1, CacheBackingRole::TAIL));
}

TEST(MemoryDiskBlockCacheTest, SplitKvEvictionAlsoRemovesTail) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);

    auto evicted = cache.popOldestEvictableBackings(CacheBlockKind::INCOMPLETE);
    ASSERT_EQ(evicted.size(), 2u);
    EXPECT_FALSE(cache.contains(1));
}

TEST(MemoryDiskBlockCacheTest, NoArgEvictionSkipsKvWithTailAndFindsNextCandidate) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);
    ASSERT_TRUE(cache.markInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1));
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(2, 20, CacheBackingRole::KV)).first);

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_EQ(evicted->backing_role, CacheBackingRole::KV);
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::KV));
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::TAIL));
    cache.releaseInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1);
}

TEST(MemoryDiskBlockCacheTest, NoArgEvictionCanDemoteByEvictingTail) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
    EXPECT_EQ(evicted->backing_role, CacheBackingRole::TAIL);
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::KV));
    EXPECT_FALSE(cache.contains(1, CacheBackingRole::TAIL));
}

TEST(MemoryDiskBlockCacheTest, NoArgEvictionReturnsEmptyWhenOnlyTailIsInFlight) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);
    ASSERT_TRUE(cache.markInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1));

    auto evicted = cache.popOldestEvictable();
    EXPECT_FALSE(evicted.has_value());
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::KV));
    EXPECT_TRUE(cache.contains(1, CacheBackingRole::TAIL));
    cache.releaseInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1);
}

TEST(MemoryDiskBlockCacheTest, SplitTailInFlightPreventsDemotion) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 10, CacheBackingRole::KV)).first);
    ASSERT_TRUE(cache.putCommitted(splitMemoryItem(1, 11, CacheBackingRole::TAIL)).first);
    ASSERT_TRUE(cache.markInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1));

    auto evicted = cache.popOldestEvictableBackings(CacheBlockKind::COMPLETE);
    EXPECT_TRUE(evicted.empty());

    cache.releaseInFlight(1, CacheBackingRole::TAIL, CacheBackingType::MEMORY, 11, -1);
    evicted = cache.popOldestEvictableBackings(CacheBlockKind::COMPLETE);
    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted[0].backing_role, CacheBackingRole::TAIL);
}

TEST(MemoryDiskBlockCacheTest, SplitTailAddFailsWhenKvInFlight) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(splitDiskItem(1, 10, CacheBackingRole::KV)).first);
    auto kv = cache.matchAndMarkInFlight(1, CacheBackingRole::KV);
    ASSERT_TRUE(kv.found);

    auto [ok, popped] = cache.putCommitted(splitDiskItem(1, 11, CacheBackingRole::TAIL));
    EXPECT_FALSE(ok);
    EXPECT_FALSE(popped.has_value());
    EXPECT_FALSE(cache.contains(1, CacheBackingRole::TAIL));

    cache.releaseInFlight(1, CacheBackingRole::KV, CacheBackingType::DISK, NULL_BLOCK_IDX, 10);
    EXPECT_TRUE(cache.putCommitted(splitDiskItem(1, 11, CacheBackingRole::TAIL)).first);
}

}  // namespace rtp_llm::test
