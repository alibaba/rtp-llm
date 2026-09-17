#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <future>

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

TEST(MemoryDiskBlockCacheTest, RemoveIfMatchDistinguishesCompleteAndIncompletePools) {
    for (auto backing : {CacheBackingType::MEMORY, CacheBackingType::DISK}) {
        MemoryDiskBlockCache cache;
        auto old_item = backing == CacheBackingType::MEMORY ? memoryItem(1, 10, false) : diskItem(1, 10, false);
        ASSERT_TRUE(cache.putCommitted(old_item).first);
        ASSERT_FALSE(cache.matchAndMarkInFlight(1, [](const auto&) { return true; }).is_complete);
        ASSERT_TRUE(cache.removeIfMatch(1, backing, old_item.block_index, old_item.disk_slot, false));

        // A new entry in the other pool can have the same index while an old read is still finishing.
        auto replacement        = old_item;
        replacement.is_complete = true;
        ASSERT_TRUE(cache.putCommitted(replacement).first);
        EXPECT_FALSE(cache.removeIfMatch(1, backing, old_item.block_index, old_item.disk_slot, false));
        EXPECT_TRUE(cache.match(1).is_complete);
        EXPECT_TRUE(cache.removeIfMatch(1, backing, replacement.block_index, replacement.disk_slot, true));
    }
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

    auto in_flight = cache.matchAndMarkInFlight(1, [](const auto&) { return true; });
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

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1, true);
    evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, MatchAndMarkInFlightPreventsEviction) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    auto match = cache.matchAndMarkInFlight(1, [](const auto&) { return true; });
    EXPECT_EQ(match.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(match.matched_index, 10);

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1, true);
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

TEST(MemoryDiskBlockCacheTest, RemovalCannotRecycleBackingBeforeReaderAcquiresReference) {
    for (auto backing : {CacheBackingType::MEMORY, CacheBackingType::DISK}) {
        SCOPED_TRACE(static_cast<int>(backing));
        MemoryDiskBlockCache cache;
        const auto item = backing == CacheBackingType::MEMORY ? memoryItem(1, 42) : diskItem(1, 42);
        ASSERT_TRUE(cache.putCommitted(item).first);

        // One cache reference, then reader A's request reference.
        std::atomic<int> references{1};
        std::atomic<int> contents{11};
        cache.matchAndMarkInFlight(1, [&](const auto&) {
            ++references;
            return true;
        });
        std::promise<void> pin_entered;
        std::promise<void> allow_pin;
        auto pin_ready = pin_entered.get_future();
        auto pin_allowed = allow_pin.get_future();
        auto reader_b = std::async(std::launch::async, [&] {
            return cache.matchAndMarkInFlight(1, [&](const auto&) {
                pin_entered.set_value();
                pin_allowed.wait();  // Stop B in the old match-to-reference window.
                ++references;
                return true;
            });
        });
        pin_ready.wait();
        std::promise<void> removal_started;
        auto removing = removal_started.get_future();
        auto finish_a = std::async(std::launch::async, [&] {
            removal_started.set_value();
            const auto removed = cache.removeIfMatch(1, backing, item.block_index, item.disk_slot, true);
            if (!removed) {
                return std::make_pair(false, false);
            }
            cache.releaseInFlight(1, backing, item.block_index, item.disk_slot, true);
            references -= 2;  // Drop the cache reference and A's request reference.
            int expected = 0;
            const bool reused = references.compare_exchange_strong(expected, 1);
            if (reused) {
                contents = 22;  // Writer C reuses block/slot 42 for another key.
            }
            return std::make_pair(true, reused);
        });
        removing.wait();
        EXPECT_EQ(finish_a.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
        allow_pin.set_value();
        const auto match = reader_b.get();
        const auto [removed, reused] = finish_a.get();
        EXPECT_TRUE(removed);
        EXPECT_FALSE(reused);
        EXPECT_EQ(match.backing_type, backing);
        EXPECT_EQ(match.matched_index, item.block_index);
        EXPECT_EQ(match.disk_slot, item.disk_slot);
        EXPECT_EQ(contents.load(), 11);
        EXPECT_FALSE(cache.contains(1));  // Invalidation/removal remains immediate after pinning.
        EXPECT_EQ(references.load(), 1);

        cache.releaseInFlight(1, backing, item.block_index, item.disk_slot, true);
        --references;
        int expected = 0;
        EXPECT_TRUE(references.compare_exchange_strong(expected, 1));
    }
}

TEST(MemoryDiskBlockCacheTest, FailedBackingAcquisitionDoesNotLeaveInFlightReference) {
    for (auto backing : {CacheBackingType::MEMORY, CacheBackingType::DISK}) {
        MemoryDiskBlockCache cache;
        const auto item = backing == CacheBackingType::MEMORY ? memoryItem(1, 42) : diskItem(1, 42);
        ASSERT_TRUE(cache.putCommitted(item).first);
        bool called = false;
        const auto match = cache.matchAndMarkInFlight(1, [&](const auto&) {
            called = true;
            return false;
        });
        EXPECT_TRUE(called);
        EXPECT_TRUE(isNullBlockIdx(match.matched_index));
        EXPECT_EQ(match.disk_slot, -1);
        auto evicted = cache.popOldestEvictable();
        ASSERT_TRUE(evicted.has_value());
        EXPECT_EQ(evicted->cache_key, 1);
        cache.matchAndMarkInFlight(1, [](const auto&) {
            ADD_FAILURE() << "A cache miss must not acquire a backing reference";
            return true;
        });
    }
}

TEST(MemoryDiskBlockCacheTest, OldReadReleaseDoesNotUnpinReplacementInAnotherPool) {
    for (auto backing : {CacheBackingType::MEMORY, CacheBackingType::DISK}) {
        MemoryDiskBlockCache cache;
        auto item = backing == CacheBackingType::MEMORY ? memoryItem(1, 42, false) : diskItem(1, 42, false);
        ASSERT_TRUE(cache.putCommitted(item).first);
        cache.matchAndMarkInFlight(1, [](const auto&) { return true; });
        ASSERT_TRUE(cache.removeIfMatch(1, backing, item.block_index, item.disk_slot, false));
        item.is_complete = true;
        ASSERT_TRUE(cache.putCommitted(item).first);
        cache.matchAndMarkInFlight(1, [](const auto&) { return true; });

        cache.releaseInFlight(1, backing, item.block_index, item.disk_slot, false);
        EXPECT_FALSE(cache.popOldestEvictable().has_value());
        cache.releaseInFlight(1, backing, item.block_index, item.disk_slot, true);
        EXPECT_TRUE(cache.popOldestEvictable().has_value());
    }
}

}  // namespace rtp_llm::test
