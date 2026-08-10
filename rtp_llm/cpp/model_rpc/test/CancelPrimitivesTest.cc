#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/schedulers/CancelIntentMap.h"

using namespace rtp_llm;

namespace {

// Write path: register + duplicate cancel overwrites the entry.
TEST(CancelIntentMapTest, RegisterAndDuplicateOverwrites) {
    CancelIntentMap map;
    EXPECT_TRUE(map.empty());

    map.registerCancel(/*request_id=*/1, ErrorCode::CANCELLED, /*now_ms=*/1000);
    EXPECT_EQ(map.size(), 1u);
    auto entry = map.match(1);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::CANCELLED);
    EXPECT_EQ(entry->arrival_time_ms, 1000);

    // Duplicate cancel for the same request overwrites in place.
    map.registerCancel(1, ErrorCode::PRIORITY_PREEMPTED, 2000);
    EXPECT_EQ(map.size(), 1u);
    entry = map.match(1);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_EQ(entry->arrival_time_ms, 2000);
}

// R1 consumption: act (reject) first, erase after.
TEST(CancelIntentMapTest, R1ConsumeOnMatch) {
    CancelIntentMap map;
    map.registerCancel(42, ErrorCode::PRIORITY_PREEMPTED, 1000);

    auto entry = map.match(42);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::PRIORITY_PREEMPTED);
    map.erase(42);

    EXPECT_FALSE(map.match(42).has_value());
    EXPECT_TRUE(map.empty());
}

// R2 consumption is idempotent: after act + erase, re-consumption is a no-op.
TEST(CancelIntentMapTest, R2ConsumeIsIdempotent) {
    CancelIntentMap map;
    map.registerCancel(7, ErrorCode::CANCELLED, 1000);

    // First schedule round: hit, stop the stream, then erase.
    ASSERT_TRUE(map.match(7).has_value());
    map.erase(7);

    // Later rounds and double-erase see nothing and change nothing.
    EXPECT_FALSE(map.match(7).has_value());
    map.erase(7);
    EXPECT_TRUE(map.empty());
}

// R3: entries older than kTtlMs are swept; fresh ones survive.
TEST(CancelIntentMapTest, TtlSweepDropsExpiredEntries) {
    CancelIntentMap map;
    const int64_t   now = 100000;
    map.registerCancel(1, ErrorCode::CANCELLED, now - CancelIntentMap::kTtlMs - 1);
    map.registerCancel(2, ErrorCode::CANCELLED, now);

    map.sweepExpired(now);
    EXPECT_FALSE(map.match(1).has_value());
    EXPECT_TRUE(map.match(2).has_value());
    EXPECT_EQ(map.size(), 1u);

    // Exactly-at-TTL is kept (strict '>' comparison).
    map.registerCancel(3, ErrorCode::CANCELLED, now - CancelIntentMap::kTtlMs);
    map.sweepExpired(now);
    EXPECT_TRUE(map.match(3).has_value());
}

// Capacity cap: inserting past kMaxEntries evicts the oldest entry.
TEST(CancelIntentMapTest, CapacityEvictsOldestEntry) {
    CancelIntentMap map;
    for (size_t i = 0; i < CancelIntentMap::kMaxEntries; ++i) {
        map.registerCancel(static_cast<int64_t>(i), ErrorCode::CANCELLED, static_cast<int64_t>(i));
    }
    EXPECT_EQ(map.size(), CancelIntentMap::kMaxEntries);

    // request 0 carries the oldest arrival_time -> evicted by the overflow insert.
    map.registerCancel(static_cast<int64_t>(CancelIntentMap::kMaxEntries), ErrorCode::CANCELLED, 999999);
    EXPECT_EQ(map.size(), CancelIntentMap::kMaxEntries);
    EXPECT_FALSE(map.match(0).has_value());
    EXPECT_TRUE(map.match(static_cast<int64_t>(CancelIntentMap::kMaxEntries)).has_value());

    // Overwriting an existing key at capacity must not evict anyone.
    map.registerCancel(1, ErrorCode::PRIORITY_PREEMPTED, 1000000);
    EXPECT_EQ(map.size(), CancelIntentMap::kMaxEntries);
    auto entry = map.match(1);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::PRIORITY_PREEMPTED);
}

// tryConsume on a hit: the entry is removed and returned in one atomic step.
TEST(CancelIntentMapTest, TryConsumeRemovesEntryOnHit) {
    CancelIntentMap map;
    map.registerCancel(11, ErrorCode::PRIORITY_PREEMPTED, 1000);

    auto entry = map.tryConsume(11);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_TRUE(map.empty());

    // Re-consumption after the erase is a no-op.
    EXPECT_FALSE(map.tryConsume(11).has_value());
}

// tryConsume on a missing key: nullopt, map untouched.
TEST(CancelIntentMapTest, TryConsumeMissingKeyLeavesMapUntouched) {
    CancelIntentMap map;
    map.registerCancel(12, ErrorCode::CANCELLED, 1000);

    EXPECT_FALSE(map.tryConsume(999).has_value());
    EXPECT_EQ(map.size(), 1u);
    EXPECT_TRUE(map.match(12).has_value());
}

// Concurrency semantics: a duplicate cancel that overwrites the entry between
// a consumer's match() and its removal is still consumed safely — tryConsume
// (atomic match + erase) returns the latest intent exactly once and a repeat
// consumption is a clean miss, never a crash or a dangling entry.
TEST(CancelIntentMapTest, TryConsumeAfterOverwriteConsumesLatestIntentOnce) {
    CancelIntentMap map;
    map.registerCancel(5, ErrorCode::CANCELLED, 1000);

    // Consumer read the old intent (stand-alone match, no erase yet)...
    ASSERT_TRUE(map.match(5).has_value());
    // ...then the Cancel RPC overwrites it with a new attribution.
    map.registerCancel(5, ErrorCode::PRIORITY_PREEMPTED, 2000);

    // The consumption claims the latest intent atomically.
    auto entry = map.tryConsume(5);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->terminal_code, ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_TRUE(map.empty());

    // A racing second consumer sees a clean miss.
    EXPECT_FALSE(map.tryConsume(5).has_value());
}

}  // namespace
