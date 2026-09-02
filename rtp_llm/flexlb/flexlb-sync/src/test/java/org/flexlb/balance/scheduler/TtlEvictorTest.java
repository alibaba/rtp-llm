package org.flexlb.balance.scheduler;

import org.flexlb.balance.execution.TtlEvictor;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Canonical replacement for the physically deleted scheduler InflightEvictor. */
class TtlEvictorTest {

    private record TestEntry(long createdAtMs) implements TtlEvictor.TtlTracked {
    }

    @Test
    void removesOnlyExpiredEntriesApprovedByTheOwnershipPredicate() {
        long now = System.currentTimeMillis();
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        map.put(1L, new TestEntry(now - 100_000L));
        map.put(2L, new TestEntry(now - 100_000L));
        map.put(3L, new TestEntry(now));
        TtlEvictor<Long, TestEntry> evictor =
                TtlEvictor.withKeyCallback(map, null);

        assertEquals(1, evictor.evictExpired(60_000L, key -> key != 2L));
        assertFalse(map.containsKey(1L));
        assertTrue(map.containsKey(2L),
                "a stronger exact owner must protect an otherwise stale entry");
        assertTrue(map.containsKey(3L));
    }

    @Test
    void emptyFreshAndFullyExpiredMapsReturnExactCounts() {
        Map<Long, TestEntry> empty = new ConcurrentHashMap<>();
        assertEquals(0, TtlEvictor.<Long, TestEntry>withKeyCallback(empty, null)
                .evictExpired(60_000L, ignored -> true));

        long now = System.currentTimeMillis();
        Map<Long, TestEntry> fresh = new ConcurrentHashMap<>();
        fresh.put(1L, new TestEntry(now));
        fresh.put(2L, new TestEntry(now - 10_000L));
        assertEquals(0, TtlEvictor.<Long, TestEntry>withKeyCallback(fresh, null)
                .evictExpired(60_000L, ignored -> true));
        assertEquals(2, fresh.size());

        Map<Long, TestEntry> expired = new ConcurrentHashMap<>();
        expired.put(1L, new TestEntry(now - 200_000L));
        expired.put(2L, new TestEntry(now - 150_000L));
        assertEquals(2, TtlEvictor.<Long, TestEntry>withKeyCallback(expired, null)
                .evictExpired(60_000L, ignored -> true));
        assertTrue(expired.isEmpty());
    }

    @Test
    void keyAwareCallbackRunsOnceForEachExactRemoval() {
        long now = System.currentTimeMillis();
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        map.put(41L, new TestEntry(now - 100_000L));
        map.put(42L, new TestEntry(now));
        AtomicInteger callbackCount = new AtomicInteger();
        AtomicLong evictedKey = new AtomicLong();
        TtlEvictor<Long, TestEntry> evictor = TtlEvictor.withKeyCallback(
                map,
                (key, value) -> {
                    callbackCount.incrementAndGet();
                    evictedKey.set(key);
                });

        assertEquals(1, evictor.evictExpired(60_000L, ignored -> true));
        assertEquals(1, callbackCount.get());
        assertEquals(41L, evictedKey.get());
    }

    @Test
    void largeMapEvictionPreservesEveryFreshIdentity() {
        long now = System.currentTimeMillis();
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        for (long requestId = 0L; requestId < 1_000L; requestId++) {
            map.put(requestId, new TestEntry(
                    requestId % 2L == 0L ? now - 100_000L : now));
        }

        int evicted = TtlEvictor.<Long, TestEntry>withKeyCallback(map, null)
                .evictExpired(60_000L, ignored -> true);

        assertEquals(500, evicted);
        assertEquals(500, map.size());
        assertTrue(map.keySet().stream().allMatch(id -> id % 2L == 1L));
    }

    @Test
    void maxAgeUsesOldestEntryAndClampsClockSkew() {
        long now = System.currentTimeMillis();
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        assertEquals(0L, TtlEvictor.maxAgeMs(map, now));

        map.put(1L, new TestEntry(now - 3_000L));
        map.put(2L, new TestEntry(now - 10_000L));
        map.put(3L, new TestEntry(now - 1_000L));
        assertEquals(10_000L, TtlEvictor.maxAgeMs(map, now));

        map.clear();
        map.put(4L, new TestEntry(now + 5_000L));
        assertEquals(0L, TtlEvictor.maxAgeMs(map, now));
    }
}
