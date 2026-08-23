package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InflightEvictorTest {

    private static final class TestEntry implements InflightEvictor.TtlTracked {
        private final long createdAtMs;
        TestEntry(long createdAtMs) { this.createdAtMs = createdAtMs; }
        @Override public long createdAtMs() { return createdAtMs; }
    }

    @Test
    void evictExpiredRemovesOldEntries() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 100_000));  // 100s old
        map.put(2L, new TestEntry(now - 50_000));   // 50s old
        map.put(3L, new TestEntry(now));             // just now

        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        int evicted = evictor.evictExpired(60_000);  // TTL = 60s

        assertEquals(1, evicted);
        assertEquals(2, map.size());
        assertTrue(map.containsKey(2L));
        assertTrue(map.containsKey(3L));
    }

    @Test
    void evictExpiredEmptyMapReturnsZero() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        assertEquals(0, evictor.evictExpired(60_000));
    }

    @Test
    void evictExpiredAllFreshReturnsZero() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now));
        map.put(2L, new TestEntry(now - 10_000));

        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        assertEquals(0, evictor.evictExpired(60_000));
        assertEquals(2, map.size());
    }

    @Test
    void evictExpiredAllExpiredReturnsAll() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 200_000));
        map.put(2L, new TestEntry(now - 150_000));

        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        assertEquals(2, evictor.evictExpired(60_000));
        assertEquals(0, map.size());
    }

    @Test
    void evictExpiredCallsOnEvictCallback() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 100_000));
        map.put(2L, new TestEntry(now - 100_000));

        AtomicInteger callbackCount = new AtomicInteger(0);
        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, entry -> callbackCount.incrementAndGet());

        evictor.evictExpired(60_000);
        assertEquals(2, callbackCount.get());
    }

    @Test
    void keyAwareCallbackReceivesTheExactEvictedKey() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        map.put(41L, new TestEntry(System.currentTimeMillis() - 100_000));
        AtomicLong evictedKey = new AtomicLong();
        InflightEvictor<Long, TestEntry> evictor = InflightEvictor.withKeyCallback(
                map, (key, value) -> evictedKey.set(key));

        assertEquals(1, evictor.evictExpired(60_000));
        assertEquals(41L, evictedKey.get());
    }

    @Test
    void evictExpiredPartialExpiryCallsCallbackOnlyForEvicted() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 100_000));  // expired
        map.put(2L, new TestEntry(now));             // fresh

        AtomicInteger callbackCount = new AtomicInteger(0);
        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, entry -> callbackCount.incrementAndGet());

        int evicted = evictor.evictExpired(60_000);
        assertEquals(1, evicted);
        assertEquals(1, callbackCount.get());
    }

    @Test
    void evictExpiredNullOnEvictDoesNotThrow() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 100_000));

        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        assertEquals(1, evictor.evictExpired(60_000)); // should not throw NPE
    }

    @Test
    void evictExpiredLargeMap() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        for (long i = 0; i < 1000; i++) {
            map.put(i, new TestEntry(i % 2 == 0 ? now - 100_000 : now));
        }

        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(map, null);
        int evicted = evictor.evictExpired(60_000);

        assertEquals(500, evicted);
        assertEquals(500, map.size());
    }

    @Test
    void maxAgeMsEmptyMapReturnsZero() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        assertEquals(0, InflightEvictor.maxAgeMs(map, now));
    }

    @Test
    void maxAgeMsSingleEntryReturnsCorrectAge() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 5_000));
        assertEquals(5_000, InflightEvictor.maxAgeMs(map, now));
    }

    @Test
    void maxAgeMsReturnsAgeOfOldestEntry() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        map.put(1L, new TestEntry(now - 3_000));
        map.put(2L, new TestEntry(now - 10_000));
        map.put(3L, new TestEntry(now - 1_000));
        assertEquals(10_000, InflightEvictor.maxAgeMs(map, now));
    }

    @Test
    void maxAgeMsNeverReturnsNegative() {
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        long now = System.currentTimeMillis();
        // Entry created in the future (clock skew) should clamp to 0
        map.put(1L, new TestEntry(now + 5_000));
        assertEquals(0, InflightEvictor.maxAgeMs(map, now));
    }
}
