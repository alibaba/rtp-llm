package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InflightEvictorTest {

    private static class TestEntry implements InflightEvictor.TtlTracked {
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
    void evictionChecksTtlWhileHoldingEntryMonitor() {
        AtomicInteger lockChecks = new AtomicInteger();
        TestEntry entry = new TestEntry(System.currentTimeMillis() - 100_000) {
            @Override
            public long createdAtMs() {
                if (Thread.holdsLock(this)) {
                    lockChecks.incrementAndGet();
                }
                return super.createdAtMs();
            }
        };
        Map<Long, TestEntry> map = new ConcurrentHashMap<>();
        map.put(1L, entry);

        new InflightEvictor<>(map, null).evictExpired(60_000L);

        assertEquals(1, lockChecks.get());
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
    void evictExpiredDoesNotRemoveFreshReplacementForSameKey() {
        long now = System.currentTimeMillis();
        TestEntry expired = new TestEntry(now - 100_000);
        TestEntry fresh = new TestEntry(now);
        ReplaceBeforeRemoveMap map = new ReplaceBeforeRemoveMap(1L, fresh);
        map.put(1L, expired);
        map.armReplacement();
        AtomicInteger callbackCount = new AtomicInteger();
        InflightEvictor<Long, TestEntry> evictor = new InflightEvictor<>(
                map, ignored -> callbackCount.incrementAndGet());

        int evicted = evictor.evictExpired(60_000);

        assertEquals(0, evicted);
        assertEquals(0, callbackCount.get());
        assertEquals(fresh, map.get(1L));
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

    private static final class ReplaceBeforeRemoveMap
            extends ConcurrentHashMap<Long, TestEntry> {
        private final long replacementKey;
        private final TestEntry replacement;
        private boolean armed;

        private ReplaceBeforeRemoveMap(long replacementKey, TestEntry replacement) {
            this.replacementKey = replacementKey;
            this.replacement = replacement;
        }

        private void armReplacement() {
            armed = true;
        }

        private void replaceIfArmed(Object key) {
            if (armed && replacementKey == (Long) key) {
                armed = false;
                super.put(replacementKey, replacement);
            }
        }

        @Override
        public TestEntry remove(Object key) {
            replaceIfArmed(key);
            return super.remove(key);
        }

        @Override
        public boolean remove(Object key, Object value) {
            replaceIfArmed(key);
            return super.remove(key, value);
        }
    }
}
