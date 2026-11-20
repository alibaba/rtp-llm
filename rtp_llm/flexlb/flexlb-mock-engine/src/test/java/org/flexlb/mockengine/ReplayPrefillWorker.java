package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Read real engine counters and publish cache membership deltas without taking a full snapshot per arrival tick. */
final class ReplayPrefillWorker {
    final MockLruBlockCache cache;
    final AtomicLong fresh;
    final AtomicLong input;
    final AtomicLong hits;
    final AtomicInteger waiting;
    private final Membership membership = new Membership();

    // The mock only changes key membership through put/remove/clear under the cache monitor.
    // Preserve access ordering and refcounts; containsKey does not refresh access order.
    private static final class Membership extends LinkedHashMap<Long, Integer> {
        final Set<Long> changed = new HashSet<>();
        Membership() { super(16, 0.75f, true); }
        @Override public Integer put(Long key, Integer value) {
            Integer old = super.put(key, value);
            if (old == null) { changed.add(key); }
            return old;
        }
        @Override public Integer remove(Object key) {
            Integer old = super.remove(key);
            if (old != null) { changed.add((Long) key); }
            return old;
        }
        @Override public void clear() {
            changed.addAll(keySet());
            super.clear();
        }
    }

    ReplayPrefillWorker(JavaMockEngineCluster.FastRpcService engine) {
        cache = (MockLruBlockCache) ReflectionTestUtils.getField(engine, "cache");
        fresh = (AtomicLong) ReflectionTestUtils.getField(engine, "contextComputeTokens");
        input = (AtomicLong) ReflectionTestUtils.getField(engine, "contextWithCacheTokens");
        hits = (AtomicLong) ReflectionTestUtils.getField(engine, "hitTokensTotal");
        waiting = (AtomicInteger) ReflectionTestUtils.getField(engine, "waitingPrefillRequests");
        synchronized (cache) {
            if (!cache.snapshotKeys().isEmpty()) { throw new IllegalStateException("install before seeding"); }
            ReflectionTestUtils.setField(cache, "blocks", membership);
        }
    }

    boolean publish(GlobalCacheIndex index, PrefillEndpoint worker) {
        synchronized (cache) {
            if (membership.changed.isEmpty()) { return false; }
            String address = worker.ipPort();
            for (long key : membership.changed) {
                if (membership.containsKey(key)) { index.addCacheBlock(key, address); }
                else { index.removeCacheBlock(address, key); }
            }
            membership.changed.clear();
            return true;
        }
    }

    int physicalKeys() {
        synchronized (cache) { return membership.size(); }
    }
}
