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
        boolean touched;
        private void recordTransition(Long key) {
            touched = true;
            // Entering and leaving between two replies cancels the membership change.
            if (!changed.add(key)) { changed.remove(key); }
        }
        Membership() { super(16, 0.75f, true); }
        @Override public Integer put(Long key, Integer value) {
            Integer old = super.put(key, value);
            if (old == null) { recordTransition(key); }
            return old;
        }
        @Override public Integer remove(Object key) {
            Integer old = super.remove(key);
            if (old != null) { recordTransition((Long) key); }
            return old;
        }
        @Override public void clear() {
            keySet().forEach(this::recordTransition);
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

    record Delta(Long[] added, Long[] removed, boolean touched) {
        boolean changed() { return added.length != 0 || removed.length != 0; }

        void apply(GlobalCacheIndex index, String address) {
            for (Long key : added) { index.addCacheBlock(key, address); }
            for (Long key : removed) { index.removeCacheBlock(address, key); }
        }
    }

    private static final Delta EMPTY = new Delta(new Long[0], new Long[0], false);

    Delta captureDelta() {
        synchronized (cache) {
            if (!membership.touched) { return EMPTY; }
            membership.touched = false;
            var added = new java.util.ArrayList<Long>();
            var removed = new java.util.ArrayList<Long>();
            for (Long key : membership.changed) {
                (membership.containsKey(key) ? added : removed).add(key);
            }
            membership.changed.clear();
            return new Delta(added.toArray(Long[]::new), removed.toArray(Long[]::new), true);
        }
    }

    boolean publish(GlobalCacheIndex index, PrefillEndpoint worker) {
        var delta = captureDelta();
        delta.apply(index, worker.ipPort());
        return delta.touched();
    }

    int physicalKeys() {
        synchronized (cache) { return membership.size(); }
    }
}
