package org.flexlb.mock.cache;

import org.flexlb.mock.cache.remote.RemoteCache;
import org.junit.jupiter.api.Test;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import static org.junit.jupiter.api.Assertions.*;

class CacheLoadTest {
    private static final List<Long> KEYS = List.of(11L, 22L, 33L);
    private static long deadline() { return System.nanoTime() + TimeUnit.SECONDS.toNanos(5); }
    private static KvCache cache(RemoteCache remote) {
        return new KvCache(2, 8, 0, KvCache.EvictionPolicy.LRU, true, 0, KvCache.EvictionPolicy.LRU, null, null, remote);
    }
    private static void await(BooleanSupplier ready) throws Exception {
        long end = deadline();
        while (!ready.getAsBoolean()) { assertTrue(System.nanoTime() < end, "Timed out"); Thread.sleep(1); }
    }
    static class Backend implements RemoteCache {
        final CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        volatile boolean failRead, failWrite, write, ignoreInterrupt;
        volatile int localBlocks, matches;
        volatile List<Long> transferred;
        public Match match(List<Long> keys, int local, long deadline) {
            matches++; localBlocks = local; return () -> keys.size();
        }
        private void block() throws InterruptedException {
            entered.countDown();
            while (true) {
                try { release.await(); return; }
                catch (InterruptedException error) { if (!ignoreInterrupt) throw error; }
            }
        }
        public void read(Match match, List<Long> keys, long deadline) throws Exception {
            transferred = List.copyOf(keys); block();
            if (failRead) throw new java.io.IOException("Read failed");
        }
        public int write(List<Long> keys, long deadline) throws Exception {
            write = true; transferred = List.copyOf(keys); block();
            if (failWrite) throw new java.io.IOException("Write failed");
            return keys.size();
        }
        public void close() { release.countDown(); }
    }

    @Test void localOnlyPreparationAndCompleteLocalHitsNeedNoRemoteWork() throws Exception {
        var remote = new Backend();
        try (var cache = cache(remote)) {
            try (var seed = cache.acquire(6, List.of()).lease()) { seed.publish(KEYS, 6); }
            try (var hit = cache.prepare(7, KEYS, deadline(), true).load()) {
                assertEquals(6, hit.result().reusedTokens()); assertEquals(0, remote.matches);
            }
            try (var local = cache.prepare(7, List.of(44L), deadline(), false).load()) {
                assertNotNull(local.result()); assertEquals(0, local.result().reusedTokens());
                assertEquals(0, remote.matches);
            }
        }
        try (var cache = cache(null); var load = cache.prepare(7, KEYS, deadline(), true).load()) {
            assertNotNull(load.result()); assertFalse(cache.remoteEnabled());
        }
    }

    @Test void readMustFinishBeforeRemoteKeysBecomeLocalAndReusable() throws Exception {
        var remote = new Backend();
        try (var cache = cache(remote); var load = cache.prepare(7, KEYS, deadline(), true).load()) {
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            assertNull(load.result()); assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            assertEquals(4, cache.snapshot(0, false).availableBlocks());
            remote.release.countDown(); await(() -> load.result() != null);
            assertEquals(6, load.result().reusedTokens()); assertEquals(6, load.result().remoteTokens());
            assertEquals(KEYS, remote.transferred); assertEquals(Set.copyOf(KEYS), cache.snapshot(-1, true).keys());
        }
    }

    @Test void readFailurePreservesOnlyTheValidLocalPrefix() throws Exception {
        var remote = new Backend(); remote.failRead = true; remote.release.countDown();
        try (var cache = cache(remote)) {
            try (var seed = cache.acquire(2, List.of()).lease()) { seed.publish(KEYS.subList(0, 1), 2); }
            try (var load = cache.prepare(7, KEYS, deadline(), true).load()) {
                await(() -> load.result() != null);
                assertEquals(2, load.result().reusedTokens()); assertEquals(0, load.result().remoteTokens());
                assertInstanceOf(java.io.IOException.class, load.result().remoteError());
                assertEquals(1, remote.localBlocks); assertEquals(KEYS.subList(1, 3), remote.transferred);
                assertEquals(Set.of(11L), cache.snapshot(-1, true).keys());
            }
        }
    }

    @Test void cancelledReadKeepsTransferPinsUntilIoStopsAndCannotPublishLateKeys() throws Exception {
        var remote = new Backend(); remote.ignoreInterrupt = true;
        try (var cache = cache(remote)) {
            var load = cache.prepare(7, KEYS, deadline(), true).load();
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            load.close(); load.close();
            assertEquals(5, cache.snapshot(0, false).availableBlocks());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            remote.release.countDown(); await(() -> cache.snapshot(0, false).freeBlocks() == 8);
            assertNull(load.result()); assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void backgroundWriteOutlivesRequestAndReleasesPinsEvenOnFailure() throws Exception {
        for (boolean fail : List.of(false, true)) {
            var remote = new Backend(); remote.failWrite = fail;
            try (var cache = cache(remote)) {
                var load = cache.prepare(7, List.of(), deadline(), false).load();
                var done = load.lease().publishAndStore(KEYS, 6, deadline()).toCompletableFuture();
                assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
                load.close();
                assertTrue(remote.write); assertFalse(done.isDone());
                assertEquals(5, cache.snapshot(0, false).availableBlocks());
                remote.release.countDown();
                if (fail) assertThrows(ExecutionException.class, () -> done.get(2, TimeUnit.SECONDS));
                else done.get(2, TimeUnit.SECONDS);
                assertEquals(8, cache.snapshot(0, false).availableBlocks());
                assertEquals(Set.copyOf(KEYS), cache.snapshot(-1, true).keys());
            }
        }
    }

    @Test void expiredAndMalformedMatchesNeverProduceRemoteHits() throws Exception {
        var remote = new Backend(); remote.release.countDown();
        try (var cache = cache(remote); var load = cache.prepare(7, KEYS, System.nanoTime() - 1, true).load()) {
            await(() -> load.result() != null);
            assertInstanceOf(TimeoutException.class, load.result().remoteError()); assertEquals(0, remote.matches);
        }
        var invalid = new Backend() {
            @Override public Match match(List<Long> keys, int local, long deadline) { return () -> keys.size() + 1; }
        };
        try (var cache = cache(invalid); var load = cache.prepare(7, KEYS, deadline(), true).load()) {
            await(() -> load.result() != null);
            assertEquals(0, load.result().reusedTokens());
            assertInstanceOf(IllegalArgumentException.class, load.result().remoteError());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void shutdownCancelsPendingWorkWithoutPublishingIntoTheClosedCache() throws Exception {
        var remote = new Backend(); remote.ignoreInterrupt = true;
        var cache = cache(remote);
        var load = cache.prepare(7, KEYS, deadline(), true).load();
        assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
        cache.close(); cache.close();
        await(() -> cache.snapshot(0, false).freeBlocks() == 8);
        assertNull(load.result()); assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        assertThrows(IllegalStateException.class, () -> cache.prepare(7, KEYS, deadline(), true));
    }

    @Test void newlyLoadedPrefixesInvalidateCapacityRetryStampBeforeTheOwnerReleasesItsLease() throws Exception {
        var remote = new Backend();
        try (var cache = new KvCache(2, 4, 0, KvCache.EvictionPolicy.LRU, true, 0,
                KvCache.EvictionPolicy.LRU, null, null, remote);
             var owner = cache.prepare(5, KEYS.subList(0, 2), deadline(), true).load()) {
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            var rejected = cache.prepare(5, KEYS.subList(0, 2), deadline(), true);
            assertEquals(KvCache.Failure.RETRYABLE, rejected.failure());
            remote.release.countDown(); await(() -> owner.result() != null);
            assertTrue(cache.allocationGeneration() > rejected.generation());
            try (var shared = cache.prepare(5, KEYS.subList(0, 2), deadline(), true).load()) {
                assertEquals(4, shared.result().reusedTokens());
                assertEquals(0, cache.snapshot(0, false).availableBlocks());
            }
        }
    }

    @Test void capacityRejectionDoesNotStartRemoteWork() {
        var remote = new Backend();
        try (var cache = cache(remote)) {
            assertEquals(KvCache.Failure.PERMANENT, cache.prepare(100, KEYS, deadline(), true).failure());
            try (var held = cache.acquire(16, List.of()).lease()) {
                assertEquals(KvCache.Failure.RETRYABLE, cache.prepare(7, KEYS, deadline(), true).failure());
            }
            assertEquals(0, remote.matches);
        }
    }
}
