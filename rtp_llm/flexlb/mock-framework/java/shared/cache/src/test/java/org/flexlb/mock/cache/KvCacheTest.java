package org.flexlb.mock.cache;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

class KvCacheTest {
    @Test void releaseIsIdempotentAndCannotPublishAfterCancellation() {
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU)) {
            var lease = cache.acquire(2, List.of()).lease();
            assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            lease.close(); lease.close();
            assertFalse(lease.publish(List.of(1L, 2L)));
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void completionKeepsLruAvailableAndSharingChargesEachPhysicalBlockOnce() {
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU)) {
            var first = cache.acquire(2, List.of()).lease(); first.markValid(2); first.publish(List.of(1L, 2L)); first.close();
            var snapshot = cache.snapshot(-1, true);
            assertEquals(8, snapshot.availableBlocks()); assertEquals(6, snapshot.freeBlocks());
            var a = cache.acquire(2, List.of(1L, 2L)).lease();
            var b = cache.acquire(2, List.of(1L, 2L)).lease();
            assertEquals(2, a.reusedBlocks()); assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            a.close(); assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            b.close(); assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertEquals(snapshot.version(), cache.snapshot(-1, true).version());
        }
    }

    @Test void failedAdmissionRollsBackPinsAndClassifiesHugeDemandWithoutIntegerOverflow() {
        try (var cache = new KvCache(8, 1, KvCache.EvictionPolicy.LRU)) {
            var seed = cache.acquire(2, List.of()).lease(); seed.markValid(2); seed.publish(List.of(1L, 2L)); seed.close();
            var owner = cache.acquire(5, List.of()).lease();
            assertNotNull(owner);
            var rejected = cache.acquire(4, List.of(1L, 2L));
            assertNull(rejected.lease()); assertEquals(KvCache.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            assertEquals(KvCache.Failure.PERMANENT, cache.acquire(Integer.MAX_VALUE, List.of()).failure());
        }
    }

    @Test void growthFailureRollsBackAndRetainsOnlyTheOriginalLease() {
        try (var cache = new KvCache(4, 0, KvCache.EvictionPolicy.LRU)) {
            var lease = cache.acquire(1, List.of()).lease();
            assertEquals(KvCache.Failure.PERMANENT, lease.growTo(6));
            assertEquals(1, lease.blocks()); assertEquals(3, cache.snapshot(-1, false).availableBlocks());
            lease.close(); assertEquals(4, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void completionCancellationRaceHasOneOwnershipTransition() throws Exception {
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var callers = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100; i++) {
                var lease = cache.acquire(2, List.of()).lease();
                lease.markValid(2);
                var ready = new CountDownLatch(1);
                var done = callers.submit(() -> { ready.await(); boolean published = lease.publish(List.of(1L, 2L)); lease.close(); return published; });
                var cancelled = callers.submit(() -> { ready.await(); lease.close(); return true; });
                ready.countDown(); done.get(); cancelled.get();
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
                assertEquals(0, cache.snapshot(-1, false).heldBlocks());
            }
        }
    }

    @Test void closeInvalidatesAllLeasesAndPublicationCannotResurrectCache() {
        var cache = new KvCache(4, 0, KvCache.EvictionPolicy.LRU);
        var lease = cache.acquire(2, List.of()).lease(); cache.close(); cache.close();
        lease.close(); assertFalse(lease.publish(List.of(1L, 2L)));
        assertThrows(IllegalStateException.class, () -> cache.acquire(1, List.of()));
        assertEquals(4, cache.snapshot(-1, false).freeBlocks());
    }
}
