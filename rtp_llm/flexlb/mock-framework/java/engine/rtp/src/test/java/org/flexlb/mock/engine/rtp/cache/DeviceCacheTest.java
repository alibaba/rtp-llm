package org.flexlb.mock.engine.rtp.cache;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

class DeviceCacheTest {
    @Test void releaseIsIdempotentAndCannotPublishAfterCancellation() {
        try (var cache = new DeviceCache(8, 0, true)) {
            var lease = cache.acquire(2, List.of()).lease();
            assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            lease.close(); lease.close();
            assertFalse(lease.complete(List.of(1L, 2L))); assertFalse(lease.publish(List.of(1L, 2L)));
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void completionKeepsLruAvailableAndSharingChargesEachPhysicalBlockOnce() {
        try (var cache = new DeviceCache(8, 0, true)) {
            var first = cache.acquire(2, List.of()).lease(); first.complete(List.of(1L, 2L));
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
        try (var cache = new DeviceCache(8, 1, true)) {
            var seed = cache.acquire(2, List.of()).lease(); seed.complete(List.of(1L, 2L));
            var owner = cache.acquire(5, List.of()).lease();
            assertNotNull(owner);
            var rejected = cache.acquire(4, List.of(1L, 2L));
            assertNull(rejected.lease()); assertEquals(DeviceCache.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            assertEquals(DeviceCache.Failure.PERMANENT, cache.acquire(Integer.MAX_VALUE, List.of()).failure());
        }
    }

    @Test void growthFailureRetainsPartialLeaseUntilOwnerReleasesIt() {
        try (var cache = new DeviceCache(4, 0, true)) {
            var lease = cache.acquire(1, List.of()).lease();
            assertEquals(DeviceCache.Failure.PERMANENT, lease.growTo(6));
            assertEquals(4, lease.blocks()); assertEquals(0, cache.snapshot(-1, false).availableBlocks());
            lease.close(); assertEquals(4, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void publishedPrefillDataStaysPinnedUntilConnectorOwnerReleasesIt() {
        try (var cache = new DeviceCache(4, 0, true)) {
            var lease = cache.acquire(2, List.of()).lease();
            assertTrue(lease.publish(List.of(1L, 2L))); assertFalse(lease.publish(List.of(1L, 2L)));
            var snapshot = cache.snapshot(-1, true);
            assertEquals(2, snapshot.referencedBlocks()); assertEquals(0, snapshot.heldBlocks());
            lease.close();
            assertEquals(4, cache.snapshot(-1, false).availableBlocks());
            assertEquals(Set.of(1L, 2L), cache.snapshot(-1, true).keys());
        }
    }

    @Test void completionCancellationRaceHasOneOwnershipTransition() throws Exception {
        try (var cache = new DeviceCache(8, 0, true); var callers = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100; i++) {
                var lease = cache.acquire(2, List.of()).lease();
                var ready = new CountDownLatch(1);
                var done = callers.submit(() -> { ready.await(); return lease.complete(List.of(1L, 2L)); });
                var cancelled = callers.submit(() -> { ready.await(); lease.close(); return true; });
                ready.countDown(); done.get(); cancelled.get();
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
                assertEquals(0, cache.snapshot(-1, false).heldBlocks());
            }
        }
    }

    @Test void closeInvalidatesAllLeasesAndPublicationCannotResurrectCache() {
        var cache = new DeviceCache(4, 0, true);
        var lease = cache.acquire(2, List.of()).lease(); cache.close(); cache.close();
        lease.close(); assertFalse(lease.complete(List.of(1L, 2L)));
        assertThrows(IllegalStateException.class, () -> cache.acquire(1, List.of()));
        assertEquals(4, cache.snapshot(-1, false).freeBlocks());
    }
}
