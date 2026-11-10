package org.flexlb.mock.engine.cache;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

/**
 * Verifies KV reservations, shared prefixes, and eviction.
 */
class KvCacheManagerTest {
    @Test void keySnapshotsAreDetachedAndEachReaderUsesItsOwnVersion() {
        try (var cache = new KvCacheManager(1, 4, 0)) {
            try (var seed = cache.acquire(2, List.of()).lease()) {
                assertTrue(seed.publish(List.of(Long.MIN_VALUE, Long.MAX_VALUE), 2));
            }
            var first = cache.snapshot(-1, true);
            var expected = new long[]{Long.MIN_VALUE, Long.MAX_VALUE};
            var actual = first.keys().clone(); Arrays.sort(actual);
            assertArrayEquals(expected, actual);
            assertEquals(0, cache.snapshot(first.version(), true).keys().length);
            var secondReader = cache.snapshot(-1, true);
            assertNotSame(first.keys(), secondReader.keys());
            assertEquals(first.version(), secondReader.version());
            Arrays.sort(secondReader.keys());
            assertArrayEquals(expected, secondReader.keys());

            try (var held = cache.acquire(1, List.of()).lease()) {
                var capacityOnly = cache.snapshot(12345, false);
                assertEquals(12345, capacityOnly.version());
                assertEquals(0, capacityOnly.keys().length);
                assertEquals(3, capacityOnly.availableBlocks());
                var unchanged = cache.snapshot(first.version(), true);
                assertEquals(first.version(), unchanged.version());
                assertEquals(0, unchanged.keys().length);
                assertEquals(3, unchanged.availableBlocks());
                assertTrue(held.publish(List.of(0L), 1));
            }
            var updated = cache.snapshot(first.version(), true);
            assertTrue(updated.version() > first.version());
            assertEquals(3, updated.keys().length);
            assertEquals(4, updated.availableBlocks());
            assertArrayEquals(actual, Arrays.stream(first.keys()).sorted().toArray());
            // Mutating a caller-owned snapshot cannot corrupt the live index.
            first.keys()[0] = 17;
            assertArrayEquals(new long[]{Long.MIN_VALUE, 0L, Long.MAX_VALUE},
                Arrays.stream(cache.snapshot(-1, true).keys()).sorted().toArray());
            assertEquals(3, cache.snapshot(updated.version() + 1, true).keys().length);
        }
    }

    @Test void retryablePreparationWaitsForCapacityChange() {
        try (var cache = new KvCacheManager(2, 2, 0)) {
            var held = cache.acquire(4, List.of()).lease();
            var failed = assertInstanceOf(KvCacheManager.Failed.class, cache.prepare(3, List.of()));
            assertEquals(KvCacheManager.Failure.RETRYABLE, failed.failure());
            assertSame(failed, failed.retry());
            held.close();
            var ready = assertInstanceOf(KvCacheManager.Ready.class, failed.retry());
            ready.lease().close();
        }
    }

    @Test void releaseIsIdempotentAndCannotPublishAfterCancellation() {
        try (var cache = new KvCacheManager(1, 8, 0)) {
            var lease = cache.acquire(2, List.of()).lease();
            assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            lease.close(); lease.close();
            assertFalse(lease.publish(List.of(1L, 2L), 2));
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertEquals(0, cache.snapshot(-1, true).keys().length);
        }
    }

    @Test void completionKeepsLruAvailableAndSharingChargesEachPhysicalBlockOnce() {
        try (var cache = new KvCacheManager(1, 8, 0)) {
            var first = cache.acquire(2, List.of()).lease(); first.publish(List.of(1L, 2L), 2); first.close();
            var snapshot = cache.snapshot(-1, true);
            assertEquals(8, snapshot.availableBlocks()); assertEquals(6, snapshot.freeBlocks());
            var a = cache.acquire(2, List.of(1L, 2L)).lease();
            var b = cache.acquire(2, List.of(1L, 2L)).lease();
            assertEquals(2, a.reusedTokens()); assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            a.close(); assertEquals(6, cache.snapshot(-1, false).availableBlocks());
            b.close(); assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertEquals(snapshot.version(), cache.snapshot(-1, true).version());
        }
    }

    @Test void failedAdmissionRollsBackPinsAndClassifiesHugeDemandWithoutIntegerOverflow() {
        try (var cache = new KvCacheManager(1, 8, 1)) {
            var seed = cache.acquire(2, List.of()).lease(); seed.publish(List.of(1L, 2L), 2); seed.close();
            var owner = cache.acquire(5, List.of()).lease();
            assertNotNull(owner);
            var rejected = cache.acquire(4, List.of(1L, 2L));
            assertNull(rejected.lease()); assertEquals(KvCacheManager.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            assertEquals(KvCacheManager.Failure.PERMANENT, cache.acquire(Integer.MAX_VALUE, List.of()).failure());
        }
    }

    @Test void growthFailureRollsBackAndRetainsOnlyTheOriginalLease() {
        try (var cache = new KvCacheManager(1, 4, 0)) {
            var lease = cache.acquire(1, List.of()).lease();
            assertEquals(KvCacheManager.Failure.PERMANENT, lease.ensureCapacity(6));
            assertEquals(3, cache.snapshot(-1, false).availableBlocks());
            lease.close(); assertEquals(4, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void completionCancellationRaceHasOneOwnershipTransition() throws Exception {
        try (var cache = new KvCacheManager(1, 8, 0); var callers = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100; i++) {
                var lease = cache.acquire(2, List.of()).lease();
                var ready = new CountDownLatch(1);
                var done = callers.submit(() -> { ready.await(); boolean published = lease.publish(List.of(1L, 2L), 2); lease.close(); return published; });
                var cancelled = callers.submit(() -> { ready.await(); lease.close(); return true; });
                ready.countDown(); done.get(); cancelled.get();
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
                assertEquals(0, cache.snapshot(-1, false).heldBlocks());
            }
        }
    }

    @Test void closeInvalidatesAllLeasesAndPublicationCannotResurrectCache() {
        var cache = new KvCacheManager(1, 4, 0);
        var lease = cache.acquire(2, List.of()).lease(); cache.close(); cache.close();
        lease.close(); assertFalse(lease.publish(List.of(1L, 2L), 2));
        assertThrows(IllegalStateException.class, () -> cache.acquire(1, List.of()));
        assertEquals(4, cache.snapshot(-1, false).freeBlocks());
    }
    @Test void tokenCapacityRoundsUpWithoutShrinkingAndRejectsOverflowWithoutMutation() {
        try (var cache = new KvCacheManager(128, 4, 0)) {
            try (var lease = cache.acquire(1, List.of()).lease()) {
                assertEquals(3, cache.snapshot(0, false).freeBlocks());
                assertNull(lease.ensureCapacity(128)); assertEquals(3, cache.snapshot(0, false).freeBlocks());
                assertNull(lease.ensureCapacity(129)); assertEquals(2, cache.snapshot(0, false).freeBlocks());
                assertNull(lease.ensureCapacity(1)); assertEquals(2, cache.snapshot(0, false).freeBlocks());
                assertEquals(KvCacheManager.Failure.PERMANENT, lease.ensureCapacity(Long.MAX_VALUE));
                assertThrows(IllegalArgumentException.class, () -> lease.ensureCapacity(-1));
                assertEquals(2, cache.snapshot(0, false).freeBlocks());
                assertThrows(IllegalArgumentException.class, () -> lease.publish(List.of(1L, 2L), 129));
                assertThrows(IllegalArgumentException.class, () -> lease.publish(List.of(1L), 257));
                assertThrows(IllegalArgumentException.class, () -> lease.publish(List.of(1L), -1));
                assertEquals(0, cache.snapshot(-1, true).keys().length);
                lease.publish(List.of(1L), 129);
            }
            try (var reader = cache.acquire(255, List.of(1L)).lease()) {
                assertEquals(128, reader.reusedTokens());
                assertEquals(2, cache.snapshot(0, false).freeBlocks());
            }
            assertEquals(KvCacheManager.Failure.PERMANENT, cache.acquire(Long.MAX_VALUE, List.of()).failure());
            assertThrows(IllegalArgumentException.class, () -> cache.acquire(-1, List.of()));
            assertThrows(IllegalArgumentException.class, () -> cache.acquire(127, List.of(1L)));
        }
    }

    @Test void fullyCachedRequestsStillCompeteForDeviceResidencyAndRetryAfterRelease() {
        try (var cache = new KvCacheManager(1, 3, 8, 0, true)) {
            for (var keys : List.of(List.of(1L, 2L, 3L), List.of(4L, 5L, 6L)))
                try (var seed = cache.acquire(3, List.of()).lease()) { seed.publish(keys, 3); }
            assertEquals(6, cache.itemCount());
            assertEquals(3, cache.snapshot(0, false).totalBlocks());
            var first = assertInstanceOf(KvCacheManager.Ready.class, cache.prepare(3, List.of(1L, 2L, 3L)));
            var failed = assertInstanceOf(KvCacheManager.Failed.class, cache.prepare(3, List.of(4L, 5L, 6L)));
            assertEquals(KvCacheManager.Failure.RETRYABLE, failed.failure()); assertSame(failed, failed.retry());
            assertEquals(0, cache.snapshot(0, false).availableBlocks());
            assertEquals(6, cache.itemCount()); // Device pressure must not evict logical cache entries.
            first.lease().close();
            var second = assertInstanceOf(KvCacheManager.Ready.class, failed.retry());
            assertEquals(3, second.lease().reusedTokens()); second.lease().close();
            assertEquals(3, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void sharedPrefixesUseOneDeviceSlotUntilTheirLastOwnerReleasesIt() {
        try (var cache = new KvCacheManager(1, 3, 8, 0, true)) {
            try (var seed = cache.acquire(2, List.of()).lease()) { seed.publish(List.of(1L, 2L), 2); }
            var a = cache.acquire(2, List.of(1L, 2L)).lease();
            var b = cache.acquire(2, List.of(1L, 2L)).lease();
            assertEquals(1, cache.snapshot(0, false).availableBlocks());
            assertNull(b.ensureCapacity(3));
            assertEquals(KvCacheManager.Failure.RETRYABLE, a.ensureCapacity(3));
            assertEquals(KvCacheManager.Failure.PERMANENT, a.ensureCapacity(4));
            assertEquals(0, cache.snapshot(0, false).availableBlocks());
            b.close(); assertEquals(1, cache.snapshot(0, false).availableBlocks());
            a.close(); assertEquals(3, cache.snapshot(0, false).availableBlocks());
            assertEquals(2, cache.itemCount());
        }
    }

    @Test void singleRequestCannotBorrowLogicalCacheCapacityAsDeviceCapacity() {
        try (var cache = new KvCacheManager(128, 2, 8, 0, true)) {
            assertEquals(KvCacheManager.Failure.PERMANENT, cache.acquire(257, List.of()).failure());
            assertEquals(2, cache.snapshot(0, false).availableBlocks());
            try (var lease = cache.acquire(256, List.of()).lease()) {
                assertEquals(KvCacheManager.Failure.PERMANENT, lease.ensureCapacity(257));
                assertEquals(0, cache.snapshot(0, false).availableBlocks());
            }
        }
    }

    @Test void deviceReserveAppliesToAdmissionButCanBeConsumedByDecodeGrowth() {
        try (var cache = new KvCacheManager(1, 4, 8, 1, true)) {
            assertEquals(KvCacheManager.Failure.PERMANENT, cache.acquire(4, List.of()).failure());
            try (var lease = cache.acquire(3, List.of()).lease()) {
                assertEquals(KvCacheManager.Failure.RETRYABLE, cache.acquire(1, List.of()).failure());
                assertNull(lease.ensureCapacity(4));
                assertEquals(0, cache.snapshot(0, false).availableBlocks());
            }
            assertEquals(4, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void logicalPoolEvictionUsesTotalCapacityRatherThanDeviceCapacity() {
        try (var cache = new KvCacheManager(1, 2, 4, 0, true)) {
            for (long key = 1; key <= 5; key++)
                try (var seed = cache.acquire(1, List.of()).lease()) { seed.publish(List.of(key), 1); }
            assertEquals(4, cache.itemCount());
            assertArrayEquals(new long[]{2, 3, 4, 5}, Arrays.stream(cache.snapshot(-1, true).keys()).sorted().toArray());
            assertEquals(2, cache.snapshot(0, false).availableBlocks());
            assertEquals(0, cache.snapshot(0, false).freeBlocks());
        }
    }

}
