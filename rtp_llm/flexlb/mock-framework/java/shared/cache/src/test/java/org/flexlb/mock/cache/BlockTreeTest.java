package org.flexlb.mock.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mock.cache.DeviceCache.EvictionPolicy.*;

/** Behavioral cases mirrored from main FullEvictionTest; no reflection into implementation state. */
class BlockTreeTest {
    private static void seed(DeviceCache cache, Long... keys) {
        var lease = cache.acquire(keys.length, List.of()).lease();
        assertNotNull(lease); assertTrue(lease.complete(List.of(keys)));
    }
    @Test void reclaimingOneLeafPreservesItsStillMatchableAncestors() {
        try (var cache = new DeviceCache(3, 0, LRU)) {
            seed(cache, 1L, 2L, 3L);
            try (var pressure = cache.acquire(1, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(1L, 2L), cache.snapshot(-1, true).keys());
            }
            try (var reused = cache.acquire(2, List.of(1L, 2L)).lease()) { assertEquals(2, reused.reusedBlocks()); }
        }
    }
    @Test void pinnedLeafProtectsItsAncestorsAndRejectionRollsBackPins() {
        try (var cache = new DeviceCache(3, 0, LRU)) {
            seed(cache, 1L, 2L, 3L);
            var pin = cache.acquire(3, List.of(1L, 2L, 3L)).lease();
            assertEquals(DeviceCache.Failure.RETRYABLE, cache.acquire(1, List.of()).failure());
            pin.close();
            assertEquals(3, cache.snapshot(-1, false).availableBlocks());
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(1L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void matchingRefreshesLruButReleaseDoesNot() {
        try (var cache = new DeviceCache(3, 0, LRU)) {
            seed(cache, 1L); seed(cache, 2L); seed(cache, 3L);
            var first = cache.acquire(1, List.of(1L)).lease();
            var second = cache.acquire(1, List.of(2L)).lease();
            second.close(); first.close();
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(2L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void concurrentOverlappingCompletionsDoNotOverwriteOrRefreshExistingCache() {
        try (var cache = new DeviceCache(3, 0, LRU)) {
            var first = cache.acquire(1, List.of()).lease();
            var overlap = cache.acquire(1, List.of()).lease();
            first.complete(List.of(1L)); seed(cache, 2L); overlap.complete(List.of(1L));
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(2L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void lfuAndFifoOnlyChangeOrderingOfEligibleTreeLeaves() {
        for (var policy : List.of(LFU, FIFO)) try (var cache = new DeviceCache(2, 0, policy)) {
            seed(cache, 1L); seed(cache, 2L);
            cache.acquire(1, List.of(1L)).lease().close();
            try (var pressure = cache.acquire(1, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(policy == FIFO ? 2L : 1L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void failedReuseBelowReserveLeavesNoNewPin() {
        try (var cache = new DeviceCache(10, 1, LRU)) {
            seed(cache, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
            var held = cache.acquire(1, List.of()).lease();
            assertNotNull(held);
            var rejected = cache.acquire(9, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L));
            assertEquals(DeviceCache.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            held.close();
        }
    }
}
