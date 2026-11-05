package org.flexlb.mock.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Behavioral cases mirrored from main FullEvictionTest; no reflection into implementation state.
 */
class BlockTreeTest {
    private static void seed(KvCache cache, Long... keys) {
        var lease = cache.acquire(keys.length, List.of()).lease();
        assertNotNull(lease); assertTrue(lease.publish(List.of(keys), keys.length)); lease.close();
    }
    @Test void reclaimingOneLeafPreservesItsStillMatchableAncestors() {
        try (var cache = new KvCache(1, 3, 0)) {
            seed(cache, 1L, 2L, 3L);
            try (var pressure = cache.acquire(1, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(1L, 2L), cache.snapshot(-1, true).keys());
            }
            try (var reused = cache.acquire(2, List.of(1L, 2L)).lease()) { assertEquals(2, reused.reusedTokens()); }
        }
    }
    @Test void pinnedLeafProtectsItsAncestorsAndRejectionRollsBackPins() {
        try (var cache = new KvCache(1, 3, 0)) {
            seed(cache, 1L, 2L, 3L);
            var pin = cache.acquire(3, List.of(1L, 2L, 3L)).lease();
            assertEquals(KvCache.Failure.RETRYABLE, cache.acquire(1, List.of()).failure());
            pin.close();
            assertEquals(3, cache.snapshot(-1, false).availableBlocks());
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(1L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void matchingRefreshesLruButReleaseDoesNot() {
        try (var cache = new KvCache(1, 3, 0)) {
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
        try (var cache = new KvCache(1, 3, 0)) {
            var first = cache.acquire(1, List.of()).lease();
            var overlap = cache.acquire(1, List.of()).lease();
            first.publish(List.of(1L), 1); first.close(); seed(cache, 2L);
            overlap.publish(List.of(1L), 1); overlap.close();
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(2L), cache.snapshot(-1, true).keys());
            }
        }
    }
    @Test void failedReuseBelowReserveLeavesNoNewPin() {
        try (var cache = new KvCache(1, 10, 1)) {
            seed(cache, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
            var held = cache.acquire(1, List.of()).lease();
            assertNotNull(held);
            var rejected = cache.acquire(9, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L));
            assertEquals(KvCache.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            held.close();
        }
    }
    @Test void failedAllocationKeepsMatchRecencyWithoutKeepingThePin() {
        try (var cache = new KvCache(1, 4, 1)) {
            seed(cache, 1L); seed(cache, 2L);
            try (var held = cache.acquire(2, List.of()).lease()) {
                assertNotNull(held);
                assertEquals(KvCache.Failure.RETRYABLE, cache.acquire(3, List.of(1L)).failure());
                assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            }
            try (var pressure = cache.acquire(3, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(1L), cache.snapshot(-1, true).keys());
            }
        }
    }
}
