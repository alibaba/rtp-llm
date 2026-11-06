package org.flexlb.mock.engine.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Behavioral cases mirrored from main FullEvictionTest; no reflection into implementation state. */
class BlockTreeTest {
    private static Set<Long> keySet(KvCacheManager cache) {
        return new HashSet<>(Arrays.stream(cache.snapshot(-1, true).keys()).boxed().toList());
    }
    private static void seed(KvCacheManager cache, Long... keys) {
        var lease = cache.acquire(keys.length, List.of()).lease();
        assertNotNull(lease); assertTrue(lease.publish(List.of(keys), keys.length)); lease.close();
    }
    @Test void reclaimingOneLeafPreservesItsStillMatchableAncestors() {
        try (var cache = new KvCacheManager(1, 3, 0)) {
            seed(cache, 1L, 2L, 3L);
            try (var pressure = cache.acquire(1, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(1L, 2L), keySet(cache));
            }
            try (var reused = cache.acquire(2, List.of(1L, 2L)).lease()) { assertEquals(2, reused.reusedTokens()); }
        }
    }
    @Test void pinnedLeafProtectsItsAncestorsAndRejectionRollsBackPins() {
        try (var cache = new KvCacheManager(1, 3, 0)) {
            seed(cache, 1L, 2L, 3L);
            var pin = cache.acquire(3, List.of(1L, 2L, 3L)).lease();
            assertEquals(KvCacheManager.Failure.RETRYABLE, cache.acquire(1, List.of()).failure());
            pin.close();
            assertEquals(3, cache.snapshot(-1, false).availableBlocks());
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(1L), keySet(cache));
            }
        }
    }
    @Test void matchingRefreshesLruButReleaseDoesNot() {
        try (var cache = new KvCacheManager(1, 3, 0)) {
            seed(cache, 1L); seed(cache, 2L); seed(cache, 3L);
            var first = cache.acquire(1, List.of(1L)).lease();
            var second = cache.acquire(1, List.of(2L)).lease();
            second.close(); first.close();
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(2L), keySet(cache));
            }
        }
    }
    @Test void concurrentOverlappingCompletionsDoNotOverwriteOrRefreshExistingCache() {
        try (var cache = new KvCacheManager(1, 3, 0)) {
            var first = cache.acquire(1, List.of()).lease();
            var overlap = cache.acquire(1, List.of()).lease();
            first.publish(List.of(1L), 1); first.close(); seed(cache, 2L);
            overlap.publish(List.of(1L), 1); overlap.close();
            try (var pressure = cache.acquire(2, List.of()).lease()) {
                assertNotNull(pressure); assertEquals(Set.of(2L), keySet(cache));
            }
        }
    }
    @Test void failedAllocationKeepsMatchRecencyWithoutKeepingThePin() {
        try (var cache = new KvCacheManager(1, 4, 1)) {
            seed(cache, 1L);
            seed(cache, 2L);
            try (var held = cache.acquire(2, List.of()).lease()) {
                assertNotNull(held);
                assertEquals(KvCacheManager.Failure.RETRYABLE, cache.acquire(3, List.of(1L)).failure());
                assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            }
            try (var pressure = cache.acquire(3, List.of()).lease()) {
                assertNotNull(pressure);
                assertEquals(Set.of(1L), keySet(cache));
            }
        }
    }

    @Test void failedReuseBelowReserveLeavesNoNewPin() {
        try (var cache = new KvCacheManager(1, 10, 1)) {
            seed(cache, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
            var held = cache.acquire(1, List.of()).lease();
            assertNotNull(held);
            var rejected = cache.acquire(9, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L));
            assertEquals(KvCacheManager.Failure.RETRYABLE, rejected.failure());
            assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            held.close();
        }
    }
    @Test void batchReclamationPreservesLruAndRestoresEverySkippedPinnedLeaf() {
        var tree = new BlockTree(); var pool = new BlockPool(6);
        var refs = pool.allocate(6, BlockPool.Kind.CACHE);
        var one = tree.insert(1, null, refs.get(0));
        var two = tree.insert(2, one, refs.get(1));
        tree.insert(3, two, refs.get(2));
        var four = tree.insert(4, null, refs.get(3));
        tree.insert(5, four, refs.get(4)); tree.insert(6, null, refs.get(5));
        var pin = pool.retain(refs.get(2), BlockPool.Kind.REQUEST);
        tree.match(List.of(4L, 5L));
        assertEquals(List.of(6L, 5L, 4L), tree.evict(3).stream().map(node -> node.key).toList());
        assertTrue(tree.evict(3).isEmpty()); // All remaining ancestors are protected by the pinned leaf.
        assertEquals(3, tree.size());
        pin.close();
        assertEquals(List.of(3L, 2L, 1L), tree.evict(3).stream().map(node -> node.key).toList());
        assertEquals(0, tree.size()); assertEquals(6, pool.snapshot().free());
    }

}
