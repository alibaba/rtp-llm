package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RoutingCachePeekTest {
    @Test
    void routingAndObservationDoNotProtectKeysFromLruEviction() {
        var cache = new MockLruBlockCache(4);
        assertTrue(MockEngineTestSupport.seedCache(cache, List.of(1L, 2L, 3L, 4L)));
        assertEquals(2, cache.peekPrefixHitBlocks(List.of(1L, 2L)));
        assertEquals(1, cache.peekPrefixHitBlocks(List.of(1L, 9L, 2L)));
        assertTrue(MockEngineTestSupport.seedCache(cache, List.of(5L)));
        assertFalse(cache.snapshotKeys().contains(1L), "observing the oldest key must not move it to the LRU tail");
        assertTrue(cache.snapshotKeys().containsAll(List.of(2L, 3L, 4L, 5L)));
    }
}
