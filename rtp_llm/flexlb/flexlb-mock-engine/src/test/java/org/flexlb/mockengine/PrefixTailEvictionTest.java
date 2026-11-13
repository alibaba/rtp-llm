package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefixTailEvictionTest {
    private static boolean seed(MockLruBlockCache cache, List<Long> keys) {
        var lease = cache.acquire(keys.size(), keys);
        assertNotNull(lease);
        return cache.admit(lease, keys);
    }

    @Test
    void capacityPressureTruncatesTheTailInsteadOfInvalidatingTheWholePrefix() {
        var cache = new MockLruBlockCache(6, 0);
        assertTrue(seed(cache, List.of(1L, 2L, 3L, 4L, 5L, 6L)));
        var lease = cache.acquire(1, List.of(7L));
        assertNotNull(lease);
        assertEquals(5, cache.peekPrefixHitBlocks(List.of(1L, 2L, 3L, 4L, 5L, 6L)));
        cache.release(lease);
        assertEquals(0, cache.heldBlocks());
        assertEquals(0, cache.referencedKeyBlocks());
    }

    @Test
    void siblingBranchesKeepTheirSharedPrefixUntilBothTailsAreGone() {
        var cache = new MockLruBlockCache(4, 0);
        assertTrue(seed(cache, List.of(1L, 2L, 3L)));
        assertTrue(seed(cache, List.of(1L, 2L, 4L)));
        var lease = cache.acquire(2, List.of(7L, 8L));
        assertNotNull(lease);
        assertEquals(2, cache.peekPrefixHitBlocks(List.of(1L, 2L, 4L)));
        cache.release(lease);
        cache.clear();
        assertTrue(seed(cache, List.of(1L, 2L, 3L, 4L)));
        var afterReset = cache.acquire(1, List.of(9L));
        assertNotNull(afterReset);
        assertEquals(3, cache.peekPrefixHitBlocks(List.of(1L, 2L, 3L, 4L)));
        cache.release(afterReset);
    }

    @Test
    void forcedEvictionAndReinsertionPreserveTheSurvivingPrefix() {
        var cache = new MockLruBlockCache(4, 0);
        assertTrue(seed(cache, List.of(1L, 2L, 3L)));
        assertTrue(cache.evict(List.of(2L)));
        assertTrue(seed(cache, List.of(1L, 2L, 3L)));
        var lease = cache.acquire(2, List.of(8L, 9L));
        assertNotNull(lease);
        assertEquals(2, cache.peekPrefixHitBlocks(List.of(1L, 2L, 3L)));
        cache.release(lease);
        assertEquals(0, cache.heldBlocks());
        assertEquals(0, cache.referencedKeyBlocks());
    }

}
