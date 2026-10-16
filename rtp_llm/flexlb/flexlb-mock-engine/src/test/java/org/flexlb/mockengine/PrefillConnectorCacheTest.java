package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class PrefillConnectorCacheTest {
    @Test void oversizedHashMetadataCannotGrowOrCorruptThePhysicalLease() {
        var cache = new MockLruBlockCache(10);
        var keys = List.of(1L, 2L, 3L);
        var lease = cache.acquire(1, keys);
        lease = cache.retainComputed(lease, keys);
        assertEquals(0, cache.heldBlocks());
        assertEquals(1, cache.referencedKeyBlocks());
        assertEquals(9, cache.availableBlocks());
        cache.release(lease);
        assertEquals(10, cache.availableBlocks());
        assertEquals(1, cache.lruKeyBlocks());
        assertTrue(cache.snapshotKeys().contains(1L));
        assertFalse(cache.snapshotKeys().contains(2L));
    }

    @Test void computedKeysStayPinnedUntilConnectorReleases() {
        var cache = new MockLruBlockCache(10);
        var lease = cache.acquire(3, List.of(1L, 2L, 3L));
        lease = cache.retainComputed(lease, List.of(1L, 2L, 3L));
        assertEquals(0, cache.heldBlocks());
        assertEquals(3, cache.referencedKeyBlocks());
        assertEquals(7, cache.availableBlocks());
        assertNull(cache.acquire(8, List.of()));
        cache.release(lease);
        assertEquals(0, cache.referencedKeyBlocks());
        assertEquals(10, cache.availableBlocks());
        assertTrue(cache.snapshotKeys().containsAll(List.of(1L, 2L, 3L)));
    }

    @Test void twoComputedCopiesSharePhysicalKeysWithoutLosingEitherReference() {
        var cache = new MockLruBlockCache(10);
        var first = cache.acquire(2, List.of(1L, 2L));
        var second = cache.acquire(2, List.of(1L, 2L));
        first = cache.retainComputed(first, List.of(1L, 2L));
        second = cache.retainComputed(second, List.of(1L, 2L));
        assertEquals(0, cache.heldBlocks());
        assertEquals(8, cache.availableBlocks());
        cache.release(first);
        assertEquals(2, cache.referencedKeyBlocks());
        cache.release(second);
        assertEquals(10, cache.availableBlocks());
    }
}
