package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class CacheRetentionQuotaTest {
    private final List<Long> keys = List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L);

    @Test
    void defaultsPreserveFullCache() {
        MockLruBlockCache cache = new MockLruBlockCache(12);
        cache.admit(keys);
        assertEquals(8, cache.prefixHitBlocks(keys));
        assertEquals(0, cache.retentionEvictions());
    }

    @Test
    void smallRetentionDoesNotRejectLargeSuccessfulRequest() {
        MockLruBlockCache cache = new MockLruBlockCache(12);
        cache.setRetentionBlocks(4);
        MockLruBlockCache.BlockLease lease = cache.acquire(8, keys);
        assertNotNull(lease);
        assertTrue(cache.admit(lease, keys));
        // Retention is an upper bound: one eight-block chain is indivisible.
        assertEquals(0, cache.snapshotKeys().size());
        assertEquals(8, cache.retentionEvictions());
        assertEquals(8, cache.evictions());
        assertEquals(12, cache.availableBlocks());
        assertEquals(0, cache.prefixHitBlocks(keys));
    }

    @Test
    void quotaNeverEvictsReferencedKeysAndAppliesAfterRelease() {
        MockLruBlockCache cache = new MockLruBlockCache(12);
        cache.admit(keys);
        MockLruBlockCache.BlockLease lease = cache.acquireWithReuse(8, keys);
        assertNotNull(lease);
        cache.setRetentionBlocks(0);
        assertEquals(8, cache.prefixHitBlocks(keys));
        cache.release(lease);
        assertTrue(cache.snapshotKeys().isEmpty());
        assertEquals(12, cache.freeBlocks());
        assertEquals(8, cache.retentionEvictions());
    }

    @Test
    void invalidQuotaDoesNotChangePool() {
        MockLruBlockCache cache = new MockLruBlockCache(12);
        assertThrows(IllegalArgumentException.class, () -> cache.setRetentionBlocks(-1));
        assertThrows(IllegalArgumentException.class, () -> cache.setRetentionBlocks(13));
        assertEquals(12, cache.retentionBlocks());
        assertEquals(12, cache.totalBlocks());
    }
}
