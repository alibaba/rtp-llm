package org.flexlb.mock.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mock.cache.KvCache.EvictionPolicy.*;

class HostCacheTest {
    private KvCache cache(boolean device, int blocks, int host) {
        return new KvCache(1, blocks, 0, LRU, device, host, FIFO, new KvCache.Watermark(.5, 1), null);
    }
    private void seed(KvCache cache, Long... keys) {
        try (var lease = cache.acquire(keys.length, List.of()).lease()) { lease.publish(List.of(keys), keys.length); }
    }

    @Test void watermarkDemotesAnEntirePrefixWithoutLosingItsLogicalIndex() {
        try (var cache = cache(true, 4, 8)) {
            seed(cache, 1L, 2L, 3L, 4L);
            assertEquals(Set.of(1L, 2L, 3L, 4L), cache.snapshot(-1, true).keys());
            assertEquals(4, cache.hostSnapshot().total() - cache.hostSnapshot().free());
            long version = cache.snapshot(-1, true).version();
            try (var lease = cache.acquire(4, List.of(1L, 2L, 3L, 4L)).lease()) {
                assertNotNull(lease); assertEquals(4, lease.reusedTokens());
                assertEquals(4, cache.hostSnapshot().free());
                assertEquals(0, cache.snapshot(-1, false).availableBlocks());
                assertEquals(version, cache.snapshot(-1, true).version());
                assertEquals(KvCache.Failure.RETRYABLE, cache.acquire(1, List.of()).failure());
            }
            assertEquals(4, cache.snapshot(-1, false).availableBlocks());
        }
    }
    @Test void hostOnlyCacheRetainsItsSourceAndNeverPublishesPrivateWorkingCopies() {
        try (var cache = cache(false, 4, 8)) {
            seed(cache, 1L, 2L);
            assertEquals(4, cache.snapshot(-1, false).freeBlocks());
            assertEquals(6, cache.hostSnapshot().free());
            try (var a = cache.acquire(2, List.of(1L, 2L)).lease();
                 var b = cache.acquire(2, List.of(1L, 2L)).lease()) {
                assertEquals(2, a.reusedTokens()); assertEquals(2, b.reusedTokens());
                assertEquals(6, cache.hostSnapshot().free());
                assertEquals(0, cache.snapshot(-1, false).referencedBlocks());
            }
            assertEquals(4, cache.snapshot(-1, false).freeBlocks());
        }
    }
    @Test void failedLoadAdmissionDoesNotConsumeHostOrLeakPins() {
        try (var cache = cache(false, 4, 4)) {
            seed(cache, 1L, 2L);
            try (var owner = cache.acquire(3, List.of()).lease()) {
                assertEquals(KvCache.Failure.RETRYABLE, cache.acquire(2, List.of(1L, 2L)).failure());
                assertEquals(Set.of(1L, 2L), cache.snapshot(-1, true).keys());
                assertEquals(2, cache.hostSnapshot().free());
            }
            assertEquals(2, cache.acquire(2, List.of(1L, 2L)).lease().reusedTokens());
        }
    }
    @Test void publicationRequiresValidityAndDoesNotReleaseTheOwner() {
        try (var cache = cache(true, 8, 4)) {
            var lease = cache.acquire(3, List.of()).lease();
            assertThrows(IllegalArgumentException.class, () -> lease.publish(List.of(1L), 0));
            lease.publish(List.of(1L), 1);
            assertThrows(IllegalArgumentException.class, () -> lease.publish(List.of(1L, 2L), 1));
            assertEquals(5, cache.snapshot(-1, false).availableBlocks());
            try (var reader = cache.acquire(1, List.of(1L)).lease()) {
                assertEquals(1, reader.reusedTokens());
                lease.close(); assertEquals(7, cache.snapshot(-1, false).availableBlocks());
            }
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }
    @Test void hostPressureKeepsAValidPrefixAndOptionalStoresNeverFailTheRequest() {
        try (var cache = cache(false, 8, 2)) {
            var lease = cache.acquire(4, List.of()).lease(); lease.publish(List.of(1L, 2L, 3L, 4L), 4);
            lease.close();
            assertEquals(Set.of(1L, 2L), cache.snapshot(-1, true).keys());
            assertEquals(8, cache.snapshot(-1, false).freeBlocks());
        }
    }
    @Test void hostWatermarksAndPolicyEvictOnlyEligibleLeaves() {
        try (var cache = new KvCache(1, 8, 0, LRU, false, 4, FIFO, null, new KvCache.Watermark(.5, 1))) {
            seed(cache, 1L); seed(cache, 2L); seed(cache, 3L); seed(cache, 4L);
            assertEquals(Set.of(3L, 4L), cache.snapshot(-1, true).keys());
            assertEquals(2, cache.hostSnapshot().free());
        }
    }
    @Test void hugeMatchedDemandIsPermanentAndFailedGrowthIsAtomic() {
        try (var cache = new KvCache(1, 8, 0, LRU)) {
            seed(cache, 1L, 2L, 3L, 4L);
            assertEquals(KvCache.Failure.PERMANENT, cache.acquire(10, List.of(1L, 2L, 3L, 4L)).failure());
            try (var lease = cache.acquire(2, List.of()).lease(); var other = cache.acquire(5, List.of()).lease()) {
                assertEquals(KvCache.Failure.RETRYABLE, lease.ensureCapacity(4));
                assertEquals(1, cache.snapshot(-1, false).availableBlocks());
            }
        }
    }
    @Test void closeReleasesBothTiersAndInvalidatesAllOwners() {
        var cache = cache(true, 4, 4); seed(cache, 1L, 2L, 3L, 4L);
        var lease = cache.acquire(1, List.of()).lease(); cache.close(); lease.close(); cache.close();
        assertEquals(4, cache.hostSnapshot().free()); assertEquals(4, cache.snapshot(-1, false).freeBlocks());
        assertFalse(lease.publish(List.of(1L), 1));
    }
    @Test void watermarkCanUnregisterActiveCacheWithoutReleasingRequestBlocks() {
        try (var cache = new KvCache(1, 10, 0, LRU, true, 20, LRU, new KvCache.Watermark(.5, .9), null)) {
            var held = cache.acquire(8, List.of()).lease();
            seed(cache, 1L); // High watermark reached; only this block is registered.
            assertEquals(8, cache.snapshot(-1, false).heldBlocks());
            held.publish(List.of(2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L), 8); held.close();
            assertEquals(10, cache.snapshot(-1, false).freeBlocks());
            assertEquals(11, cache.hostSnapshot().free());
        }
    }

    @Test void fullHostKeepsBothItsKeysAndDeviceTransferSource() {
        try (var cache = cache(true, 4, 1)) {
            var lease = cache.acquire(4, List.of()).lease();
            lease.publish(List.of(1L, 2L, 3L, 4L), 4);
            assertEquals(0, cache.hostSnapshot().free());
            assertEquals(Set.of(1L, 2L, 3L, 4L), cache.snapshot(-1, true).keys());
            assertEquals(0, cache.snapshot(-1, false).freeBlocks(), "Active request keeps the demoted physical source");
            assertEquals(1, cache.snapshot(-1, false).heldBlocks());
            lease.close();
            assertEquals(1, cache.snapshot(-1, false).freeBlocks());
            assertEquals(Set.of(1L, 2L, 3L, 4L), cache.snapshot(-1, true).keys());
        }
    }

}
