package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

class TreeEvictionTest {
    @org.junit.jupiter.api.io.TempDir
    java.nio.file.Path tempDir;

    @Test
    void engineEventChannelRecordsWholeChainAndActualFreedCount() throws Exception {
        var model = MockEngineTestSupport.performanceModel(tempDir, "20", 0.1, 1000.0);
        try (var cluster = MockEngineTestCluster.create(model, 61500, 1, 0);
             var log = JavaMockEngineCluster.EngineEventLog.open(
                     tempDir.resolve("engine_events.jsonl").toString())) {
            var service = cluster.prefill(0);
            service.setEngineEventLog(log);
            var field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
            field.setAccessible(true);
            var cache = (MockLruBlockCache) field.get(service);
            cache.admit(List.of(1L, 2L, 3L));
            cache.setRetentionBlocks(2);
            var row = new com.fasterxml.jackson.databind.ObjectMapper().readTree(
                    java.nio.file.Files.readString(tempDir.resolve("engine_events.jsonl")));
            assertEquals("evict_chain", row.path("event").asText());
            assertEquals(3, row.path("leaf_key").asLong());
            assertEquals("[3,2,1]", row.path("chain_keys").toString());
            assertEquals(3, row.path("blocks_freed").asInt());
            assertEquals("retention", row.path("reason").asText());
            assertFalse(row.path("engine_name").asText().isEmpty());
        }
    }

    private MockLruBlockCache branched(List<MockLruBlockCache.EvictionEvent> events) {
        var cache = new MockLruBlockCache(5, 0);
        cache.setEvictionListener(events::add);
        cache.admit(List.of(1L, 2L, 3L, 4L));
        cache.admit(List.of(2L, 5L));
        return cache;
    }

    @Test
    void oldestTailRoundsUpAndBranchProtectsSharedTrunk() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = branched(events);
        var request = cache.acquire(1, List.of());
        assertNotNull(request);
        assertEquals(Set.of(1L, 2L, 5L), cache.snapshotKeys());
        assertEquals(List.of(4L, 3L), events.get(0).chainKeys());
        assertEquals(2, events.get(0).blocksFreed());
        assertEquals(2, cache.evictions());
        cache.release(request);
        // Once the fork disappears the remaining branch is a single chain.
        var next = cache.acquire(3, List.of());
        assertNotNull(next);
        assertEquals(List.of(5L, 2L, 1L), events.get(1).chainKeys());
        assertTrue(cache.snapshotKeys().isEmpty());
        cache.release(next);
        assertEquals(5, cache.freeBlocks());
    }

    @Test
    void touchMovesLeafButDoesNotReparentSuffix() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = branched(events);
        assertEquals(2, cache.prefixHitBlocks(List.of(3L, 4L)));
        cache.acquire(1, List.of());
        assertEquals(List.of(5L), events.get(0).chainKeys());
        assertEquals(Set.of(1L, 2L, 3L, 4L), cache.snapshotKeys());
    }

    @Test
    void referencedAncestorSurvivesWhileItsUnreferencedTailIsEvicted() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = branched(events);
        var held = cache.acquireWithReuse(1, List.of(3L));
        assertNotNull(held);
        var extra = cache.acquire(1, List.of());
        assertNotNull(extra);
        assertEquals(List.of(4L, 3L), events.get(0).chainKeys());
        assertEquals(1, events.get(0).blocksFreed());
        assertEquals(1, cache.referencedKeyBlocks());
        assertTrue(cache.snapshotKeys().contains(3L));
        cache.release(held);
        cache.release(extra);
        assertEquals(5, cache.availableBlocks());
    }

    @Test
    void pinnedLeafLeavesLruAndReturnsAfterRelease() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = branched(events);
        var held = cache.acquireWithReuse(1, List.of(4L));
        cache.acquire(1, List.of());
        assertEquals(List.of(5L), events.get(0).chainKeys());
        cache.release(held);
        cache.setRetentionBlocks(0);
        assertTrue(cache.snapshotKeys().isEmpty());
        assertEquals(4, cache.retentionEvictions());
    }

    @Test
    void explicitEvictionIsKeyLocalAndClearDiscardsOldTopology() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = branched(events);
        assertTrue(cache.evict(List.of(2L)));
        assertEquals(Set.of(1L, 3L, 4L, 5L), cache.snapshotKeys());
        assertTrue(events.isEmpty());
        cache.clear();
        cache.admit(List.of(4L, 2L));
        cache.setRetentionBlocks(1);
        assertEquals(List.of(2L, 4L), events.get(0).chainKeys());
        assertEquals(2, cache.evictions());
        assertEquals(2, cache.retentionEvictions());
        assertEquals(5, cache.freeBlocks());
    }

    @Test
    void computedPublicationBuildsTreeWithoutUnpinningConnector() {
        var events = new ArrayList<MockLruBlockCache.EvictionEvent>();
        var cache = new MockLruBlockCache(6, 0);
        cache.setEvictionListener(events::add);
        var lease = cache.acquire(3, List.of(1L, 2L, 3L));
        lease = cache.retainComputed(lease, List.of(1L, 2L, 3L));
        cache.setRetentionBlocks(0);
        assertTrue(events.isEmpty());
        assertEquals(3, cache.referencedKeyBlocks());
        cache.release(lease);
        assertEquals(List.of(3L, 2L, 1L), events.get(0).chainKeys());
        assertEquals(6, cache.freeBlocks());
    }

    @Test
    void structurallyProtectedCapacityCannotBeOverAllocated() {
        var cache = new MockLruBlockCache(3, 0);
        cache.admit(List.of(1L, 2L, 3L));
        var held = cache.acquireWithReuse(1, List.of(3L));
        assertNotNull(held);
        // A suffix-only request can pin the only leaf without pinning ancestors.
        assertEquals(2, cache.availableBlocks());
        assertEquals(MockLruBlockCache.AllocationFailure.RETRYABLE,
                cache.acquireDetailed(1, List.of()).failure());
        assertEquals(MockLruBlockCache.AllocationFailure.RETRYABLE,
                cache.acquireWithReuseDetailed(1, List.of()).failure());
        assertEquals(0, cache.freeBlocks());
        assertEquals(0, cache.heldBlocks());
        cache.release(held);
        assertNotNull(cache.acquire(1, List.of()));
        assertEquals(2, cache.freeBlocks());
    }

    @Test
    void prefillPinsHitBeforeEvictingTailForUncachedSuffix() {
        var cache = new MockLruBlockCache(5, 0);
        cache.admit(List.of(1L, 2L, 3L));
        cache.admit(List.of(8L, 9L));
        var lease = cache.acquire(3, List.of(1L, 2L, 4L));
        assertNotNull(lease);
        assertTrue(cache.snapshotKeys().containsAll(List.of(1L, 2L)));
        assertEquals(2, cache.referencedKeyBlocks());
        cache.admit(lease, List.of(1L, 2L, 4L));
        assertTrue(cache.freeBlocks() >= 0);
        assertEquals(5, cache.availableBlocks());
    }
}
