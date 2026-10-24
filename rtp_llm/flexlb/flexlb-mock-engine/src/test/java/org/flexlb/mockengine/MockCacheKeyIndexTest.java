package org.flexlb.mockengine;

import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockCacheKeyIndexTest {
    @Test void versionsFollowLogicalNodesAcrossCopiesAndCapacityChanges() {
        var index = new MockCacheKeyIndex();
        var device = new MockLruBlockCache(8);
        var host = new MockMemoryBlockCache(8, ignored -> {});
        device.setKeyListener(index::update); host.setKeyListener(index::update);
        assertEquals(-1, index.version(null));
        device.admit(List.of(1L, 2L));
        var original = index.snapshot(null, -1, true);
        assertEquals(1, original.version());
        assertEquals(2, original.natural().length);
        host.write(List.of(1L, 2L));
        device.admit(List.of(1L, 2L));
        assertEquals(original.version(), index.version(null), "copies and duplicate inserts retain nodes");
        var lease = device.acquire(2, List.of(1L, 2L));
        assertNotNull(lease);
        assertEquals(0, index.snapshot(null, original.version(), true).natural().length);
        device.release(lease);
        assertEquals(original.version(), index.version(null), "references are capacity changes only");
        device.evict(List.of(2L, 1L));
        assertEquals(original.version(), index.version(null), "host backing keeps the logical nodes");
        host.evictTo(0);
        var empty = index.snapshot(null, original.version(), true);
        assertEquals(original.version() + 2, empty.version());
        assertEquals(0, empty.natural().length);
        assertEquals(2, original.natural().length, "in-flight snapshot is immutable");
        assertEquals(700, index.snapshot(null, 700, false).version());
    }

    @Test void unionVersionControlsFixturePayloadAndSnapshotSurvivesChurn() {
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var fixture = new CacheSyncFixture(0, 1, 10, 2, 2, clock::get);
        var index = new MockCacheKeyIndex();
        index.update(7L, true);
        var initial = index.snapshot(fixture, -1, true);
        assertEquals(12, initial.external().count());
        var same = index.snapshot(fixture, initial.version(), true);
        assertNull(same.external()); assertEquals(0, same.natural().length);
        index.update(8L, true);
        var naturalChange = index.snapshot(fixture, initial.version(), true);
        assertNotNull(naturalChange.external(), "natural change requires the whole union, including unchanged fixture");
        assertEquals(2, naturalChange.natural().length);
        fixture.churn(1, 1); clock.set(2_000_000);
        var churned = index.snapshot(fixture, naturalChange.version(), true);
        assertEquals(naturalChange.version() + 2, churned.version());
        assertEquals(0, initial.external().offset());
        assertEquals(2, churned.external().offset());
        assertEquals(3L, fixture.snapshot().get("full_responses"));
        assertEquals(1L, fixture.snapshot().get("unchanged_responses"));
    }

    @Test void concurrentSnapshotsNeverPairAKeySetWithAnotherVersion() throws Exception {
        var index = new MockCacheKeyIndex();
        var device = new MockLruBlockCache(2);
        device.setKeyListener(index::update);
        var writer = java.util.concurrent.CompletableFuture.runAsync(() -> {
            for (int i = 0; i < 10000; i++) { device.admit(List.of(1L)); device.evict(List.of(1L)); }
        });
        do {
            var snapshot = index.snapshot(null, Long.MIN_VALUE, true);
            assertEquals((snapshot.version() + 1) % 2, snapshot.natural().length,
                    "each odd mutation publishes the key, each even mutation removes it");
        } while (!writer.isDone());
        writer.get(5, java.util.concurrent.TimeUnit.SECONDS);
        assertEquals(19999, index.version(null));
    }
}
