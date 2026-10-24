package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.util.HashSet;
import static org.junit.jupiter.api.Assertions.*;

class CacheSyncFixtureTest {
    @Test
    void globalSharedSetHasExactOwnershipAndSignedWireKeys() throws Exception {
        var union = new HashSet<Long>();
        var sets = new java.util.ArrayList<java.util.Set<Long>>();
        for (int worker = 0; worker < 6; worker++) {
            var fixture = new CacheSyncFixture(worker, 6, 176, 65, 2);
            var response = fixture.response();
            assertEquals(241, response.getCacheKeysCount());
            assertEquals(response, EngineRpcService.CacheStatusPB.parseFrom(response.toByteArray()));
            assertEquals(2, response.getVersion());
            sets.add(response.getCacheKeysMap().keySet());
            union.addAll(response.getCacheKeysMap().keySet());
        }
        assertEquals(6 * 176 + 65, union.size());
        var common = new HashSet<>(sets.get(0));
        sets.forEach(common::retainAll);
        assertEquals(65, common.size());
        for (int a = 0; a < sets.size(); a++) for (int c = a + 1; c < sets.size(); c++) {
            var overlap = new HashSet<>(sets.get(a)); overlap.retainAll(sets.get(c));
            assertEquals(common, overlap, "every pair shares exactly the global set");
        }
        assertTrue(union.stream().anyMatch(key -> key < 0));
    }

    @Test void elapsedTimeNotPollsControlsPrivateReplacementAndPauseCatchesUp() {
        var now = new java.util.concurrent.atomic.AtomicLong();
        var slow = new CacheSyncFixture(0, 6, 176, 65, 2, now::get);
        var fast = new CacheSyncFixture(0, 6, 176, 65, 2, now::get);
        slow.churn(17, 1); fast.churn(17, 1);
        var initial = slow.response();
        for (int i = 0; i < 80; i++) fast.response();
        assertEquals(0L, fast.snapshot().get("added_keys"));
        now.set(5_000_001L);
        var a = slow.response(); var b = fast.response();
        assertEquals(a, b);
        assertEquals(85L, slow.snapshot().get("added_keys"));
        assertEquals(5L, slow.snapshot().get("changed_versions"));
        var added = new HashSet<>(a.getCacheKeysMap().keySet()); added.removeAll(initial.getCacheKeysMap().keySet());
        assertEquals(85, added.size());
        now.set(6_000_001L);
        fast.churn(0, 1); // The unpolled sixth interval is accounted before pausing.
        assertEquals(102L, fast.snapshot().get("added_keys"));
        var frozen = fast.response(); now.set(60_000_000L);
        assertEquals(frozen, fast.response());
        assertEquals(102L, fast.snapshot().get("removed_keys"));
    }

    @Test void sharedKeysRemainCommonDuringPrivateChurn() {
        var now = new java.util.concurrent.atomic.AtomicLong();
        var a = new CacheSyncFixture(0, 2, 176, 65, 2, now::get);
        var b = new CacheSyncFixture(1, 2, 176, 65, 2, now::get);
        var before = new HashSet<>(a.response().getCacheKeysMap().keySet());
        before.retainAll(b.response().getCacheKeysMap().keySet());
        a.churn(17, 1); now.set(10_000_001L);
        var after = new HashSet<>(a.response().getCacheKeysMap().keySet());
        after.retainAll(b.response().getCacheKeysMap().keySet());
        assertEquals(65, before.size()); assertEquals(before, after);
    }

    @Test void invalidGeometryAndRatesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new CacheSyncFixture(0, 0, 1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new CacheSyncFixture(0, 1, 1_000_000, 1, 2));
        var fixture = new CacheSyncFixture(0, 1, 100, 10, 2);
        assertThrows(IllegalArgumentException.class, () -> fixture.churn(101, 1));
        assertThrows(IllegalArgumentException.class, () -> fixture.churn(10, 0));
        assertThrows(IllegalArgumentException.class, () -> fixture.churn(-1, 1));
        fixture.churn(0, 6000);
        assertEquals(110, fixture.response().getCacheKeysCount());
    }

    @Test void unchangedVersionOmitsKeysAndCountsOnlyFullPayloads() {
        var fixture = new CacheSyncFixture(0, 6, 176, 65, 2, () -> 0L);
        for (long requested : new long[]{-1, 2, 2, 3})
            assertEquals(requested == 2 ? 0 : 241, fixture.response(requested).getCacheKeysCount());
        assertEquals(2L, fixture.snapshot().get("unchanged_responses"));
        assertEquals(2L, fixture.snapshot().get("full_responses"));
        assertEquals(482L, fixture.snapshot().get("entries"));
        assertEquals(0L, fixture.snapshot().get("added_keys"));
    }
}
