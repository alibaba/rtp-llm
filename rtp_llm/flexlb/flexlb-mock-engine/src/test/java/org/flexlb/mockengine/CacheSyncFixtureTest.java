package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.util.HashSet;
import static org.junit.jupiter.api.Assertions.*;

class CacheSyncFixtureTest {
    @Test
    void ringHasExactOwnershipAndSignedWireKeys() throws Exception {
        var union = new HashSet<Long>();
        for (int worker = 0; worker < 6; worker++) {
            var fixture = new CacheSyncFixture(worker, 6, 176, 65, 2);
            var response = fixture.response();
            assertEquals(306, response.getCacheKeysCount());
            assertEquals(response, EngineRpcService.CacheStatusPB.parseFrom(response.toByteArray()));
            assertEquals(2, response.getVersion());
            union.addAll(response.getCacheKeysMap().keySet());
        }
        assertEquals(6 * (176 + 65), union.size());
        assertTrue(union.stream().anyMatch(key -> key < 0));
    }

    @Test
    void SmallDiffAndFreezePreserveCountsAndFullResponses() {
        var fixture = new CacheSyncFixture(0, 6, 176, 65, 2);
        var before = fixture.response();
        fixture.churn(17, 3);
        assertSame(before, fixture.response());
        var after = fixture.response();
        var added = new HashSet<>(after.getCacheKeysMap().keySet());
        added.removeAll(before.getCacheKeysMap().keySet());
        var removed = new HashSet<>(before.getCacheKeysMap().keySet());
        removed.removeAll(after.getCacheKeysMap().keySet());
        assertEquals(17, added.size());
        assertEquals(17, removed.size());
        assertEquals(306, after.getCacheKeysCount());
        assertEquals(3, after.getVersion());
        fixture.churn(0, 0);
        assertSame(after, fixture.response());
        assertEquals(17L, fixture.snapshot().get("added_keys"));
        assertEquals(4L * 306, fixture.snapshot().get("entries"));
    }

    @Test
    void invalidGeometryAndUnboundedDiffAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new CacheSyncFixture(0, 2, 1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new CacheSyncFixture(0, 6, 1_000_001, 0, 2));
        var fixture = new CacheSyncFixture(0, 6, 100, 10, 2);
        assertThrows(IllegalArgumentException.class, () -> fixture.churn(101, 1));
        assertThrows(IllegalArgumentException.class, () -> fixture.churn(10, 0));
    }
    @Test
    void everyRequestedVersionKeepsTheFullRealEnginePayload() {
        var fixture = new CacheSyncFixture(0, 6, 176, 65, 2, 0, () -> 0L);
        assertEquals(306, fixture.response(-1).getCacheKeysCount());
        assertEquals(306, fixture.response(2).getCacheKeysCount());
        assertEquals(2, fixture.response(2).getVersion());
        assertEquals(306, fixture.response(3).getCacheKeysCount());
        assertEquals(0L, fixture.snapshot().get("unchanged_responses"));
        assertEquals(4L, fixture.snapshot().get("full_responses"));
        assertEquals(4L * 306, fixture.snapshot().get("entries"));
    }

    @Test
    void legacyOmissionPropertyCannotChangeTheWireContract() {
        String property = "flexlb.mock.cache.omit-unchanged";
        String previous = System.getProperty(property);
        try {
            System.setProperty(property, "true");
            var fixture = new CacheSyncFixture(0, 6, 176, 65, 2);
            assertEquals(306, fixture.wireResponse(2).toProto().getCacheKeysCount());
            assertEquals(0L, fixture.snapshot().get("unchanged_responses"));
        } finally {
            if (previous == null) System.clearProperty(property);
            else System.setProperty(property, previous);
        }
    }

    @Test
    void clockChurnIsIndependentFromPollingFrequencyAndCanBeFrozen() {
        var now = new java.util.concurrent.atomic.AtomicLong();
        var first = new CacheSyncFixture(0, 6, 176, 65, 2, 1000, now::get);
        var second = new CacheSyncFixture(0, 6, 176, 65, 2, 1000, now::get);
        first.churn(17, 2); second.churn(17, 2);
        for (int i = 0; i < 40; i++) second.response(2);
        now.set(1001);
        var full = first.response(2);
        var changed = second.response(2);
        assertEquals(full.getCacheKeysMap(), changed.getCacheKeysMap());
        assertEquals(17L, first.snapshot().get("added_keys"));
        assertEquals(17L, second.snapshot().get("added_keys"));
        second.churn(0, 0);
        now.set(10000);
        second.response(-1);
        assertEquals(17L, second.snapshot().get("added_keys"));
    }

    @Test
    void controlClockUsesElapsedTimeAndValidatesItsBudget() {
        var now = new java.util.concurrent.atomic.AtomicLong();
        var fixture = new CacheSyncFixture(0, 6, 176, 65, 2, 0, now::get);
        assertThrows(IllegalArgumentException.class, () -> fixture.clockInterval(0));
        fixture.clockInterval(6000); fixture.churn(17, 1);
        for (int i = 0; i < 40; i++) fixture.response();
        assertEquals(0L, fixture.snapshot().get("added_keys"));
        now.set(12_000_000_001L);
        fixture.response();
        assertEquals(34L, fixture.snapshot().get("added_keys"));
        assertEquals(4L, fixture.version());
    }

    @Test
    void gradualGrowthKeepsCountersMonotonicAndReachesTheSameRing() {
        var gradual = new CacheSyncFixture(0, 6, 1, 0, 2);
        gradual.response();
        gradual.grow(88, 32); gradual.response();
        gradual.grow(176, 65);
        var full = new CacheSyncFixture(0, 6, 176, 65, 2);
        assertEquals(full.response().getCacheKeysMap(), gradual.response().getCacheKeysMap());
        assertEquals(305L, gradual.snapshot().get("added_keys"));
        assertEquals(0L, gradual.snapshot().get("removed_keys"));
        assertEquals(3L, gradual.snapshot().get("responses"));
        assertThrows(IllegalArgumentException.class, () -> gradual.grow(100, 65));
        gradual.churn(17, 1);
        assertThrows(IllegalArgumentException.class, () -> gradual.grow(200, 65));
    }

}
