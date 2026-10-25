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
}
