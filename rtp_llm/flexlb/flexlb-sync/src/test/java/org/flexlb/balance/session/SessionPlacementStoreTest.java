package org.flexlb.balance.session;

import org.flexlb.config.RoutingConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SessionPlacementStoreTest {

    @Test
    void isolatesModelsAndExpiresPlacements() {
        AtomicLong now = new AtomicLong(1_000L);
        SessionPlacementStore store = new SessionPlacementStore(10, now::get);

        store.record("model-a", "session-1", "10.0.0.1:9000");
        store.record("model-b", "session-1", "10.0.0.2:9000");

        assertEquals("10.0.0.1:9000",
                store.find("model-a", "session-1", 500L).orElseThrow().ipPort());
        assertEquals("10.0.0.2:9000",
                store.find("model-b", "session-1", 500L).orElseThrow().ipPort());

        now.addAndGet(501L);
        assertTrue(store.find("model-a", "session-1", 500L).isEmpty());
    }

    @Test
    void boundsRetainedPlacementsAndRejectsOversizedSessionIds() {
        AtomicLong now = new AtomicLong(1_000L);
        SessionPlacementStore store = new SessionPlacementStore(2, now::get);

        store.record("model", "session-1", "10.0.0.1:9000");
        store.record("model", "session-2", "10.0.0.2:9000");
        store.record("model", "session-3", "10.0.0.3:9000");
        store.record("model", "x".repeat(257), "10.0.0.4:9000");
        store.cleanUp();

        assertTrue(store.estimatedSize() <= 2);
        assertTrue(store.find("model", "x".repeat(257), 500L).isEmpty());
    }

    @Test
    void rejectsSessionIdsOutsideTheAsciiWireContract() {
        SessionPlacementStore store = new SessionPlacementStore();

        store.record("model", "contains space", "10.0.0.1:9000");
        store.record("model", "emoji_😀", "10.0.0.2:9000");

        assertTrue(store.find("model", "contains space", 500L).isEmpty());
        assertTrue(store.find("model", "emoji_😀", 500L).isEmpty());
    }

    @Test
    void expiresIdlePlacements() {
        AtomicLong now = new AtomicLong(1_000L);
        SessionPlacementStore store = new SessionPlacementStore(10, now::get);

        store.record("model", "session-1", "10.0.0.1:9000");
        assertEquals(1L, store.estimatedSize());

        now.addAndGet(RoutingConfig.SessionAffinityConfig.MAX_TTL_MS + 1L);
        store.cleanUp();

        assertEquals(0L, store.estimatedSize());
    }

    @Test
    void invalidationRemovesNewSessionPlacement() {
        SessionPlacementStore store = new SessionPlacementStore();
        store.record("model", "session-1", "10.0.0.1:9000");

        store.invalidate("model", "session-1");

        assertTrue(store.find("model", "session-1", 1_000L).isEmpty());
    }

    @Test
    void usesConfiguredMaximumSize() {
        FlexlbConfig config = new FlexlbConfig();
        RoutingConfig.SessionAffinityConfig affinity = new RoutingConfig.SessionAffinityConfig();
        affinity.setTtlMs(1_000L);
        affinity.setMaxEntries(2L);
        config.getRouter().getRoles().getPrefill().setSessionAffinity(affinity);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        SessionPlacementStore store = new SessionPlacementStore(configService);

        store.record("model", "session-1", "10.0.0.1:9000");
        store.record("model", "session-2", "10.0.0.2:9000");
        store.record("model", "session-3", "10.0.0.3:9000");
        store.cleanUp();

        assertTrue(store.estimatedSize() <= 2L);
    }
}
