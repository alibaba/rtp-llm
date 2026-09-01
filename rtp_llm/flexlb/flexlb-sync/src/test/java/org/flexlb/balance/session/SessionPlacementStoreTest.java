package org.flexlb.balance.session;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SessionPlacementStoreTest {

    @Test
    void isolatesModelsAndExpiresPlacements() {
        AtomicLong now = new AtomicLong(1_000L);
        SessionPlacementStore store = new SessionPlacementStore(10, now::get);

        record(store, "model-a", "session-1", "10.0.0.1:9000", 101L);
        record(store, "model-b", "session-1", "10.0.0.2:9000", 102L);

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

        record(store, "model", "session-1", "10.0.0.1:9000", 101L);
        record(store, "model", "session-2", "10.0.0.2:9000", 102L);
        record(store, "model", "session-3", "10.0.0.3:9000", 103L);
        record(store, "model", "x".repeat(257), "10.0.0.4:9000", 104L);
        store.cleanUp();

        assertTrue(store.estimatedSize() <= 2);
        assertTrue(store.find("model", "x".repeat(257), 500L).isEmpty());
    }

    @Test
    void invalidationRemovesNewSessionPlacement() {
        SessionPlacementStore store = new SessionPlacementStore();
        record(store, "model", "session-1", "10.0.0.1:9000", 101L);

        store.reset("model", "session-1");

        assertTrue(store.find("model", "session-1", 1_000L).isEmpty());
    }

    @Test
    void rejectsCompletionPredatingSessionReset() {
        SessionPlacementStore store = new SessionPlacementStore();
        long oldEpoch = store.currentEpoch("model", "session-1");
        long newEpoch = store.reset("model", "session-1");

        store.record("model", "session-1", "10.0.0.1:9000", 101L, oldEpoch);
        assertTrue(store.find("model", "session-1", 1_000L).isEmpty());

        store.record("model", "session-1", "10.0.0.2:9000", 102L, newEpoch);
        assertEquals("10.0.0.2:9000",
                store.find("model", "session-1", 1_000L).orElseThrow().ipPort());
    }

    @Test
    void missingStateRejectsStaleCompletion() {
        SessionPlacementStore store = new SessionPlacementStore();

        long epoch = store.currentEpoch("model", "session-1");
        assertTrue(epoch > 0L);
        assertEquals(epoch, store.currentEpoch("model", "session-1"));
        store.record("model", "session-2", "10.0.0.1:9000", 101L, 0L);

        assertTrue(store.find("model", "session-2", 1_000L).isEmpty());
    }

    private static void record(SessionPlacementStore store, String model, String sessionId,
                               String ipPort, long requestId) {
        store.record(model, sessionId, ipPort, requestId, store.currentEpoch(model, sessionId));
    }
}
