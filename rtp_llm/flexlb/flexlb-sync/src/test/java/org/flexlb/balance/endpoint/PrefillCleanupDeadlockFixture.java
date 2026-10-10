package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Package-local admission APIs exercised through real committed requests. */
public final class PrefillCleanupDeadlockFixture {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(lock,
            PrefillActiveIndex.ordered(4, Comparator.comparingLong(RequestRoute::requestId)),
            clock::get);
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });
    private final RequestRoute next;

    public PrefillCleanupDeadlockFixture(long requestId, boolean batch) {
        RequestRoute first = item(requestId);
        next = item(requestId + 1);
        if (batch) {
            enqueue(first);
            {
                var reservation = state.reserveBatch(first, 1L, 2, generation.tryAcquireHandoff()).reservation();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation);
                    try (var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(first), 20L)) {
                        assertNotNull(handoff);
                    }
                }
            }
        } else {
            {
                var reservation = state.reserveUnqueuedRoute(first, 20L, 0L).reservation();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation);
                    try (var handoff = EndpointTestSupport.commitRoutes(state,
                            List.of(first), List.of(reservation), generation.tryAcquireHandoff())) {
                        assertNotNull(handoff);
                    }
                }
            }
        }
        enqueue(next);
        clock.set(200);
    }

    public void sweepBatches(LongPredicate retain) {
        assertEquals(0, EndpointTestSupport.evictPrefill(state, 10L, retain));
        assertEquals(1, state.stats().batchCount());
    }

    public void sweepIndividuals(LongPredicate retain) {
        assertEquals(0, EndpointTestSupport.evictPrefill(state, 10L, retain));
        assertEquals(1, state.stats().individuallyOwnedRequests());
    }

    public void reserveNextBatch() {
        {
            var reservation = state.reserveBatch(next, 2L, 2, generation.tryAcquireHandoff()).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertNotNull(reservation);
            }
        }
    }

    private void enqueue(RequestRoute item) {
        lock.lock();
        try { assertTrue(state.enqueueActiveLocked(item, Long.MAX_VALUE)); }
        finally { lock.unlock(); }
    }

    private static RequestRoute item(long id) {
        RequestRoute item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(id);
        when(item.seqLen()).thenReturn(100L);
        return item;
    }
}
