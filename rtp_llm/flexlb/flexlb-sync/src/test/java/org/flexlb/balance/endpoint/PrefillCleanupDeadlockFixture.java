package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Package-local admission APIs exercised through real committed requests. */
public final class PrefillCleanupDeadlockFixture {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock,
                    PrefillActiveIndex.ordered(
                            4,
                            Comparator.comparingLong(
                                    org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem
                                            ::requestId)),
                    clock::get);
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});
    private final RequestRoute next;

    public PrefillCleanupDeadlockFixture(long requestId, boolean batch) {
        RequestRoute first = item(requestId);
        next = item(requestId + 1);
        if (batch) {
            enqueue(first);
            {
                var reservation = state.reserveBatch(1L, 2, generation.tryPinGeneration()).value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation);
                    try (var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, reservation, List.of(first), 20L)) {
                        assertNotNull(handoff);
                    }
                }
            }
        } else {
            {
                var reservation = state.reserveUnqueuedRoute(first, 20L, 0L).value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation);
                    try (var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, first, reservation)) {
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
            var reservation = state.reserveBatch(2L, 2, generation.tryPinGeneration()).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertNotNull(reservation);
            }
        }
    }

    private void enqueue(RequestRoute item) {
        lock.lock();
        try {
            assertSame(item, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item));
        } finally {
            lock.unlock();
        }
    }

    private static RequestRoute item(long id) {
        RequestRoute item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(id);
        when(item.seqLen()).thenReturn(100L);
        return item;
    }
}
