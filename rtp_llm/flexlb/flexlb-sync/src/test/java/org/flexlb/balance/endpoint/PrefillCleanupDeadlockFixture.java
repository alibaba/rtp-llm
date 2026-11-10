package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteResult;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Package-local admission APIs exercised through real committed requests.
 */
public final class PrefillCleanupDeadlockFixture {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock, OrderedPrefillRequestQueue.fifo(4), clock::get);
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});
    private final RouteResult next;

    public PrefillCleanupDeadlockFixture(long requestId, boolean batch) {
        RouteResult first = item(requestId);
        next = item(requestId + 1);
        if (batch) {
            enqueue(first);
            {
                var reservation =
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(1L, 2);
                {
                    assertNotNull(reservation);
                    {
                        var handoff =
                                EndpointTestSupport.commitBatch(
                                        state, reservation, List.of(first), 20L);

                        assertNotNull(handoff);
                    }
                }
            }
        } else {
            {
                var reservation =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                        state, first, 20L, 0L)
                                .value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation);
                    {
                        var handoff =
                                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                        state, first);

                        assertNotNull(handoff);
                    }
                }
            }
        }
        enqueue(next);
        clock.set(200);
    }

    public void sweepBatches(Predicate<String> retain) {
        assertEquals(0, EndpointTestSupport.evictPrefill(state, 10L, retain));
        assertEquals(1, state.stats().batchCount());
    }

    public void sweepIndividuals(Predicate<String> retain) {
        assertEquals(0, EndpointTestSupport.evictPrefill(state, 10L, retain));
        assertEquals(1, state.stats().individuallyOwnedRequests());
    }

    public void reserveNextBatch() {
        {
            var reservation =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(2L, 2);
            {
                assertNotNull(reservation);
            }
        }
    }

    private void enqueue(RouteResult item) {
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            item,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item));
                });
    }

    private static RouteResult item(long id) {
        RouteResult item = mock(RouteResult.class);
        when(item.requestId()).thenReturn(String.valueOf(id));
        when(item.seqLen()).thenReturn(100L);
        return item;
    }
}
