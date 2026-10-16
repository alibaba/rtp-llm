package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Package-local admission APIs exercised through real committed requests. */
public final class PrefillCleanupDeadlockFixture {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(lock,
            PrefillActiveIndex.ordered(4, Comparator.comparing(ScheduledRequest::requestId)),
            clock::get, () -> { });
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });
    private final ScheduledRequest next;

    public PrefillCleanupDeadlockFixture(long requestId, boolean batch) {
        ScheduledRequest first = item(requestId);
        next = item(requestId + 1);
        if (batch) {
            enqueue(first);
            try (var reservation = state.reserveBatch(first, 1L, 2, generation.tryAcquireHandoff()).reservation()) {
                assertNotNull(reservation);
                try (var handoff = reservation.commit(List.of(first), 20L)) {
                    assertNotNull(handoff);
                }
            }
        } else {
            try (var reservation = state.reserveUnqueuedRoute(first, 20L, 0L).reservation()) {
                assertNotNull(reservation);
                try (var handoff = state.commitRouteGroup(
                        List.of(first), List.of(reservation), generation.tryAcquireHandoff())) {
                    assertNotNull(handoff);
                }
            }
        }
        enqueue(next);
        clock.set(200);
    }

    public void sweepBatches(Predicate<String> retain) {
        assertEquals(0, state.evictExpiredBatches(10L, retain));
        assertEquals(1, state.stats().batchCount());
    }

    public void sweepIndividuals(Predicate<String> retain) {
        assertEquals(0, state.evictExpiredIndividuals(10L, retain));
        assertEquals(1, state.stats().individuallyOwnedRequests());
    }

    public void reserveNextBatch() {
        try (var reservation = state.reserveBatch(next, 2L, 2, generation.tryAcquireHandoff()).reservation()) {
            assertNotNull(reservation);
        }
    }

    private void enqueue(ScheduledRequest item) {
        lock.lock();
        try { assertTrue(state.enqueueActiveUnderLock(item, Long.MAX_VALUE)); }
        finally { lock.unlock(); }
    }

    private static ScheduledRequest item(long id) {
        ScheduledRequest item = mock(ScheduledRequest.class);
        when(item.requestId()).thenReturn(Long.toString(id));
        when(item.seqLen()).thenReturn(100L);
        return item;
    }
}
