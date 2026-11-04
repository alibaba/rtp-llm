package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillRequestQueueContractTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void replacementPreservesExactOwnershipAndRejectsAnExpiredOwner(boolean alternative) {
        var order =
                Comparator.comparingLong(
                        org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem::enqueueSeq);
        PrefillRequestQueue queue =
                alternative ? new ListQueue(order) : PrefillActiveIndex.ordered(4, order);
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), queue, () -> 100L);
        var head = route(1, 1);
        var next = route(2, 2);
        var sameDecode = mock(DecodeEndpoint.class);
        when(head.decodeEp()).thenReturn(sameDecode);
        when(next.decodeEp()).thenReturn(sameDecode);
        org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, head);
        org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, next);
        state.resourceLock().lock();
        try {
            assertTrue(state.selectionCurrentLocked(head, next));
            when(head.requestExpired(100)).thenReturn(true);
            assertFalse(state.selectionCurrentLocked(head, next));
        } finally {
            state.resourceLock().unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void replacementKeepsExactOwnershipAndStableSnapshots(boolean alternative) {
        var order =
                Comparator.comparingLong(
                        org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem::enqueueSeq);
        PrefillRequestQueue queue =
                alternative ? new ListQueue(order) : PrefillActiveIndex.ordered(4, order);
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), queue, () -> 100L);
        var first = route(1, 1);
        var second = route(2, 2);
        var stale = route(1, 1);
        assertSame(second, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, second));
        assertSame(first, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, first));
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, stale));
        var snapshot = queue.capture();
        long version = queue.version();
        assertSame(first, queue.peek());
        assertEquals(
                List.of(1L, 2L),
                snapshot.items().stream().map(item -> Long.parseLong(item.requestId())).toList());
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, stale));
        assertEquals(version, queue.version());
        when(first.requestExpired(100)).thenReturn(true);
        assertSame(first, queue.peek(), "expiry does not remove queue membership");
        assertSame(
                first, org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, first));
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, first));
        assertTrue(queue.version() > version);
        assertEquals(List.of(second), state.closeQueue());
        assertTrue(state.queueStopped());
        assertTrue(state.closeQueue().isEmpty(), "only the first closer receives the members");
        when(stale.prefillEp()).thenReturn(state);
        assertFalse(state.offerQueued(stale), "closing the queue stops new membership");
        verify(first, never()).ctx();
        verify(second, never()).ctx();
        assertTrue(queue.isEmpty());
        assertEquals(
                2L,
                state.admissionSummary(0, 0).occupiedRequests(),
                "queue removal does not settle either request");
        assertTrue(state.releaseRequest(first));
        assertTrue(state.releaseRequest(second));
        assertEquals(0L, state.admissionSummary(0, 0).occupiedRequests());
        assertEquals(0, state.stats().locallyOwnedRequests());
        assertEquals(2, snapshot.items().size());
        assertThrows(UnsupportedOperationException.class, () -> snapshot.items().clear());
    }

    @Test
    void rejectedInsertionDoesNotPublishOrRetainAReservation() {
        var queue = mock(PrefillRequestQueue.class);
        when(queue.enabled()).thenReturn(true);
        var endpoint = EndpointTestSupport.prefillEndpoint(new ReentrantLock(), queue, () -> 100L);
        var rejected = route(1, 1);
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(endpoint, rejected));
        verify(rejected, never()).attachPrefillReservation(any());
        assertEquals(0L, endpoint.admissionSummary(0, 0).occupiedRequests());
        assertFalse(endpoint.releaseRequest(rejected));

        var retry = route(1, 2);
        when(queue.add(retry)).thenReturn(true);
        assertSame(retry, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(endpoint, retry));
        assertEquals(1L, endpoint.admissionSummary(0, 0).occupiedRequests());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queueOwnsBlockedHeadOrderAndSuccessfulDispatchAllowance(boolean alternative) {
        Comparator<PrefillRequestQueue.QueueItem> order =
                Comparator.comparingLong(PrefillRequestQueue.QueueItem::enqueueSeq);
        PrefillRequestQueue queue =
                alternative ? new ListQueue(order) : PrefillActiveIndex.ordered(4, order);
        var state = EndpointTestSupport.prefillEndpoint(new ReentrantLock(), queue, () -> 100L);
        var head = route(1, 1);
        var candidate = route(2, 2);
        when(head.decodeEp()).thenReturn(mock(DecodeEndpoint.class));
        when(candidate.decodeEp()).thenReturn(mock(DecodeEndpoint.class));
        EndpointTestSupport.enqueue(state, head);
        EndpointTestSupport.enqueue(state, candidate);
        assertEquals(
                alternative ? List.of() : List.of(candidate),
                state.captureQueue(10, head, true).items());
        assertTrue(state.captureQueue(10, head, false).items().isEmpty());
        // A failed preparation sends no dispatch event and retains the same candidates.
        assertEquals(
                alternative ? List.of() : List.of(candidate),
                state.captureQueue(10, head, true).items());
        state.resourceLock().lock();
        try {
            queue.recordDispatch(head);
        } finally {
            state.resourceLock().unlock();
        }
        assertTrue(state.captureQueue(10, head, true).items().isEmpty());
        assertEquals(List.of(head, candidate), state.captureQueue(10).items());
        assertEquals(
                2L,
                state.admissionSummary(0, 0).occupiedRequests(),
                "queue decisions never settle requests");
    }

    private static RequestRoute route(long id, long sequence) {
        var route = mock(RequestRoute.class);
        when(route.requestId()).thenReturn(String.valueOf(id));
        when(route.enqueueSeq()).thenReturn(sequence);
        when(route.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return route;
    }

    /** Deliberately uses different storage from PrefillActiveIndex. */
    private static final class ListQueue implements PrefillRequestQueue {
        private final List<RequestRoute> items = new ArrayList<>();
        private final Comparator<org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem> order;
        private long version;

        ListQueue(Comparator<org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem> order) {
            this.order = order;
        }

        public boolean enabled() {
            return true;
        }

        public long version() {
            return version;
        }

        public boolean add(RequestRoute route) {
            if (contains(route)) return false;
            items.add(route);
            items.sort(order);
            version++;
            return true;
        }

        public boolean remove(RequestRoute route) {
            for (int i = 0; i < items.size(); i++) {
                if (items.get(i) == route) {
                    items.remove(i);
                    version++;
                    return true;
                }
            }
            return false;
        }

        public boolean contains(RequestRoute route) {
            return items.stream().anyMatch(item -> item == route);
        }

        public RequestRoute peek() {
            return items.isEmpty() ? null : items.getFirst();
        }

        public boolean isEmpty() {
            return items.isEmpty();
        }

        public int size() {
            return items.size();
        }

        public int size(int priority) {
            return (int) items.stream().filter(item -> item.priority() == priority).count();
        }

        public void clear() {
            if (!items.isEmpty()) {
                items.clear();
                version++;
            }
        }

        public Iterator<RequestRoute> iterator() {
            return List.copyOf(items).iterator();
        }

        public Snapshot capture() {
            var copy = List.copyOf(items);
            return new Snapshot(copy, order);
        }
    }
}
