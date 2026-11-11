package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.RouteResult;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks queue implementations preserve exact request identity and stable snapshots during validation, replacement and removal.
 */
class PrefillRequestQueueContractTest {
    @Test
    void invalidMemberDoesNotPublishQueueState() {
        var queue = PrefillRequestQueue.fifo(4);
        var accepted = route(1, 1);
        var invalid = route(2, 2);
        queue.add(accepted);
        var before = queue.capture();
        long revision = queue.version();
        when(invalid.priority()).thenReturn(-1);

        assertThrows(IndexOutOfBoundsException.class, () -> queue.add(invalid));

        assertFalse(queue.contains(invalid));
        assertEquals(1, queue.size());
        assertSame(accepted, queue.peek());
        assertEquals(revision, queue.version());
        assertSame(before, queue.capture());
    }

    @Test
    void queueBusinessOperationsValidateAndRemoveExactMembersWithoutCallerLock() {
        var queue = PrefillRequestQueue.fifo(4);
        var state = EndpointTestSupport.prefillEndpoint(new ReentrantLock(), queue, () -> 100L);
        var head = route(1, 1);
        var next = route(2, 2);
        var stale = route(1, 1);
        EndpointTestSupport.enqueue(state, head);
        EndpointTestSupport.enqueue(state, next);
        var snapshot = state.captureQueue(10);
        assertSame(head, state.captureQueue(1).head());
        assertFalse(state.releaseQueuedRequest(stale));
        assertEquals(snapshot.queueVersion(), state.captureQueue(1).queueVersion());
        assertTrue(state.releaseQueuedRequest(head));
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertSame(next, state.captureQueue(1).head());
        assertFalse(state.releaseQueuedRequest(head));
        assertTrue(state.releaseQueuedRequest(next));
        assertEquals(List.of(head, next), snapshot.items());
        assertTrue(state.captureQueue(10).items().isEmpty());
        assertEquals(
                0L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertFalse(state.releaseRequest(head));
        assertFalse(state.releaseRequest(next));
        verify(head, never()).ctx();
        verify(next, never()).ctx();
    }

    @Test
    void removalKeepsExactIdentityAndStableSnapshots() {
        PrefillRequestQueue queue = PrefillRequestQueue.fifo(4);
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
        assertSame(second, queue.peek());
        assertEquals(
                List.of(2L, 1L),
                snapshot.items().stream().map(item -> Long.parseLong(item.requestId())).toList());
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, stale));
        assertEquals(version, queue.version());
        when(first.requestExpired(100)).thenReturn(true);
        assertSame(second, queue.peek(), "expiry does not remove queue membership");
        assertSame(
                first, org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, first));
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(state, first));
        assertTrue(queue.version() > version);
        assertEquals(List.of(second), state.closeQueue());
        assertTrue(state.queueStopped());
        assertTrue(state.closeQueue().isEmpty(), "only the first closer receives the members");
        when(stale.prefillEndpoint()).thenReturn(state);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.tryEnqueueRoute(state, stale),
                "closing the queue stops new membership");
        verify(first, never()).ctx();
        verify(second, never()).ctx();
        assertTrue(queue.isEmpty());
        assertEquals(
                2L,
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state),
                "queue removal does not settle either request");
        assertTrue(state.releaseRequest(first));
        assertTrue(state.releaseRequest(second));
        assertEquals(
                0L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertEquals(0, state.stats().locallyOwnedRequests());
        assertEquals(2, snapshot.items().size());
        assertThrows(UnsupportedOperationException.class, () -> snapshot.items().clear());
    }

    @Test
    void selectionMustBeAContinuousPrefixOfTheAcceptedRequests() {
        var queue = PrefillRequestQueue.fifo(4);
        var state = EndpointTestSupport.prefillEndpoint(new ReentrantLock(), queue, () -> 100L);
        var first = route(30, 30);
        var second = route(10, 10);
        var third = route(20, 20);
        EndpointTestSupport.enqueue(state, first);
        EndpointTestSupport.enqueue(state, second);
        EndpointTestSupport.enqueue(state, third);
        for (var invalid :
                List.of(List.of(second), List.of(first, third), List.of(second, first))) {
            var result =
                    state.tryStartQueuedDelivery(
                            new Delivery.Selection(
                                    invalid, false, 0L, 0, 0L, null, new long[invalid.size()]));
            assertFalse(result.accepted());
            assertEquals(List.of(first, second, third), state.captureQueue(10).items());
        }
        assertTrue(
                state.tryStartQueuedDelivery(
                                new Delivery.Selection(
                                        List.of(first, second),
                                        false,
                                        0L,
                                        0,
                                        0L,
                                        null,
                                        new long[2]))
                        .accepted());
        assertEquals(List.of(third), state.captureQueue(10).items());
    }

    @Test
    void rejectedInsertionDoesNotPublishOrRetainAReservation() {
        var queue = mock(PrefillRequestQueue.class);
        when(queue.enabled()).thenReturn(true);
        when(queue.capture()).thenReturn(new PrefillRequestQueue.Snapshot(List.of()));
        var endpoint = EndpointTestSupport.prefillEndpoint(new ReentrantLock(), queue, () -> 100L);
        var rejected = route(1, 1);
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(endpoint, rejected));
        assertEquals(
                0L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(endpoint));
        assertFalse(endpoint.releaseRequest(rejected));

        var retry = route(1, 2);
        when(queue.add(retry)).thenReturn(true);
        assertSame(retry, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(endpoint, retry));
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(endpoint));
    }

    private static RouteResult route(long id, long sequence) {
        var route = mock(RouteResult.class);
        when(route.requestId()).thenReturn(String.valueOf(id));
        when(route.enqueueSeq()).thenReturn(sequence);
        when(route.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return route;
    }
}
