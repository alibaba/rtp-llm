package org.flexlb.balance.endpoint;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.RequestRoute;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.balance.endpoint.PrefillState.QueueChange.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class WorkerQueueContractTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void replacementMayForbidBypassButCannotBypassAnExpiredOwner(boolean alternative) {
        var order = Comparator.comparingLong(RequestRoute::enqueueSeq);
        WorkerQueue queue =
                alternative ? new ListQueue(order) : PrefillActiveIndex.ordered(4, order);
        var state = new PrefillState(new ReentrantLock(), queue, () -> 100L);
        var head = route(1, 1);
        var next = route(2, 2);
        when(head.decodeEp()).thenReturn(mock(DecodeEndpoint.class));
        when(next.decodeEp()).thenReturn(mock(DecodeEndpoint.class));
        state.updateQueue(ENQUEUE, head, 100);
        state.updateQueue(ENQUEUE, next, 100);
        state.queueGuard().lock();
        try {
            assertEquals(!alternative, state.canBypassLocked(head, next));
            when(head.requestExpired(100)).thenReturn(true);
            assertFalse(state.canBypassLocked(head, next));
        } finally {
            state.queueGuard().unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void replacementKeepsExactOwnershipAndStableSnapshots(boolean alternative) {
        var order = Comparator.comparingLong(RequestRoute::enqueueSeq);
        WorkerQueue queue =
                alternative ? new ListQueue(order) : PrefillActiveIndex.ordered(4, order);
        var state = new PrefillState(new ReentrantLock(), queue, () -> 100L);
        var first = route(1, 1);
        var second = route(2, 2);
        var stale = route(1, 1);
        assertSame(second, state.updateQueue(ENQUEUE, second, 100));
        assertSame(first, state.updateQueue(ENQUEUE, first, 100));
        assertNull(state.updateQueue(ENQUEUE, stale, 100));
        var snapshot = queue.capture();
        long version = queue.version();
        assertSame(first, queue.peek());
        assertEquals(
                List.of(1L, 2L),
                snapshot.projectedItems().stream().map(GroupPlanner.Item::requestId).toList());
        assertNull(state.updateQueue(REMOVE, stale, 100));
        assertEquals(version, queue.version());
        when(first.requestExpired(100)).thenReturn(true);
        assertNull(state.updateQueue(REMOVE_UNEXPIRED, first, 100));
        assertSame(first, state.updateQueue(REMOVE, first, 100));
        assertNull(state.updateQueue(REMOVE, first, 100));
        assertTrue(queue.version() > version);
        assertSame(second, state.updateQueue(STOP, null, 100));
        assertTrue(queue.isEmpty());
        assertSame(second, state.updateQueue(ACK_STOP, second, 100));
        assertEquals(0, state.stats().locallyOwnedRequests());
        assertEquals(2, snapshot.projectedItems().size());
        assertThrows(UnsupportedOperationException.class, () -> snapshot.projectedItems().clear());
    }

    private static RequestRoute route(long id, long sequence) {
        var route = mock(RequestRoute.class);
        when(route.requestId()).thenReturn(id);
        when(route.enqueueSeq()).thenReturn(sequence);
        when(route.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return route;
    }

    /** Deliberately uses different storage from PrefillActiveIndex. */
    private static final class ListQueue implements WorkerQueue {
        private final List<RequestRoute> items = new ArrayList<>();
        private final Comparator<RequestRoute> order;
        private long version;

        ListQueue(Comparator<RequestRoute> order) {
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
            var copy =
                    items.stream()
                            .map(
                                    item ->
                                            new GroupPlanner.Item(
                                                    item.requestId(),
                                                    item.priority(),
                                                    item.enqueueSeq(),
                                                    item.enqueuedAtMs(),
                                                    item.expiresAtMs(),
                                                    item.seqLen(),
                                                    item.hitCache()))
                            .toList();
            return () -> copy;
        }
    }
}
