package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks queue ordering, exact item identity and immutable captures while additions and removals change the live queue.
 */
class PrefillRequestQueueTest {
    @Test
    void acceptanceOrderIgnoresPriorityAndOriginalEnqueueSequence() {
        var queue = PrefillRequestQueue.fifo(4);
        var first = item(30, 10);
        var second = item(10, 90);
        var third = item(20, 50);
        queue.add(first);
        queue.add(second);
        queue.add(third);
        assertEquals(List.of(30L, 10L, 20L), ids(queue.capture()));
        assertSame(first, queue.peek());
    }

    @Test
    void removingMiddleItemPreservesOrderAndReceivingItAgainAppendsIt() {
        var queue = PrefillRequestQueue.fifo(4);
        var first = item(30, 10);
        var middle = item(10, 90);
        var last = item(20, 50);
        queue.add(first);
        queue.add(middle);
        queue.add(last);
        var beforeRemoval = queue.capture();
        assertTrue(queue.remove(middle));
        assertEquals(List.of(30L, 20L), ids(queue.capture()));
        assertTrue(queue.add(middle));
        assertEquals(List.of(30L, 20L, 10L), ids(queue.capture()));
        assertEquals(List.of(30L, 10L, 20L), ids(beforeRemoval));
    }

    @Test
    void directModeNeverAcquiresQueueMembership() {
        var index = PrefillRequestQueue.disabled();
        var request = item(1, 50);
        var empty = index.capture();
        assertThrows(IllegalStateException.class, () -> index.add(request));
        assertThrows(IllegalStateException.class, () -> index.add(null));
        assertFalse(index.contains(request));
        assertFalse(index.remove(request));
        index.clear();
        assertNull(index.peek());
        assertTrue(index.isEmpty());
        assertEquals(0, index.size());
        assertEquals(0, index.size(-1));
        assertEquals(0, index.size(101));
        assertEquals(0L, index.version());
        assertSame(empty, index.capture());
        assertTrue(empty.items().isEmpty());
        assertFalse(index.iterator().hasNext());
        assertSame(index, PrefillRequestQueue.disabled());
    }

    @Test
    void capturedEmptinessDoesNotMaterializeOrFollowLaterQueueChanges() {
        var index = PrefillRequestQueue.fifo(4);
        var request = item(1, 50);
        when(request.seqLen())
                .thenThrow(new AssertionError("must not materialize prediction input"));
        var empty = index.capture();
        index.add(request);
        var populated = index.capture();
        assertEquals(0, capturedSize(empty));
        assertEquals(1, capturedSize(populated));
        index.clear();
        assertEquals(1, capturedSize(populated));
        assertEquals(0, capturedSize(index.capture()));
    }

    @Test
    void additionsAndRemovalsMatchReferenceOrderAndPreserveOldCaptures() {
        var index = PrefillRequestQueue.fifo(16);
        var expected = new ArrayList<RouteResult>();
        var random = new Random(36);
        for (int step = 0; step < 400; step++) {
            long oldVersion = index.version();
            var old = index.capture();
            var oldItems = List.copyOf(expected);
            if (expected.isEmpty() || random.nextBoolean()) {
                var request = item(step, random.nextInt(4));
                assertTrue(index.add(request));
                long addedVersion = index.version();
                assertFalse(index.add(request));
                assertEquals(addedVersion, index.version());
                expected.add(request);
            } else {
                var request = expected.remove(random.nextInt(expected.size()));
                assertTrue(index.remove(request));
                assertFalse(index.contains(request));
                long removedVersion = index.version();
                assertFalse(index.remove(request));
                assertEquals(removedVersion, index.version());
            }
            assertTrue(index.version() > oldVersion);
            assertEquals(
                    oldItems.stream().map(item -> Long.parseLong(item.requestId())).toList(),
                    ids(old));
            assertEquals(
                    expected.stream().map(item -> Long.parseLong(item.requestId())).toList(),
                    ids(index.capture()));
            for (int priority = 0; priority <= 100; priority++) {
                int exactPriority = priority;
                assertEquals(
                        expected.stream().filter(item -> item.priority() == exactPriority).count(),
                        index.size(priority));
            }
            assertSame(index.capture(), index.capture());
            assertEquals(expected.isEmpty() ? null : expected.getFirst(), index.peek());
        }
        var beforeClear = index.capture();
        long beforeClearVersion = index.version();
        index.clear();
        assertEquals(!expected.isEmpty(), index.version() > beforeClearVersion);
        long clearedVersion = index.version();
        index.clear();
        assertEquals(clearedVersion, index.version());
        for (int priority = 0; priority <= 100; priority++) {
            assertEquals(0, index.size(priority));
        }
        assertTrue(index.capture().items().isEmpty());
        assertEquals(
                expected.stream().map(item -> Long.parseLong(item.requestId())).toList(),
                ids(beforeClear));
    }

    @Test
    void equalOrderingKeysStillRequireExactIdentity() {
        var index = PrefillRequestQueue.fifo(2);
        var first = item(1, 50);
        var second = item(1, 50);
        assertTrue(index.add(first));
        assertTrue(index.add(second));
        assertFalse(index.remove(item(1, 50)));
        var ordered = new ArrayList<RouteResult>();
        index.forEach(ordered::add);
        assertEquals(List.of(first, second), ordered);
        var captured = index.capture();
        var projected = captured.items();
        assertEquals(2, projected.size());
        assertSame(first, projected.getFirst());
        assertSame(second, projected.getLast());
        assertNotSame(projected.getFirst(), projected.getLast());
        assertTrue(index.remove(first));
        assertSame(second, index.peek());
        assertSame(projected.getLast(), index.capture().items().getFirst());
        assertSame(projected, captured.items());
        assertEquals(2, captured.items().size());
    }

    @Test
    @Timeout(10)
    void concurrentCapturesShareMaterializationAndUnchangedRequestValues() throws Exception {
        var index = PrefillRequestQueue.fifo(2);
        var first = item(1, 50);
        index.add(first);
        var old = index.capture();
        index.add(item(2, 40));
        var current = index.capture();
        try (var pool = Executors.newFixedThreadPool(16)) {
            var start = new CountDownLatch(1);
            var tasks = new ArrayList<java.util.concurrent.Future<?>>();
            for (int i = 0; i < 64; i++) {
                var capture = i % 2 == 0 ? old : current;
                tasks.add(
                        pool.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    return capture.items();
                                }));
            }
            start.countDown();
            for (int i = 0; i < tasks.size(); i++) {
                assertSame(
                        (i % 2 == 0 ? old : current).items(),
                        tasks.get(i).get(5, TimeUnit.SECONDS));
            }
        }
        assertSame(old.items().getFirst(), current.items().getFirst());
        verify(first, org.mockito.Mockito.never()).seqLen();
        assertThrows(UnsupportedOperationException.class, () -> current.items().clear());
    }

    @Test
    void captureDoesNotReadPlanningInputs() {
        var index = PrefillRequestQueue.fifo(2);
        var first = item(1, 50);
        var second = item(2, 40);
        when(second.seqLen()).thenThrow(new IllegalStateException("test failure")).thenReturn(10L);
        index.add(first);
        index.add(second);
        var captured = index.capture();
        assertEquals(2, captured.items().size());
        assertSame(captured.items(), captured.items());
        verify(first, org.mockito.Mockito.never()).seqLen();
        verify(second, org.mockito.Mockito.never()).seqLen();
    }

    private static List<Long> ids(PrefillRequestQueue.Snapshot capture) {
        return capture.items().stream().map(route -> Long.parseLong(route.requestId())).toList();
    }

    @SuppressWarnings("unchecked")
    private static int capturedSize(PrefillRequestQueue.Snapshot capture) {
        return capture.items().size();
    }

    private static RouteResult item(long id, int priority) {
        var request = mock(RouteResult.class);
        when(request.requestId()).thenReturn(String.valueOf(id));
        when(request.enqueueSeq()).thenReturn(id);
        when(request.priority()).thenReturn(priority);
        when(request.seqLen()).thenReturn(10L);
        when(request.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return request;
    }
}
