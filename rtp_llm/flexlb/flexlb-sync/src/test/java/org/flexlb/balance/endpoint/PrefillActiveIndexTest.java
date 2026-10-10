package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillActiveIndexTest {
    private static final Comparator<RequestRoute> ORDER =
            Comparator.comparingInt(RequestRoute::priority).reversed()
                    .thenComparingLong(RequestRoute::enqueueSeq);

    @Test
    void directModeNeverAcquiresQueueMembership() {
        var index = PrefillActiveIndex.disabled();
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
        assertTrue(empty.projectedItems().isEmpty());
        assertFalse(index.iterator().hasNext());
        assertSame(index, PrefillActiveIndex.disabled());
    }

    @Test
    void capturedEmptinessDoesNotMaterializeOrFollowLaterQueueChanges() {
        var index = PrefillActiveIndex.ordered(4, ORDER);
        var request = item(1, 50);
        when(request.seqLen()).thenThrow(new AssertionError("must not materialize prediction input"));
        var empty = index.capture();
        index.add(request);
        var populated = index.capture();
        assertTrue(empty.isEmpty());
        assertFalse(populated.isEmpty());
        index.clear();
        assertFalse(populated.isEmpty());
        assertTrue(index.capture().isEmpty());
    }

    @Test
    void additionsAndRemovalsMatchReferenceOrderAndPreserveOldCaptures() {
        var index = PrefillActiveIndex.ordered(16, ORDER);
        var expected = new ArrayList<RequestRoute>();
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
                expected.sort(ORDER);
            } else {
                var request = expected.remove(random.nextInt(expected.size()));
                assertTrue(index.remove(request));
                assertFalse(index.contains(request));
                long removedVersion = index.version();
                assertFalse(index.remove(request));
                assertEquals(removedVersion, index.version());
            }
            assertTrue(index.version() > oldVersion);
            assertEquals(oldItems.stream().map(RequestRoute::requestId).toList(), ids(old));
            assertEquals(expected.stream().map(RequestRoute::requestId).toList(), ids(index.capture()));
            for (int priority = 0; priority <= 100; priority++) {
                int exactPriority = priority;
                assertEquals(expected.stream().filter(item -> item.priority() == exactPriority).count(),
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
        assertTrue(index.capture().isEmpty());
        assertEquals(expected.stream().map(RequestRoute::requestId).toList(), ids(beforeClear));
    }

    @Test
    void equalOrderingKeysStillRequireExactIdentity() {
        var index = PrefillActiveIndex.ordered(2, ORDER);
        var first = item(1, 50);
        var second = item(1, 50);
        assertTrue(index.add(first));
        assertTrue(index.add(second));
        assertFalse(index.remove(item(1, 50)));
        var ordered = new ArrayList<RequestRoute>();
        index.forEach(ordered::add);
        assertEquals(List.of(first, second), ordered);
        var captured = index.capture();
        var projected = captured.projectedItems();
        assertEquals(2, projected.size());
        assertEquals(projected.getFirst(), projected.getLast());
        assertNotSame(projected.getFirst(), projected.getLast());
        assertTrue(index.remove(first));
        assertSame(second, index.peek());
        assertSame(projected.getLast(), index.capture().projectedItems().getFirst());
        assertSame(projected, captured.projectedItems());
        assertEquals(2, captured.projectedItems().size());
    }

    @Test
    @Timeout(10)
    void concurrentCapturesShareMaterializationAndUnchangedRequestValues() throws Exception {
        var index = PrefillActiveIndex.ordered(2, ORDER);
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
                tasks.add(pool.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    return capture.projectedItems();
                }));
            }
            start.countDown();
            for (int i = 0; i < tasks.size(); i++) {
                assertSame((i % 2 == 0 ? old : current).projectedItems(),
                        tasks.get(i).get(5, TimeUnit.SECONDS));
            }
        }
        assertSame(old.projectedItems().getFirst(), current.projectedItems().getFirst());
        verify(first, times(1)).seqLen();
        assertThrows(UnsupportedOperationException.class, () -> current.projectedItems().clear());
    }

    @Test
    void failedMaterializationCanBeRetriedWithoutPublishingPartialResult() {
        var index = PrefillActiveIndex.ordered(2, ORDER);
        var first = item(1, 50);
        var second = item(2, 40);
        when(second.seqLen()).thenThrow(new IllegalStateException("test failure")).thenReturn(10L);
        index.add(first);
        index.add(second);
        var captured = index.capture();
        assertThrows(IllegalStateException.class, captured::projectedItems);
        assertEquals(2, captured.projectedItems().size());
        assertSame(captured.projectedItems(), captured.projectedItems());
        verify(first, times(1)).seqLen();
    }

    private static List<Long> ids(PrefillActiveIndex.Capture capture) {
        return capture.projectedItems().stream().map(org.flexlb.balance.planner.GroupPlanner.Item::requestId).toList();
    }

    private static RequestRoute item(long id, int priority) {
        var request = mock(RequestRoute.class);
        when(request.requestId()).thenReturn(id);
        when(request.enqueueSeq()).thenReturn(id);
        when(request.priority()).thenReturn(priority);
        when(request.seqLen()).thenReturn(10L);
        when(request.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return request;
    }
}
