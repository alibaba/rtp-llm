package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillActiveIndexTest {
    private static final Comparator<ScheduledRequest> ORDER =
            Comparator.comparingInt(ScheduledRequest::priority).reversed()
                    .thenComparingLong(ScheduledRequest::enqueueSeq);

    @Test
    void additionsAndRemovalsMatchReferenceOrderAndPreserveOldCaptures() {
        var index = PrefillActiveIndex.ordered(16, ORDER);
        var expected = new ArrayList<ScheduledRequest>();
        var random = new Random(36);
        for (int step = 0; step < 400; step++) {
            var old = index.capture();
            var oldItems = List.copyOf(expected);
            if (expected.isEmpty() || random.nextBoolean()) {
                var request = item(step, random.nextInt(4));
                assertTrue(index.add(request));
                assertFalse(index.add(request));
                expected.add(request);
                expected.sort(ORDER);
            } else {
                var request = expected.remove(random.nextInt(expected.size()));
                assertTrue(index.remove(request));
                assertFalse(index.contains(request));
                assertFalse(index.remove(request));
            }
            assertEquals(oldItems, old.items());
            assertEquals(expected, index.capture().items());
            for (int priority = 0; priority <= 100; priority++) {
                int exactPriority = priority;
                assertEquals(expected.stream().filter(item -> item.priority() == exactPriority).count(),
                        index.size(priority));
            }
            assertSame(index.capture(), index.capture());
            assertEquals(expected.isEmpty() ? null : expected.getFirst(), index.peek());
        }
        var beforeClear = index.capture();
        index.clear();
        for (int priority = 0; priority <= 100; priority++) {
            assertEquals(0, index.size(priority));
        }
        assertTrue(index.capture().items().isEmpty());
        assertEquals(expected, beforeClear.items());
    }

    @Test
    void equalOrderingKeysStillRequireExactIdentity() {
        var index = PrefillActiveIndex.ordered(2, ORDER);
        var first = item(1, 50);
        var second = item(1, 50);
        assertTrue(index.add(first));
        assertTrue(index.add(second));
        assertFalse(index.remove(item(1, 50)));
        assertEquals(List.of(first, second), index.capture().items());
        assertTrue(index.remove(first));
        assertSame(second, index.peek());
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
        assertThrows(UnsupportedOperationException.class, () -> current.items().clear());
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

    private static ScheduledRequest item(long id, int priority) {
        var request = mock(ScheduledRequest.class);
        when(request.requestId()).thenReturn(id);
        when(request.enqueueSeq()).thenReturn(id);
        when(request.priority()).thenReturn(priority);
        when(request.seqLen()).thenReturn(10L);
        when(request.expiresAtMs()).thenReturn(Long.MAX_VALUE);
        return request;
    }
}
