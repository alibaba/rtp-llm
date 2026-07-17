package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.ArrayList;
import java.util.concurrent.PriorityBlockingQueue;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link FixedWindowBatcherAlgorithm#headWaitMs(BatcherContext)}.
 *
 * <p>Verifies that the algorithm computes head wait time using the
 * fixed-window semantics ({@code fixedWaitMs - elapsedMs}) rather
 * than leaking sortKey-as-deadline assumptions.
 */
class FixedWindowBatcherAlgorithmTest {

    @Test
    void headWaitMsReturnsRemainingWindowTime() {
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);

        long now = System.currentTimeMillis();
        BatchItem head = enqueuedItem(1L, now - 200);
        PriorityBlockingQueue<BatchItem> queue = queueWith(head);

        BatcherContext ctx = new BatcherContext("test", null, config, null, queue, mock(BatchSchedulerReporter.class));

        // Head enqueued 200ms ago, window=300ms → ~100ms remaining (±5ms tolerance for timing)
        long waitMs = algo.headWaitMs(ctx);
        assertTrue(waitMs >= 95 && waitMs <= 100,
                "Expected ~100ms remaining, got " + waitMs);
    }

    @Test
    void headWaitMsReturnsZeroWhenQueueEmpty() {
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));

        BatcherContext ctx = new BatcherContext("test", null, config, null, queue, mock(BatchSchedulerReporter.class));

        assertEquals(0, algo.headWaitMs(ctx));
    }

    @Test
    void headWaitMsReturnsZeroWhenElapsedExceedsWindow() {
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);

        long now = System.currentTimeMillis();
        // Enqueued 500ms ago → elapsed > fixedWaitMs
        BatchItem head = enqueuedItem(1L, now - 500);
        PriorityBlockingQueue<BatchItem> queue = queueWith(head);

        BatcherContext ctx = new BatcherContext("test", null, config, null, queue, mock(BatchSchedulerReporter.class));

        assertEquals(0, algo.headWaitMs(ctx));
    }

    @Test
    void headWaitMsDiffersFromSortKeyBasedDefault() {
        // The default BatcherAlgorithm.headWaitMs() treats sortKey as deadline.
        // FixedWindow must NOT follow that pattern — its sortKey is enqueuedAtMs (past).
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);

        long now = System.currentTimeMillis();
        BatchItem head = enqueuedItem(1L, now - 200);
        // Default headWaitMs would compute: sortKey - now = (now-200) - now = -200 → 0
        // FixedWindow headWaitMs computes: fixedWaitMs - elapsed = 300 - 200 = 100
        PriorityBlockingQueue<BatchItem> queue = queueWith(head);

        BatcherContext ctx = new BatcherContext("test", null, config, null, queue, mock(BatchSchedulerReporter.class));

        long fixedWaitMs = algo.headWaitMs(ctx);
        assertTrue(fixedWaitMs >= 95 && fixedWaitMs <= 100,
                "FixedWindow should return ~100ms remaining, got " + fixedWaitMs);
    }

    @Test
    void contextQueueDepthTracksMutationsWithoutQueueSizeReads() {
        BatchItem first = enqueuedItem(1L, 1L);
        BatchItem second = enqueuedItem(2L, 2L);
        PriorityBlockingQueue<BatchItem> queue = queueWith(first, second);
        BatcherContext ctx = new BatcherContext(
                "test", null, new FlexlbConfig(), null, queue,
                mock(BatchSchedulerReporter.class));

        assertEquals(2, ctx.size());
        assertEquals(first, ctx.poll());
        assertEquals(1, ctx.size());
        assertTrue(ctx.remove(second));
        assertEquals(0, ctx.size());
        assertTrue(ctx.isEmpty());

        queue.add(first);
        BatcherContext drainCtx = new BatcherContext(
                "test", null, new FlexlbConfig(), null, queue,
                mock(BatchSchedulerReporter.class));
        drainCtx.drainTo(new ArrayList<>());
        assertEquals(0, drainCtx.size());
    }

    // ---- helpers ----

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        BatchItem item = new BatchItem(null, null, null, null, null, null, null, 0, enqueuedAtMs);
        item.setSortKey(enqueuedAtMs);  // FixedWindow: sortKey = enqueuedAtMs
        return item;
    }

    private static PriorityBlockingQueue<BatchItem> queueWith(BatchItem... items) {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
        for (BatchItem item : items) {
            queue.add(item);
        }
        return queue;
    }
}
