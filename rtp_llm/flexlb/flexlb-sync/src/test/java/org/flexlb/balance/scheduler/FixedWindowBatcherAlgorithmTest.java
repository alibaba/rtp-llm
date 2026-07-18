package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.Comparator;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link FixedWindowBatcherAlgorithm}.
 *
 * <p>Verifies head wait time computation, queue deadline drop, and token-cap
 * filtering behavior.
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

    @Test
    void sloCaseDispatchesWhenPredictionReachesThreshold() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis()),
                        enqueuedItem(2, System.currentTimeMillis())),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> items = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
        verify(handler).onBatchReady(items.capture(), meta.capture());
        assertEquals(2, items.getValue().size());
        assertEquals("predict_threshold", meta.getValue().reason());
    }

    @Test
    void sloCaseDispatchesAtFixedWindowWhenPredictionIsBelowThreshold() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 170)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
        verify(handler).onBatchReady(anyList(), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
    }

    @Test
    void sloCaseDispatchesWhenBatchReachesMaxSize() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem[] items = new BatchItem[32];
        long now = System.currentTimeMillis();
        for (int index = 0; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1, now);
        }
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
        verify(handler).onBatchReady(dispatched.capture(), meta.capture());
        assertEquals(32, dispatched.getValue().size());
        assertEquals("batch_full", meta.getValue().reason());
    }

    @Test
    void processQueue_dropsExpiredHeadRequest() throws InterruptedException {
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedMaxInflightBatches(0); // disable backpressure
        config.setFlexlbBatchEnqueueDeadlineMs(100);     // 100ms queue deadline
        config.setFlexlbBatchFixedWaitMs(10000);         // long window — don't trigger window timeout

        long now = System.currentTimeMillis();
        // Enqueued 200ms ago — exceeds the 100ms deadline
        BatchItem head = enqueuedItem(1L, now - 200);
        PriorityBlockingQueue<BatchItem> queue = queueWith(head);

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext ctx = new BatcherContext("test", null, config, handler, queue, mock(BatchSchedulerReporter.class));

        algo.processQueue(ctx);

        // The head should have been dropped via onExpired
        Mockito.verify(handler).onExpired(head);
        assertTrue(queue.isEmpty(), "Queue should be empty after dropping expired head");
    }

    @Test
    void processQueue_tokenCapFiltersOversizedRequests() throws InterruptedException {
        FixedWindowBatcherAlgorithm algo = new FixedWindowBatcherAlgorithm();

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedMaxInflightBatches(0); // disable backpressure
        config.setFlexlbBatchEnqueueDeadlineMs(100000);  // long deadline — don't trigger
        config.setFlexlbBatchFixedWaitMs(0);             // immediate window timeout
        config.setFlexlbBatchMaxCapacity(1000);          // 1000 token cap
        config.setFlexlbBatchSizeMax(10);                // large batch size

        long now = System.currentTimeMillis();
        BatchItem item1 = itemWithSeqLen(1L, 600, now - 10);
        BatchItem item2 = itemWithSeqLen(2L, 600, now - 5);
        PriorityBlockingQueue<BatchItem> queue = queueWith(item1, item2);

        PrefillEndpoint prefillEp = mock(PrefillEndpoint.class);
        Mockito.when(prefillEp.getIp()).thenReturn("test");
        Mockito.when(prefillEp.ipPort()).thenReturn("test:8080");

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext ctx = new BatcherContext("test", prefillEp, config, handler, queue, mock(BatchSchedulerReporter.class));

        algo.processQueue(ctx);

        // Only item1 should be dispatched (600 <= 1000).
        // item2 would exceed the cap (600 + 600 = 1200 > 1000) and is skipped.
        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<BatchItem>> captor = ArgumentCaptor.forClass(List.class);
        Mockito.verify(handler).onBatchReady(captor.capture(), Mockito.any());
        assertEquals(1, captor.getValue().size(), "Only one item should fit within token cap");
        assertEquals(1L, captor.getValue().get(0).requestId());
    }

    // ---- helpers ----

    private static FlexlbConfig sloCaseConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchPredictThresholdMs(500);
        config.setFlexlbBatchFixedWaitMs(160);
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(10_000);
        return config;
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        BatchItem item = new BatchItem(null, null, null, null, null, null, null, 0, enqueuedAtMs);
        item.setSortKey(enqueuedAtMs);  // FixedWindow: sortKey = enqueuedAtMs
        return item;
    }

    private static BatchItem itemWithSeqLen(long requestId, long seqLen, long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        BatchItem item = new BatchItem(ctx, null, null, null, null, null, null, 0, enqueuedAtMs);
        item.setSortKey(enqueuedAtMs);
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
