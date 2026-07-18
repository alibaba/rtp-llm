package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
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
    void contextQueueDepthTracksMutationsWithoutQueueSizeReads() {
        BatchItem first = enqueuedItem(1L, 1L);
        BatchItem second = enqueuedItem(2L, 2L);
        PriorityBlockingQueue<BatchItem> queue = queueWith(first, second);
        BatcherContext ctx = context(
                "test", null, new FlexlbConfig(), null, queue,
                mock(BatchSchedulerReporter.class));

        assertEquals(2, ctx.size());
        assertTrue(ctx.remove(first));
        assertEquals(1, ctx.size());
        assertTrue(ctx.remove(second));
        assertEquals(0, ctx.size());
        assertTrue(ctx.isEmpty());

        queue.add(first);
        BatcherContext drainCtx = context(
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
        BatcherContext context = context(
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
        BatcherContext context = context(
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
        BatcherContext context = context(
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
    void fixedWindowBatchDoesNotExceedEngineBatchTokenLimit() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(200);
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now - 3, 60),
                        enqueuedItem(2, now - 2, 50),
                        enqueuedItem(3, now - 1, 30)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onBatchReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L, 3L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(90L, dispatched.getValue().stream().mapToLong(BatchItem::seqLen).sum());
        assertEquals(1, context.size());
        assertEquals(2L, context.peek().requestId());
    }

    @Test
    void everyDispatchedMrcrBatchSatisfiesEngineStrictTokenAdmission() throws InterruptedException {
        final int requestCount = 32;
        final long seqLen = 32_769L;
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchSizeMax(requestCount);
        config.setFlexlbBatchMaxCapacity(engineBatchTokenLimit);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(131_072L);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        BatchItem[] items = new BatchItem[requestCount];
        long now = System.currentTimeMillis();
        for (int index = 0; index < requestCount; index++) {
            items[index] = enqueuedItem(index + 1L, now - 400 + index, seqLen);
        }
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();
        algorithm.processQueue(context);
        algorithm.processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler, times(2)).onBatchReady(
                dispatched.capture(), org.mockito.ArgumentMatchers.any());
        List<List<BatchItem>> batches = dispatched.getAllValues();

        assertEquals(List.of(31, 1), batches.stream().map(List::size).toList());
        assertEquals(requestCount, batches.stream().mapToInt(List::size).sum());
        for (List<BatchItem> batch : batches) {
            long totalTokens = batch.stream().mapToLong(BatchItem::seqLen).sum();
            assertTrue(totalTokens < engineBatchTokenLimit,
                    "Engine would reject batch with total_tokens=" + totalTokens);
        }
        assertEquals(0, context.size());
    }

    @Test
    void maxSeqLenIsUsedWhenWorkerDoesNotReportBatchTokenLimit() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now - 2, 60), enqueuedItem(2, now - 1, 40)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onBatchReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(1, context.size());
    }

    @Test
    void requestAtEngineTokenLimitIsRejectedBeforeDispatch() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        BatchItem item = enqueuedItem(1, System.currentTimeMillis(), 100);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(item),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler).onOfferFailure(eq(item), any(IllegalArgumentException.class));
        verify(handler, never()).onBatchReady(anyList(), any(DispatchMeta.class));
        assertEquals(0, context.size());
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
    void queueWaitMs_emptyQueue_returnsFixedWaitMs() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(8);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWith(), mock(BatchSchedulerReporter.class));

        assertEquals(300L, new FixedWindowBatcherAlgorithm().queueWaitMs(ctx));
    }

    @Test
    void queueWaitMs_batchMaxOne_returnsZero() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(1);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWithN(5, System.currentTimeMillis()), mock(BatchSchedulerReporter.class));

        assertEquals(0L, new FixedWindowBatcherAlgorithm().queueWaitMs(ctx));
    }

    @Test
    void queueWaitMs_fillingLastBatch_returnsZero() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(8);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWithN(15, System.currentTimeMillis()), mock(BatchSchedulerReporter.class));

        assertEquals(0L, new FixedWindowBatcherAlgorithm().queueWaitMs(ctx));
    }

    @Test
    void queueWaitMs_partialBatchAfterDispatch_returnsFixedWaitMs() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(8);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWithN(20, System.currentTimeMillis()), mock(BatchSchedulerReporter.class));

        assertEquals(300L, new FixedWindowBatcherAlgorithm().queueWaitMs(ctx));
    }

    @Test
    void queueWaitMs_expiredWindow_returnsZero() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(8);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWithN(3, System.currentTimeMillis() - 400), mock(BatchSchedulerReporter.class));

        assertEquals(0L, new FixedWindowBatcherAlgorithm().queueWaitMs(ctx));
    }

    @Test
    void queueWaitMs_returnsRemainingWindow() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchFixedWaitMs(300L);
        config.setFlexlbBatchSizeMax(8);

        BatcherContext ctx = new BatcherContext("test", null, config, null,
                queueWithN(3, System.currentTimeMillis() - 100), mock(BatchSchedulerReporter.class));
        long waitMs = new FixedWindowBatcherAlgorithm().queueWaitMs(ctx);

        assertTrue(waitMs >= 190 && waitMs <= 200, "Expected about 200ms, got " + waitMs);
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
        BatchItem item = new BatchItem(null, null, null, null, null, null, null, enqueuedAtMs);
        item.setSortKey(enqueuedAtMs);  // FixedWindow: sortKey = enqueuedAtMs
        return item;
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        BatchItem item = new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
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

    private static PriorityBlockingQueue<BatchItem> queueWithN(int count, long enqueuedAtMs) {
        PriorityBlockingQueue<BatchItem> queue = queueWith();
        for (int i = 0; i < count; i++) {
            queue.add(enqueuedItem(i + 1L, enqueuedAtMs));
        }
        return queue;
    }

    private static BatcherContext context(String key, PrefillEndpoint endpoint,
                                          FlexlbConfig config, BatchDecisionHandler handler,
                                          PriorityBlockingQueue<BatchItem> queue,
                                          BatchSchedulerReporter reporter) {
        BatchItem head = queue.peek();
        return new BatcherContext(key, endpoint, config, handler, queue,
                new AtomicInteger(queue.size()),
                new AtomicLong(head == null ? 0 : head.sortKey()), reporter);
    }
}
