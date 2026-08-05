package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Pure decision tests for {@link FixedWindowBatcherAlgorithm#decide}.
 *
 * <p>The algorithm has no side effects, so every test asserts the returned
 * {@link BatchDecision} (and that the queue is left untouched) instead of
 * verifying endpoint or reporter interactions.
 */
class FixedWindowBatcherAlgorithmTest {

    @Test
    void contextQueueDepthTracksMutationsWithoutQueueSizeReads() {
        BatchItem first = enqueuedItem(1L, 1L);
        BatchItem second = enqueuedItem(2L, 2L);
        PriorityBlockingQueue<BatchItem> queue = queueWith(first, second);
        BatcherContext ctx = context(
                "test", null, new FlexlbConfig(), queue,
                mock(BatchSchedulerReporter.class));

        assertEquals(2, ctx.size());
        assertTrue(ctx.remove(first));
        assertEquals(1, ctx.size());
        assertTrue(ctx.remove(second));
        assertEquals(0, ctx.size());
        assertTrue(ctx.isEmpty());

        queue.add(first);
        BatcherContext drainCtx = context(
                "test", null, new FlexlbConfig(), queue,
                mock(BatchSchedulerReporter.class));
        drainCtx.drainTo(new ArrayList<>());
        assertEquals(0, drainCtx.size());
    }

    @Test
    void emptyQueueYieldsNullDecision() {
        BatcherContext context = context(
                "test", null, sloCaseConfig(), queueWith(),
                mock(BatchSchedulerReporter.class));

        assertNull(new FixedWindowBatcherAlgorithm().decide(context));
    }

    @Test
    void sloCaseDispatchesWhenPredictionReachesThreshold() {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, System.currentTimeMillis()),
                        enqueuedItem(2, System.currentTimeMillis())),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(2, dispatch.items().size());
        assertEquals("predict_threshold", dispatch.reason());
        assertEquals(2, dispatch.queueSizeBefore());
        // Pure decision: the queue is untouched
        assertEquals(2, context.size());
    }

    @Test
    void sloCaseDispatchesAtFixedWindowWhenPredictionIsBelowThreshold() {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 170)),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals("fixed_window_timeout", dispatch.reason());
        assertEquals(1, dispatch.items().size());
        assertTrue(dispatch.headWaitMs() >= 170);
    }

    @Test
    void sloCaseDispatchesWhenBatchReachesMaxSize() {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatchItem[] items = new BatchItem[32];
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1, now);
        }
        BatcherContext context = context(
                "test", endpoint, config, queueWith(items),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(32, dispatch.items().size());
        assertEquals("batch_full", dispatch.reason());
        assertEquals(32, dispatch.queueSizeBefore());
    }

    @Test
    void backpressureYieldsNullParkDecision() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.prefillInflightCount()).thenReturn(1);

        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 1_000)),
                mock(BatchSchedulerReporter.class));

        assertNull(new FixedWindowBatcherAlgorithm().decide(context));
        assertEquals(1, context.size());
    }

    @Test
    void deadlineExceededHeadYieldsDropDecision() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchEnqueueDeadlineMs(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);

        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 1_000, 10);
        BatcherContext context = context(
                "test", endpoint, config, queueWith(head),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Drop drop = assertInstanceOf(BatchDecision.Drop.class, decision);
        assertEquals(BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED, drop.cause());
        assertEquals(head, drop.item());
        assertTrue(drop.detail().contains("deadline_ms=100"));
        // Pure decision: the item is neither removed nor settled
        assertEquals(1, context.size());
        assertFalse(head.future().isDone());
    }

    @Test
    void fixedWindowBatchUsesEnginePaddedTokenCost() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(200);
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        long now = System.currentTimeMillis() - 1_000;
        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 50),
                        enqueuedItem(3, now + 2, 30)),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
        assertEquals(60L, dispatch.items().stream().mapToLong(BatchItem::seqLen).sum());
        // Pure decision: all three items remain queued
        assertEquals(3, context.size());
    }

    @Test
    void largeMrcrRequestIsDispatchedAloneWhenPaddedBatchWouldOverflow() {
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(13);
        config.setFlexlbBatchMaxCapacity(engineBatchTokenLimit);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(engineBatchTokenLimit);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        BatchItem[] items = new BatchItem[13];
        long now = System.currentTimeMillis() - 1_000;
        items[0] = enqueuedItem(1L, now, 929_760L);
        for (int index = 1; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1L, now + index, 9_192L);
        }

        BatcherContext context = context(
                "test", endpoint, config, queueWith(items),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
        assertEquals(13, context.size());
    }

    @Test
    void dynamicKvBudgetLimitsOnlyAdditionalBatchMembers() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(100);
        status.getAvailableKvCacheTokens().set(70);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        long now = System.currentTimeMillis() - 1_000;
        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 20),
                        enqueuedItem(3, now + 2, 5)),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void everyDispatchedMrcrBatchSatisfiesEngineStrictTokenAdmission() {
        final int requestCount = 32;
        final long seqLen = 32_769L;
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchSizeMax(requestCount);
        config.setFlexlbBatchMaxCapacity(engineBatchTokenLimit);
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(131_072L);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        BatchItem[] items = new BatchItem[requestCount];
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < requestCount; index++) {
            items[index] = enqueuedItem(index + 1L, now + index, seqLen);
        }
        BatcherContext context = context(
                "test", endpoint, config, queueWith(items),
                mock(BatchSchedulerReporter.class));

        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();

        // Decision cycle 1 — batcher would remove the picked items, so
        // simulate the execution step between the two pure decisions.
        BatchDecision.Dispatch first = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide(context));
        first.items().forEach(context::remove);

        BatchDecision.Dispatch second = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide(context));
        second.items().forEach(context::remove);

        List<List<BatchItem>> batches = List.of(first.items(), second.items());
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
    void maxSeqLenIsUsedWhenWorkerDoesNotReportBatchTokenLimit() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config,
                queueWith(enqueuedItem(1, now, 60), enqueuedItem(2, now + 1, 40)),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void requestAtEngineTokenLimitIsRejectedBeforeDispatch() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        BatchItem item = enqueuedItem(1, 1, 100);
        BatcherContext context = context(
                "test", endpoint, config, queueWith(item),
                mock(BatchSchedulerReporter.class));

        BatchDecision decision = new FixedWindowBatcherAlgorithm().decide(context);

        BatchDecision.Drop drop = assertInstanceOf(BatchDecision.Drop.class, decision);
        assertEquals(BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY, drop.cause());
        assertEquals(item, drop.item());
        assertTrue(drop.detail().contains("seq_len=100"));
        assertTrue(drop.detail().contains("capacity=100"));
        // Pure decision: settlement happens in the batcher, not the algorithm
        assertFalse(item.future().isDone());
        assertEquals(1, context.size());
    }

    // ---- helpers ----

    private static FlexlbConfig sloCaseConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchPredictThresholdMs(500);
        config.setFlexlbBatchFixedWaitMs(160);
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(10_000);
        return config;
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        BatchItem item = new BatchItem(null, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
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
                balanceContext, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
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

    private static BatcherContext context(String key, PrefillEndpoint endpoint,
                                          FlexlbConfig config,
                                          PriorityBlockingQueue<BatchItem> queue,
                                          BatchSchedulerReporter reporter) {
        return new BatcherContext(key, endpoint, config,
                queue, new AtomicInteger(queue.size()), reporter);
    }
}
