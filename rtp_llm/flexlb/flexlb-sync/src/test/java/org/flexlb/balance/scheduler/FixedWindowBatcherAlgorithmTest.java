package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

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

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis()),
                        enqueuedItem(2, System.currentTimeMillis())),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> items = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(items.capture(), meta.capture());
        assertEquals(2, items.getValue().size());
        assertEquals("predict_threshold", meta.getValue().reason());
    }

    @Test
    void sloCaseDispatchesAtFixedWindowWhenPredictionIsBelowThreshold() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 170)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
    }

    @Test
    void sloCaseDispatchesWhenBatchReachesMaxSize() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem[] items = new BatchItem[32];
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1, now);
        }
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(32, dispatched.getValue().size());
        assertEquals("batch_full", meta.getValue().reason());
    }

    @Test
    void fixedWindowBatchUsesEnginePaddedTokenCost() throws InterruptedException {
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

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis() - 1_000;
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 50),
                        enqueuedItem(3, now + 2, 30)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(60L, dispatched.getValue().stream().mapToLong(BatchItem::seqLen).sum());
        assertEquals(2, context.size());
        assertEquals(2L, context.peek().requestId());
    }

    @Test
    void largeMrcrRequestIsDispatchedAloneWhenPaddedBatchWouldOverflow() throws InterruptedException {
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
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        BatchItem[] items = new BatchItem[13];
        long now = System.currentTimeMillis() - 1_000;
        items[0] = enqueuedItem(1L, now, 929_760L);
        for (int index = 1; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1L, now + index, 9_192L);
        }

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(12, context.size());
    }

    @Test
    void dynamicKvBudgetLimitsOnlyAdditionalBatchMembers() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(100);
        status.getAvailableKvCacheTokens().set(70);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        long now = System.currentTimeMillis() - 1_000;
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 20),
                        enqueuedItem(3, now + 2, 5)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(2, context.size());
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
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(131_072L);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        BatchItem[] items = new BatchItem[requestCount];
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < requestCount; index++) {
            items[index] = enqueuedItem(index + 1L, now + index, seqLen);
        }
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();
        algorithm.processQueue(context);
        algorithm.processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler, times(2)).onDecisionGroupReady(
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

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60), enqueuedItem(2, now + 1, 40)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), org.mockito.ArgumentMatchers.any());
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

        BatchItem item = enqueuedItem(1, 1, 100);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(item),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler).onOfferFailure(eq(item), any(IllegalArgumentException.class));
        verify(handler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        assertEquals(0, context.size());
    }

    @Test
    void routeRequestCapDoesNotChangeLogicalBatchDecisionTiming()
            throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(60_000);
        config.setFlexlbBatchSizeMax(4);
        config.setAutoTpmPrefillMaxInflightRequestsPerWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        routeDecisionItem(1, now),
                        routeDecisionItem(2, now + 1),
                        routeDecisionItem(3, now + 2)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        assertEquals(3, context.size());
    }

    @Test
    void routeRequestCapLimitsDeliveryOnlyAfterLogicalGroupIsReady()
            throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(60_000);
        config.setFlexlbBatchSizeMax(4);
        config.setAutoTpmPrefillMaxInflightRequestsPerWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1, 0, 1);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        routeDecisionItem(1, now),
                        routeDecisionItem(2, now + 1),
                        routeDecisionItem(3, now + 2),
                        routeDecisionItem(4, now + 3)),
                reporter);

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> delivered = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(delivered.capture(), meta.capture());
        assertEquals(List.of(1L), delivered.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_full", meta.getValue().reason());
        assertEquals(3, context.size());
        assertTrue(context.isActiveEmpty());
        assertEquals(3, context.readyDeliveryCount());
        verifyNoInteractions(reporter);

        // The logical group is already ready. A full request cap must not
        // send it back through the fixed-window decision, and freeing one
        // slot must hand off the next member immediately in original order.
        assertEquals(BatcherContext.ReadyDeliveryResult.CAPACITY_BLOCKED,
                context.deliverReadyRequests());
        assertEquals(BatcherContext.ReadyDeliveryResult.DELIVERED,
                context.deliverReadyRequests());

        ArgumentCaptor<List<BatchItem>> allDeliveries = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> allMeta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler, times(2)).onDecisionGroupReady(allDeliveries.capture(), allMeta.capture());
        assertEquals(List.of(2L), allDeliveries.getAllValues().get(1).stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_full", allMeta.getAllValues().get(1).reason());
        assertEquals(2, context.size());
        assertEquals(2, context.readyDeliveryCount());
    }

    @Test
    void routeHeadStopsAtModeBoundaryAndCannotBypassBatchInflightGate()
            throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        config.setAutoTpmPrefillMaxInflightRequestsPerWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        when(endpoint.getInflightBatchCount()).thenReturn(1);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatchItem route = routeDecisionItem(1, now - 2);
        BatchItem batch = enqueuedItem(2, now - 1, 1);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(route, batch),
                mock(BatchSchedulerReporter.class));
        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();

        algorithm.processQueue(context);

        ArgumentCaptor<List<BatchItem>> firstDelivery = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(firstDelivery.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(1L), firstDelivery.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
        verify(endpoint, never()).getInflightBatchCount();

        // Once the BATCH_ENQUEUE item becomes head, its own inflight gate applies.
        // It must not have ridden along with the preceding route decision.
        algorithm.processQueue(context);

        verify(handler, times(1)).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        verify(endpoint).getInflightBatchCount();
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
    }

    // ---- queue_deadline_exceeded valve gating ----

    @Test
    void autoTpmOnHeadWithoutPriorityFieldPastQueueDeadlineIsNotDropped() throws InterruptedException {
        // hasPriority gate removed: the drop exemption depends only on the
        // switch, so even a head whose priority field was never set
        // (impossible in production — normalize() always assigns 1-100) is
        // dispatched instead of dropped.
        FlexlbConfig config = sloCaseConfig();
        config.setAutoTpmEnabled(true);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 11_000, 100);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(head),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
    }

    @Test
    void autoTpmOnPriorityHeadPastQueueDeadlineIsNotDropped() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        config.setAutoTpmEnabled(true);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 11_000, 100, 50);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(head),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
    }

    @Test
    void autoTpmOffLegacyHeadPastQueueDeadlineIsDroppedParity() throws InterruptedException {
        FlexlbConfig config = sloCaseConfig();
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 11_000, 100);
        BatcherContext context = context(
                "test", null, config, handler, queueWith(head),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler).onExpired(head);
        assertEquals(0, context.size());
    }

    @Test
    void autoTpmOffPriorityHeadPastQueueDeadlineIsStillDroppedParity() throws InterruptedException {
        // Pre-fix behavior: with the switch off, priority carried on the item
        // has no effect — the legacy drop applies to everyone.
        FlexlbConfig config = sloCaseConfig();
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 11_000, 100, 50);
        BatcherContext context = context(
                "test", null, config, handler, queueWith(head),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler).onExpired(head);
        assertEquals(0, context.size());
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
        return enqueuedItem(requestId, enqueuedAtMs, seqLen, 0);
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        if (priority > 0) {
            balanceContext.setBudget(ScheduleBudget.forDeadline(
                    priority, enqueuedAtMs, enqueuedAtMs + 30_000));
        }
        BatchItem item = new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
        item.setSortKey(enqueuedAtMs);
        return item;
    }

    private static BatchItem routeDecisionItem(long requestId, long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setScheduleMode(ScheduleModeEnum.QUEUE);
        BatchItem item = new BatchItem(
                context, null, null, null, null, null, null, enqueuedAtMs);
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
                                          FlexlbConfig config, DecisionGroupHandler handler,
                                          PriorityBlockingQueue<BatchItem> queue,
                                          BatchSchedulerReporter reporter) {
        return new BatcherContext(key, endpoint, config, handler, queue,
                new AtomicInteger(queue.size()), reporter);
    }
}
