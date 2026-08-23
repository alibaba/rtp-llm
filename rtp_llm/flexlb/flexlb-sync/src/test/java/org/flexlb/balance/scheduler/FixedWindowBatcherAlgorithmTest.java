package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class FixedWindowBatcherAlgorithmTest {

    private static DecisionGroupHandler resolvingHandler() {
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        doAnswer(invocation -> {
            TestCapacityAdmission.complete(invocation.getArgument(0));
            return null;
        }).when(handler).onDecisionGroupAdmitted(
                any(AdmittedDecisionGroup.class),
                any(DecisionGroupMetadata.class));
        return handler;
    }

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
        assertEquals(0, ctx.size());

        queue.add(first);
        BatcherContext drainCtx = context(
                "test", null, new FlexlbConfig(), null, queue,
                mock(BatchSchedulerReporter.class));
        drainCtx.drainTo(new ArrayList<>());
        assertEquals(0, drainCtx.size());
    }

    @Test
    void incompleteCollectionWindowReturnsAbsoluteEventDrivenWait()
            throws InterruptedException {
        long nowMs = 1_000_000L;
        long windowOpenedAtMs = 995_000L;
        long collectionWindowMs = 10_000L;
        long expiresAtMs = 1_020_000L;
        long expectedWakeAtMs = 1_005_000L;
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config)
                .setMaxCollectionWaitMs(collectionWindowMs);
        SchedulingTestConfig.useFixedWindowDecision(config)
                .setMaxPredictedExecutionMs(0L);

        BatchItem head = expiringItem(1L, windowOpenedAtMs, expiresAtMs);
        PriorityBlockingQueue<BatchItem> queue = queueWith(head);
        AtomicInteger queueDepth = new AtomicInteger(1);
        AtomicLong queueVersion = new AtomicLong(41L);
        ReentrantLock queueLock = new ReentrantLock();
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        DeliveryCapacityAdmission capacityAdmission =
                mock(DeliveryCapacityAdmission.class);
        BatcherContext context = new BatcherContext(
                "collection-wait-test",
                null,
                config,
                handler,
                capacityAdmission,
                queue,
                queueDepth,
                queueVersion,
                queueLock,
                WorkerBatcher.FIFO_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class)) {
            @Override
            long now() {
                return nowMs;
            }
        };
        queueLock.lock();
        try {
            context.incrementSchedulingInputVersion();
        } finally {
            queueLock.unlock();
        }

        BatcherCycleResult result =
                new FixedWindowBatcherAlgorithm().processQueue(context);

        BatcherCycleResult.AwaitingSchedulingChange waiting = assertInstanceOf(
                BatcherCycleResult.AwaitingSchedulingChange.class, result);
        assertSame(head, waiting.head());
        assertEquals(41L, waiting.queueVersion());
        assertEquals(1L, waiting.schedulingInputVersion());
        assertEquals(expectedWakeAtMs, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.COLLECTION_WINDOW,
                waiting.reason());
        assertEquals(List.of(head), context.activeItemsInSchedulingOrder());
        verifyNoInteractions(handler, capacityAdmission);
    }

    @Test
    void zeroCollectionWindowStillFormsAGroupFromCurrentlyAvailableRequests()
            throws InterruptedException {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 1),
                        enqueuedItem(2, now + 1, 1),
                        enqueuedItem(3, now + 2, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> metadata =
                ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), metadata.capture());
        assertEquals(List.of(1L, 2L, 3L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("fixed_window_timeout", metadata.getValue().reason());
        assertEquals(0, context.size());
    }

    @Test
    void capsBatchGrowthAtPredictedExecutionBudget() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        List<Integer> predictedSizes = new ArrayList<>();
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation -> {
            int size = ((List<?>) invocation.getArgument(0)).size();
            predictedSizes.add(size);
            return size == 1 ? 499.0 : 501.0;
        });

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis()),
                        enqueuedItem(2, System.currentTimeMillis())),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> items = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(items.capture(), meta.capture());
        assertEquals(List.of(1L), items.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        assertEquals(1, context.size(), "the over-budget member stays queued");
        assertEquals(List.of(1, 2), predictedSizes);
    }

    @Test
    void dispatchesAtFixedWindowWhenPredictionIsBelowThreshold() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(499.0);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 170)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        verify(predictor, times(1)).predictBatchMs(anyList());
    }

    @Test
    void dispatchesWhenBatchReachesMaxSize() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(499.0);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        DecisionGroupHandler handler = resolvingHandler();
        BatchItem[] items = new BatchItem[32];
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < items.length; index++) {
            items[index] = enqueuedItem(index + 1, now);
        }
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(32, dispatched.getValue().requests().size());
        assertEquals("batch_full", meta.getValue().reason());
        verify(predictor, times(32)).predictBatchMs(anyList());
    }

    @Test
    void explicitPredictionLimitKeepsEqualMemberAndDispatchesBeforeWindow()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(3);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation ->
                ((List<?>) invocation.getArgument(0)).size() == 1 ? 100.0 : 500.0);

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1),
                        enqueuedItem(2, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L, 2L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        assertEquals(0, context.size());
    }

    @Test
    void explicitSingletonAtPredictionLimitDispatchesBeforeWindow()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta =
                ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        assertEquals(0, context.size());
    }

    @Test
    void fullBatchDispatchesLargestFeasibleGroup() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(499.0);

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 1),
                        enqueuedItem(2, now + 1, 1),
                        enqueuedItem(3, now + 2, 1),
                        enqueuedItem(4, now + 3, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L, 2L, 3L, 4L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_full", meta.getValue().reason());
        assertEquals(0, context.size());
        verify(predictor, times(4)).predictBatchMs(anyList());
    }

    @Test
    void timeoutDispatchesLargestFeasibleGroup() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(10);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation ->
                ((List<?>) invocation.getArgument(0)).size() * 200.0);

        long old = System.currentTimeMillis() - 1_000;
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, old, 1), enqueuedItem(2, old + 1, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(2, dispatched.getValue().requests().size());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        verify(predictor, times(2)).predictBatchMs(anyList());
    }

    @Test
    void nonMonotonicPredictionCapsAtTheFirstOverBudgetMember() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(5);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        // A cheaper four-member prediction must not tempt the group past the
        // three-member spike that already exceeds the budget.
        List<Integer> predictedSizes = new ArrayList<>();
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation -> {
            int size = ((List<?>) invocation.getArgument(0)).size();
            predictedSizes.add(size);
            return size == 3 ? 600.0 : 100.0;
        });

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 1),
                        enqueuedItem(2, now + 1, 1),
                        enqueuedItem(3, now + 2, 1),
                        enqueuedItem(4, now + 3, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler).onDecisionGroupAdmitted(
                dispatched.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(1L, 2L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals(List.of(1, 2, 3), predictedSizes,
                "growth stops at the spike instead of climbing back to four");
    }

    @Test
    void computeLimitedModePrefixWaitsForWindowBeforeDispatchingFeasiblePrefix()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(20);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(100.0);

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 20),
                        enqueuedItem(2, now + 1, 20),
                        enqueuedItem(3, now + 2, 60),
                        enqueuedItem(4, now + 3, 5)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        Thread.sleep(30L);
        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L, 2L), dispatched.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        assertEquals(2, context.size());
        verify(predictor, times(4)).predictBatchMs(anyList());
    }

    @Test
    void batchCapacityAdmissionBlocksAfterOnePrediction()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(1_000.0);
        AtomicInteger batchAdmissionAttempts = new AtomicInteger();
        DeliveryCapacityAdmission capacityAdmission = batchCapacityAdmission(
                () -> false, batchAdmissionAttempts);

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, capacityAdmission,
                queueWith(enqueuedItem(config, 1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any(DecisionGroupMetadata.class));
        verify(predictor, times(1)).predictBatchMs(anyList());
        assertEquals(1, batchAdmissionAttempts.get());
        verify(endpoint, never()).getInflightBatchCount();
        assertEquals(1, context.size());
    }

    @Test
    void zeroConfiguredBatchLimitStillUsesAuthoritativeReservation()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(0);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(501.0);
        AtomicInteger batchAdmissionAttempts = new AtomicInteger();
        DeliveryCapacityAdmission capacityAdmission = batchCapacityAdmission(
                () -> true, batchAdmissionAttempts);

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, capacityAdmission,
                queueWith(enqueuedItem(config, 1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), meta.capture());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        assertEquals(1, batchAdmissionAttempts.get());
        verify(endpoint, never()).getInflightBatchCount();
    }

    @Test
    void nanPredictionFallsBackToFixedWindowTimeout() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(Double.NaN);

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        verify(predictor, times(1)).predictBatchMs(anyList());
    }

    @Test
    void fixedWindowBatchUsesEnginePaddedTokenCost() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(200);
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        DecisionGroupHandler handler = resolvingHandler();
        long now = System.currentTimeMillis() - 1_000;
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 50),
                        enqueuedItem(3, now + 2, 30)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L), dispatched.getValue().requests().stream().map(BatchItem::requestId).toList());
        assertEquals(60L, dispatched.getValue().requests().stream().mapToLong(BatchItem::seqLen).sum());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        assertEquals(2, context.size());
        assertEquals(2L, context.peek().requestId());
    }

    @Test
    void largeMrcrRequestIsDispatchedAloneWhenPaddedBatchWouldOverflow() throws InterruptedException {
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(13);

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

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().requests().stream().map(BatchItem::requestId).toList());
        assertEquals(12, context.size());
    }

    @Test
    void kvLimitedModePrefixWaitsForWindowBeforeDispatchingFeasiblePrefix()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(3);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(20);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(100);
        status.getAvailableKvCacheTokens().set(70);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60),
                        enqueuedItem(2, now + 1, 20),
                        enqueuedItem(3, now + 2, 5)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        Thread.sleep(30L);
        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L), dispatched.getValue().requests().stream().map(BatchItem::requestId).toList());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        assertEquals(2, context.size());
        assertEquals(2L, context.peek().requestId());
    }

    @Test
    void everyDispatchedMrcrBatchSatisfiesEngineStrictTokenAdmission() throws InterruptedException {
        final int requestCount = 32;
        final long seqLen = 32_769L;
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(requestCount);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

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
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(items),
                mock(BatchSchedulerReporter.class));

        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();
        algorithm.processQueue(context);
        algorithm.processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler, times(2)).onDecisionGroupAdmitted(
                dispatched.capture(), org.mockito.ArgumentMatchers.any());
        List<List<BatchItem>> batches = dispatched.getAllValues().stream()
                .map(AdmittedDecisionGroup::requests)
                .toList();

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
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        DecisionGroupHandler handler = resolvingHandler();
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, now, 60), enqueuedItem(2, now + 1, 40)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> dispatched = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler).onDecisionGroupAdmitted(dispatched.capture(), org.mockito.ArgumentMatchers.any());
        assertEquals(List.of(1L), dispatched.getValue().requests().stream().map(BatchItem::requestId).toList());
        assertEquals(1, context.size());
    }

    @Test
    void requestAtEngineTokenLimitIsRejectedBeforeDispatch() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        BatchItem item = enqueuedItem(1, 1, 100);
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(item),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler).onOfferFailure(eq(item), any(IllegalArgumentException.class));
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any(DecisionGroupMetadata.class));
        assertEquals(0, context.size());
    }

    @Test
    void offerDuringPredictionBelongsToTheNextStableSnapshot()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(4);

        long now = System.currentTimeMillis();
        BatchItem low = enqueuedItem(1, now, 1, 10);
        BatchItem high = enqueuedItem(2, now + 1, 1, 100);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        queue.add(low);
        AtomicInteger queueDepth = new AtomicInteger(1);
        AtomicLong queueVersion = new AtomicLong();
        ReentrantLock queueLock = new ReentrantLock();

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        AtomicBoolean offered = new AtomicBoolean();
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            if (offered.compareAndSet(false, true)) {
                // Prediction runs after the ordered snapshot is captured. The
                // new higher-priority offer belongs to the next selection cut
                // and does not revoke the already captured request.
                queueLock.lock();
                try {
                    queue.add(high);
                    queueDepth.incrementAndGet();
                    queueVersion.incrementAndGet();
                } finally {
                    queueLock.unlock();
                }
            }
            return 600.0;
        });

        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler,
                TestCapacityAdmission.alwaysAvailable(), queue, queueDepth,
                queueVersion, queueLock, WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> delivered = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler).onDecisionGroupAdmitted(
                delivered.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(low), delivered.getValue().requests());
        assertEquals(List.of(high), context.activeItemsInSchedulingOrder());
        assertEquals(1, context.size());
        verify(predictor, times(1)).predictBatchMs(anyList());
    }

    @Test
    void routeHeadStopsAtModeBoundaryAndBatchHeadUsesGroupCapacityAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(100.0);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        AtomicInteger batchAdmissionAttempts = new AtomicInteger();
        DeliveryCapacityAdmission capacityAdmission = batchCapacityAdmission(
                () -> false, batchAdmissionAttempts);
        DecisionGroupHandler handler = resolvingHandler();
        long now = System.currentTimeMillis();
        BatchItem route = routeDecisionItem(1, now - 2);
        BatchItem batch = enqueuedItem(config, 2, now - 1, 1);
        BatcherContext context = context(
                "test", endpoint, config, handler, capacityAdmission,
                queueWith(route, batch),
                mock(BatchSchedulerReporter.class));
        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();

        algorithm.processQueue(context);

        ArgumentCaptor<AdmittedDecisionGroup> firstDelivery = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler).onDecisionGroupAdmitted(firstDelivery.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(1L), firstDelivery.getValue().requests().stream()
                .map(BatchItem::requestId).toList());
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
        assertEquals(0, batchAdmissionAttempts.get(),
                "route delivery owns no QUEUE batch slot");
        verify(endpoint, never()).getInflightBatchCount();
        verify(predictor, times(1)).predictBatchMs(anyList());

        // Once the BATCH_ENQUEUE item becomes head, it must reserve the one
        // group-scoped QUEUE batch slot before callback ownership can move.
        // It must not have ridden along with the preceding route decision.
        algorithm.processQueue(context);

        verify(handler, times(1)).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any(DecisionGroupMetadata.class));
        assertEquals(1, batchAdmissionAttempts.get());
        verify(endpoint, never()).getInflightBatchCount();
        verify(predictor, times(2)).predictBatchMs(anyList());
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
    }

    @Test
    void queueMutationBetweenPredictionAndAdmissionInvalidatesWholeGroup()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        long now = System.currentTimeMillis();
        BatchItem first = enqueuedItem(1, now, 1);
        BatchItem second = enqueuedItem(2, now + 1, 1);
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(first, second),
                mock(BatchSchedulerReporter.class));
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation -> {
            if (((List<?>) invocation.getArgument(0)).size() == 2) {
                assertTrue(context.remove(second));
            }
            return 100.0;
        });

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(2)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(List.of(first), context.activeItemsInSchedulingOrder());
    }

    @Test
    void continuousLowAndHighPriorityOffersDoNotStarveCapturedGroups()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);

        long now = System.currentTimeMillis() - 2_000L;
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                256, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        for (int index = 0; index < 200; index++) {
            queue.add(enqueuedItem(index + 1L, now + index, 1, 50));
        }
        AtomicInteger queueDepth = new AtomicInteger(queue.size());
        AtomicLong queueVersion = new AtomicLong();
        ReentrantLock queueLock = new ReentrantLock();

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler,
                TestCapacityAdmission.alwaysAvailable(), queue, queueDepth,
                queueVersion, queueLock, WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));

        AtomicInteger predictionCalls = new AtomicInteger();
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation -> {
            if (((List<?>) invocation.getArgument(0)).size() == 1) {
                int call = predictionCalls.incrementAndGet();
                // Alternate an offer below and above the priority of the captured
                // group. Both offers linearize after that decision snapshot and
                // must wait for a following decision rather than invalidating it.
                int laterPriority = (call & 1) == 0 ? 1 : 100;
                BatchItem later = enqueuedItem(
                        10_000L + call, now + 1_000L + call, 1, laterPriority);
                queueLock.lock();
                try {
                    queue.add(later);
                    queueDepth.incrementAndGet();
                    queueVersion.incrementAndGet();
                } finally {
                    queueLock.unlock();
                }
            }
            return 100.0;
        });

        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();
        for (int tick = 0; tick < 50; tick++) {
            algorithm.processQueue(context);
        }

        ArgumentCaptor<AdmittedDecisionGroup> delivered = ArgumentCaptor.forClass(AdmittedDecisionGroup.class);
        verify(handler, times(50)).onDecisionGroupAdmitted(
                delivered.capture(), any(DecisionGroupMetadata.class));
        assertTrue(delivered.getAllValues().stream()
                .allMatch(group -> group.requests().size() == 4));
        assertEquals(200, delivered.getAllValues().stream()
                .mapToInt(group -> group.requests().size()).sum());
        assertEquals(50, context.size());
        assertEquals(50, predictionCalls.get());
        verify(predictor, times(200)).predictBatchMs(anyList());
    }

    @Test
    void headRemovalAfterStableSnapshotCannotSplitDecisionGroup()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        long now = System.currentTimeMillis();
        BatchItem first = enqueuedItem(1, now, 1);
        BatchItem second = enqueuedItem(2, now + 1, 1);
        AtomicReference<BatcherContext> contextRef = new AtomicReference<>();
        AtomicBoolean removed = new AtomicBoolean();
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            if (removed.compareAndSet(false, true)) {
                // Removing a captured member during prediction revokes the
                // entire candidate group at the ownership check.
                assertTrue(contextRef.get().remove(first));
            }
            return 600.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(first, second),
                mock(BatchSchedulerReporter.class));
        contextRef.set(context);

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(List.of(second), context.activeItemsInSchedulingOrder());
    }

    @Test
    void learningRevisionChangeDuringPredictionPreventsStaleAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        AtomicLong generation = new AtomicLong();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.generation()).thenAnswer(ignored -> generation.get());
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            generation.incrementAndGet();
            return 600.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(config, 1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, context.size());
    }

    @Test
    void requestThatExpiresDuringSlowPredictionCannotBeAdmitted()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        long now = System.currentTimeMillis();
        long expiresAtMs = now + 200L;
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            while (System.currentTimeMillis() <= expiresAtMs) {
                Thread.sleep(1L);
            }
            return 600.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(expiringItem(1, now, expiresAtMs)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, context.size(),
                "the expired member stays owned until the next expiration pass");
    }

    @Test
    void windowElapsedDuringSlowPredictionDispatchesInTheSamePass()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(20);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            Thread.sleep(75L);
            return 100.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> metadata =
                ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), metadata.capture());
        assertEquals("fixed_window_timeout", metadata.getValue().reason());
        verify(predictor, times(1)).predictBatchMs(anyList());
        assertEquals(0, context.size());
    }

    @Test
    void learningRevisionChangeDuringCapacityReservationPreventsStaleAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        AtomicLong generation = new AtomicLong();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.generation()).thenAnswer(ignored -> generation.get());
        when(predictor.predictBatchMs(anyList())).thenReturn(600.0);
        DecisionGroupHandler handler = resolvingHandler();
        DeliveryCapacityAdmission available = TestCapacityAdmission.alwaysAvailable();
        AtomicInteger batchReservationReleases = new AtomicInteger();
        DeliveryCapacityAdmission capacityAdmission = new DeliveryCapacityAdmission() {
            @Override
            public AdmissionResult tryReserveItemCapacity(BatchItem item) {
                generation.incrementAndGet();
                return available.tryReserveItemCapacity(item);
            }

            @Override
            public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
                return new BatchCapacityReserved(new BatchCapacityReservation() {
                    private boolean released;

                    @Override
                    public BatchItem head() {
                        return head;
                    }

                    @Override
                    public BatchLoadPublicationResult establishBatchLoadPublication(
                            List<BatchItem> requests) {
                        return new BatchLoadPublicationEstablished(() -> { });
                    }

                    @Override
                    public BatchDispatcher.SubmissionPermit transferToBatchLifecycle(
                            long batchId,
                            long predictedMs,
                            List<BatchItem> requests) {
                        throw new AssertionError(
                                "stale selection must not register batch lifecycle");
                    }

                    @Override
                    public void completeDeliveryHandoff() {
                        throw new AssertionError(
                                "stale selection must not complete a delivery handoff");
                    }

                    @Override
                    public synchronized void release() {
                        if (!released) {
                            released = true;
                            batchReservationReleases.incrementAndGet();
                        }
                    }
                });
            }
        };
        BatcherContext context = context(
                "test", endpoint, config, handler, capacityAdmission,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, batchReservationReleases.get(),
                "selection invalidation must release its provisional group slot");
        assertEquals(1, context.size());
    }

    @Test
    void learningRevisionChangeBeforeOwnershipTransferPreventsStaleAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        AtomicLong generation = new AtomicLong();
        AtomicInteger predictorReads = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenAnswer(ignored -> {
            if (predictorReads.incrementAndGet() == 3) {
                generation.incrementAndGet();
            }
            return predictor;
        });
        when(predictor.generation()).thenAnswer(ignored -> generation.get());
        when(predictor.predictBatchMs(anyList())).thenReturn(600.0);
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(config, 1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, context.size());
    }

    @Test
    void capacityBecomingUnavailableDuringPredictionBlocksAtSingleAdmissionGate()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        AtomicBoolean capacityAvailable = new AtomicBoolean(true);
        AtomicInteger batchAdmissionAttempts = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            capacityAvailable.set(false);
            return 600.0;
        });
        DeliveryCapacityAdmission capacityAdmission = batchCapacityAdmission(
                capacityAvailable::get, batchAdmissionAttempts);
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler, capacityAdmission,
                queueWith(enqueuedItem(config, 1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        assertEquals(1, batchAdmissionAttempts.get());
        verify(endpoint, never()).getInflightBatchCount();
        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, context.size());
    }

    @Test
    void engineComputeCapacityDropDuringPredictionPreventsAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            // P=[100,100] has padded compute shape 200; equality is rejected
            // by the Engine's strict max_batch_tokens_size gate.
            status.setMaxBatchTokensSize(200);
            return 100.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 100),
                        enqueuedItem(2, now + 1, 100)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(2)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(2, context.size());
    }

    @Test
    void engineKvCapacityDropDuringPredictionPreventsAdmission()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(1_000);
        status.getAvailableKvCacheTokens().set(1_000);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            status.getAvailableKvCacheTokens().set(199);
            return 100.0;
        });
        DecisionGroupHandler handler = resolvingHandler();
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 100),
                        enqueuedItem(2, now + 1, 100)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(2)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(2, context.size());
    }

    @Test
    void singletonWaitsWithoutPredictionWhenKvIsInsufficient()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(1_000);
        status.getAvailableKvCacheTokens().set(99);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        DecisionGroupHandler handler = resolvingHandler();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 100)),
                mock(BatchSchedulerReporter.class));
        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();

        algorithm.processQueue(context);
        algorithm.processQueue(context);

        verifyNoInteractions(predictor);
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class), any());
        assertEquals(1, context.size());
    }

    // ---- queue_deadline_exceeded valve gating ----

    // ---- helpers ----

    private static FlexlbConfig batchConfig() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useBatchDispatcher(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(160);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(32);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(0);
        SchedulingTestConfig.useBatchDispatcher(config).setEnqueueRpcTimeoutMs(10_000);
        return config;
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        return enqueuedItem(requestId, enqueuedAtMs, 0);
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen) {
        return enqueuedItem(requestId, enqueuedAtMs, seqLen, 0);
    }

    private static BatchItem enqueuedItem(
            FlexlbConfig requestConfig,
            long requestId,
            long enqueuedAtMs,
            long seqLen) {
        return enqueuedItem(requestConfig, requestId, enqueuedAtMs, seqLen, 0);
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen, int priority) {
        return enqueuedItem(
                new FlexlbConfig(), requestId, enqueuedAtMs, seqLen, priority);
    }

    private static BatchItem enqueuedItem(
            FlexlbConfig requestConfig,
            long requestId,
            long enqueuedAtMs,
            long seqLen,
            int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(requestConfig);
        if (priority > 0) {
            balanceContext.setSchedulingMetadata(SchedulingMetadata.explicit(
                    priority, enqueuedAtMs + 30_000));
        }
        return new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
    }

    private static BatchItem expiringItem(long requestId, long enqueuedAtMs,
                                          long expiresAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1);
        request.setPriority(50);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(new FlexlbConfig());
        balanceContext.setSchedulingMetadata(
                SchedulingMetadata.explicit(50, expiresAtMs));
        return new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
    }

    /** Admitted under NON_BATCH dispatch, so it stays a route decision. */
    private static BatchItem routeDecisionItem(long requestId, long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        FlexlbConfig admitted = new FlexlbConfig();
        SchedulingTestConfig.useNonBatchDispatcher(admitted);
        balanceContext.setConfig(admitted);
        return new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
    }

    private static PriorityBlockingQueue<BatchItem> queueWith(BatchItem... items) {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.FIFO_QUEUE_ORDER);
        for (BatchItem item : items) {
            queue.add(item);
        }
        return queue;
    }

    private static BatcherContext context(String key, PrefillEndpoint endpoint,
                                          FlexlbConfig config, DecisionGroupHandler handler,
                                          PriorityBlockingQueue<BatchItem> queue,
                                          BatchSchedulerReporter reporter) {
        return context(
                key,
                endpoint,
                config,
                handler,
                TestCapacityAdmission.alwaysAvailable(),
                queue,
                reporter);
    }

    private static BatcherContext context(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DecisionGroupHandler handler,
            DeliveryCapacityAdmission capacityAdmission,
            PriorityBlockingQueue<BatchItem> queue,
            BatchSchedulerReporter reporter) {
        return new BatcherContext(
                key,
                endpoint,
                config,
                handler,
                capacityAdmission,
                queue,
                new AtomicInteger(queue.size()),
                new AtomicLong(),
                new ReentrantLock(),
                WorkerBatcher.FIFO_QUEUE_ORDER,
                reporter);
    }

    /**
     * Test capacity gate with one observable, authoritative group reservation
     * attempt. Per-request capacity remains independently available.
     */
    private static DeliveryCapacityAdmission batchCapacityAdmission(
            BooleanSupplier batchCapacityAvailable,
            AtomicInteger batchAdmissionAttempts) {
        DeliveryCapacityAdmission available = TestCapacityAdmission.alwaysAvailable();
        return new DeliveryCapacityAdmission() {
            @Override
            public AdmissionResult tryReserveItemCapacity(BatchItem item) {
                return available.tryReserveItemCapacity(item);
            }

            @Override
            public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
                batchAdmissionAttempts.incrementAndGet();
                if (!batchCapacityAvailable.getAsBoolean()) {
                    return new BatchCapacityUnavailable(
                            CapacityResource.PREFILL_BATCH,
                            batchCapacityAvailable::getAsBoolean);
                }
                return available.tryReserveBatchCapacity(head);
            }
        };
    }
}
