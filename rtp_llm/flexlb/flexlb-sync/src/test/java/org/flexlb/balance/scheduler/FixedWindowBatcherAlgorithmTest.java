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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

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
    void capsBatchGrowthAtPredictedExecutionBudget() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");
        List<Integer> predictedSizes = new ArrayList<>();
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation -> {
            int size = ((List<?>) invocation.getArgument(0)).size();
            predictedSizes.add(size);
            return size == 1 ? 499.0 : 500.0;
        });

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
        assertEquals(List.of(1L), items.getValue().stream()
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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis() - 170)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
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
        verify(predictor, times(32)).predictBatchMs(anyList());
    }

    @Test
    void singletonPredictionAtThresholdDispatchesImmediately() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 131_072)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L), dispatched.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        verify(predictor, times(1)).predictBatchMs(anyList());
    }

    @Test
    void fullBatchDispatchesLargestFeasibleGroup() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(4);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(499.0);

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 1),
                        enqueuedItem(2, now + 1, 1),
                        enqueuedItem(3, now + 2, 1),
                        enqueuedItem(4, now + 3, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L, 2L, 3L, 4L), dispatched.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_full", meta.getValue().reason());
        assertEquals(0, context.size());
        verify(predictor, times(4)).predictBatchMs(anyList());
    }

    @Test
    void timeoutDispatchesLargestFeasibleGroup() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(4);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenAnswer(invocation ->
                ((List<?>) invocation.getArgument(0)).size() * 200.0);

        long old = System.currentTimeMillis() - 1_000;
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, old, 1), enqueuedItem(2, old + 1, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(2, dispatched.getValue().size());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        verify(predictor, times(2)).predictBatchMs(anyList());
    }

    @Test
    void nonMonotonicPredictionCapsAtTheFirstOverBudgetMember() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(5);

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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 1),
                        enqueuedItem(2, now + 1, 1),
                        enqueuedItem(3, now + 2, 1),
                        enqueuedItem(4, now + 3, 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(
                dispatched.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(1L, 2L), dispatched.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals(List.of(1, 2, 3), predictedSizes,
                "growth stops at the spike instead of climbing back to four");
    }

    @Test
    void capacityLimitedFullFallbackDispatchesLargestFeasiblePrefix()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(4);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(100.0);

        long now = System.currentTimeMillis();
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 20),
                        enqueuedItem(2, now + 1, 20),
                        enqueuedItem(3, now + 2, 60),
                        enqueuedItem(4, now + 3, 5)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L, 2L), dispatched.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_full", meta.getValue().reason());
        assertEquals(2, context.size());
        verify(predictor, times(2)).predictBatchMs(anyList());
    }

    @Test
    void batchInflightCapPrecedesPrediction() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getInflightBatchCount()).thenReturn(1);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(1_000.0);

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(handler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        verifyNoInteractions(predictor);
        assertEquals(1, context.size());
    }

    @Test
    void zeroBatchInflightCapKeepsGateDisabled() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(0);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("predicted_execution_cap", meta.getValue().reason());
        verify(endpoint, never()).getInflightBatchCount();
    }

    @Test
    void nanPredictionFallsBackToFixedWindowTimeout() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(predictor.predictBatchMs(anyList())).thenReturn(Double.NaN);

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        verify(predictor, times(1)).predictBatchMs(anyList());
    }

    @Test
    void fixedWindowBatchUsesEnginePaddedTokenCost() throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

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
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(dispatched.capture(), meta.capture());
        assertEquals(List.of(1L), dispatched.getValue().stream().map(BatchItem::requestId).toList());
        assertEquals(60L, dispatched.getValue().stream().mapToLong(BatchItem::seqLen).sum());
        assertEquals("fixed_window_timeout", meta.getValue().reason());
        assertEquals(2, context.size());
        assertEquals(2L, context.peek().requestId());
    }

    @Test
    void largeMrcrRequestIsDispatchedAloneWhenPaddedBatchWouldOverflow() throws InterruptedException {
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(13);

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
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

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

        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(requestCount);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

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
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

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
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

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
    void offerBetweenAdvisoryPeekAndSnapshotUsesStablePriorityOrder()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
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
        AtomicInteger inflightReads = new AtomicInteger();
        when(endpoint.getInflightBatchCount()).thenAnswer(ignored -> {
            if (inflightReads.incrementAndGet() == 1) {
                // The first read follows the lock-free advisory peek of low.
                // The authoritative ordered snapshot must observe high first.
                queueLock.lock();
                try {
                    queue.add(high);
                    queueDepth.incrementAndGet();
                    queueVersion.incrementAndGet();
                } finally {
                    queueLock.unlock();
                }
            }
            return 0;
        });
        when(predictor.predictBatchMs(anyList())).thenReturn(100.0);

        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler, queue, queueDepth,
                queueVersion, queueLock, WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> delivered = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(
                delivered.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(high, low), delivered.getValue());
        assertTrue(context.isActiveEmpty());
        assertEquals(0, context.size());
        verify(predictor, times(2)).predictBatchMs(anyList());
    }

    @Test
    void routeHeadStopsAtModeBoundaryAndCannotBypassBatchInflightGate()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(100.0);
        when(endpoint.getInflightBatchCount()).thenReturn(1);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        // A batch dispatcher configures no per-request inflight cap, so route
        // delivery is uncapped.
        when(endpoint.availableRequestSlots(0)).thenReturn(Integer.MAX_VALUE);
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
        verify(predictor, times(1)).predictBatchMs(anyList());

        // Once the BATCH_ENQUEUE item becomes head, its own inflight gate applies.
        // It must not have ridden along with the preceding route decision.
        algorithm.processQueue(context);

        verify(handler, times(1)).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        verify(endpoint).getInflightBatchCount();
        verify(predictor, times(1)).predictBatchMs(anyList());
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
    }

    @Test
    void queueMutationBetweenPredictionAndStageInvalidatesWholeGroup()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        long now = System.currentTimeMillis();
        BatchItem first = enqueuedItem(1, now, 1);
        BatchItem second = enqueuedItem(2, now + 1, 1);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
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
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(List.of(first), context.sortedItems());
    }

    @Test
    void continuousLowAndHighPriorityOffersDoNotStarveCapturedGroups()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(4);

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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = new BatcherContext(
                "test", endpoint, config, handler, queue, queueDepth,
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

        ArgumentCaptor<List<BatchItem>> delivered = ArgumentCaptor.forClass(List.class);
        verify(handler, times(50)).onDecisionGroupReady(
                delivered.capture(), any(DecisionGroupMetadata.class));
        assertTrue(delivered.getAllValues().stream().allMatch(group -> group.size() == 4));
        assertEquals(200, delivered.getAllValues().stream().mapToInt(List::size).sum());
        assertEquals(50, context.size());
        assertEquals(50, predictionCalls.get());
        verify(predictor, times(200)).predictBatchMs(anyList());
    }

    @Test
    void headRemovalAfterStableSnapshotCannotSplitDecisionGroup()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(600.0);
        long now = System.currentTimeMillis();
        BatchItem first = enqueuedItem(1, now, 1);
        BatchItem second = enqueuedItem(2, now + 1, 1);
        AtomicReference<BatcherContext> contextRef = new AtomicReference<>();
        AtomicInteger inflightReads = new AtomicInteger();
        when(endpoint.getInflightBatchCount()).thenAnswer(ignored -> {
            // Read 1 is the cheap advisory gate. Read 2 happens after the
            // ordered snapshot was captured but before prediction.
            if (inflightReads.incrementAndGet() == 2) {
                assertTrue(contextRef.get().remove(first));
            }
            return 0;
        });
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler, queueWith(first, second),
                mock(BatchSchedulerReporter.class));
        contextRef.set(context);

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(List.of(second), context.sortedItems());
    }

    @Test
    void learningRevisionChangeDuringPredictionPreventsStaleStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

        AtomicLong generation = new AtomicLong();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.generation()).thenAnswer(ignored -> generation.get());
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            generation.incrementAndGet();
            return 600.0;
        });
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(1, context.size());
    }

    @Test
    void learningRevisionChangeDuringFinalGatePreventsStaleStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        AtomicLong generation = new AtomicLong();
        AtomicInteger inflightReads = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getInflightBatchCount()).thenAnswer(ignored -> {
            if (inflightReads.incrementAndGet() == 3) {
                generation.incrementAndGet();
            }
            return 0;
        });
        when(predictor.generation()).thenAnswer(ignored -> generation.get());
        when(predictor.predictBatchMs(anyList())).thenReturn(600.0);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(endpoint, times(3)).getInflightBatchCount();
        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(1, context.size());
    }

    @Test
    void learningRevisionChangeAtAtomicStagePreventsStaleStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(1, context.size());
    }

    @Test
    void batchInflightLimitReachedDuringPredictionPreventsStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(1);

        AtomicInteger inflight = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getInflightBatchCount()).thenAnswer(ignored -> inflight.get());
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            inflight.incrementAndGet();
            return 600.0;
        });
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 1)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(endpoint, times(3)).getInflightBatchCount();
        verify(predictor, times(1)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(1, context.size());
    }

    @Test
    void engineComputeCapacityDropDuringPredictionPreventsStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 100),
                        enqueuedItem(2, now + 1, 100)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(2)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(2, context.size());
    }

    @Test
    void engineKvCapacityDropDuringPredictionPreventsStage()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(
                        enqueuedItem(1, now, 100),
                        enqueuedItem(2, now + 1, 100)),
                mock(BatchSchedulerReporter.class));

        new FixedWindowBatcherAlgorithm().processQueue(context);

        verify(predictor, times(2)).predictBatchMs(anyList());
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(2, context.size());
    }

    @Test
    void singletonWaitsWithoutPredictionWhenKvIsInsufficient()
            throws InterruptedException {
        FlexlbConfig config = batchConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) 500);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(1_000);
        status.getAvailableKvCacheTokens().set(99);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = context(
                "test", endpoint, config, handler,
                queueWith(enqueuedItem(1, System.currentTimeMillis(), 100)),
                mock(BatchSchedulerReporter.class));
        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm();

        algorithm.processQueue(context);
        algorithm.processQueue(context);

        verifyNoInteractions(predictor);
        verify(handler, never()).onDecisionGroupReady(anyList(), any());
        assertEquals(1, context.size());
    }

    // ---- queue_deadline_exceeded valve gating ----

    // ---- helpers ----

    private static FlexlbConfig batchConfig() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEarlyDispatchPredictedExecutionMs((long) (500));
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(160);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(32);
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

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(new FlexlbConfig());
        if (priority > 0) {
            balanceContext.setSchedulingMetadata(SchedulingMetadata.explicit(
                    priority, enqueuedAtMs + 30_000));
        }
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
        return new BatcherContext(key, endpoint, config, handler, queue,
                new AtomicInteger(queue.size()), reporter);
    }
}
