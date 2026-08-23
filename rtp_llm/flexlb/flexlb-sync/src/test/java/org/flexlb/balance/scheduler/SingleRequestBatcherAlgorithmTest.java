package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isA;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SingleRequestBatcherAlgorithmTest {

    @Test
    void computeCapacityDropAtFinalGateKeepsActiveThenRejectsNextTick()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        WorkerStatus status = statusWithUnlimitedKv();
        // seqLen=100 fits the first strict limit, but not the final re-read.
        when(status.getMaxBatchTokensSize()).thenReturn(200L, 100L);
        PrefillEndpoint endpoint = endpoint(status);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem item = batchItem(config, 1L, 100L);
        BatcherContext context = context(config, endpoint, handler, item);

        SingleRequestBatcherAlgorithm algorithm = new SingleRequestBatcherAlgorithm();
        algorithm.processQueue(context);

        verify(handler, never()).onDecisionGroupReady(anyList(),
                org.mockito.ArgumentMatchers.any());
        verify(handler, never()).onOfferFailure(eq(item),
                isA(BatchTokenCapacityExceededException.class));
        assertEquals(1, context.size());
        assertSame(item, context.peek());

        // Mockito repeats the last value: the next tick observes the now
        // permanently impossible singleton at the normal rejection gate.
        algorithm.processQueue(context);

        verify(handler).onOfferFailure(eq(item),
                isA(BatchTokenCapacityExceededException.class));
        assertEquals(0, context.size());
    }

    @Test
    void kvCapacityDropAtFinalGateKeepsRequestActive()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getMaxBatchTokensSize()).thenReturn(1_000L);
        when(status.getTotalKvCacheTokens()).thenReturn(new AtomicLong(1_000L));
        when(status.getAvailableKvCacheTokens()).thenReturn(
                new AtomicLong(200L), new AtomicLong(50L));
        PrefillEndpoint endpoint = endpoint(status);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem item = batchItem(config, 2L, 100L);
        BatcherContext context = context(config, endpoint, handler, item);

        new SingleRequestBatcherAlgorithm().processQueue(context);

        verify(status, times(2)).getAvailableKvCacheTokens();
        verify(handler, never()).onDecisionGroupReady(anyList(),
                org.mockito.ArgumentMatchers.any());
        assertEquals(1, context.size());
        assertSame(item, context.peek());
    }

    @Test
    void batchInflightCapacityDropAtFinalGateKeepsRequestActive()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        SchedulingTestConfig.useBatchDispatcher(config)
                .setMaxInflightBatchesPerPrefillWorker(1);
        WorkerStatus status = statusWithUnlimitedKv();
        when(status.getMaxBatchTokensSize()).thenReturn(1_000L);
        PrefillEndpoint endpoint = endpoint(status);
        when(endpoint.getInflightBatchCount()).thenReturn(0, 1);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem item = batchItem(config, 3L, 100L);
        BatcherContext context = context(config, endpoint, handler, item);

        new SingleRequestBatcherAlgorithm().processQueue(context);

        verify(endpoint, times(2)).getInflightBatchCount();
        verify(handler, never()).onDecisionGroupReady(anyList(),
                org.mockito.ArgumentMatchers.any());
        assertEquals(1, context.size());
        assertSame(item, context.peek());
    }

    private static FlexlbConfig singleBatchConfig() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config)
                .setMaxInflightBatchesPerPrefillWorker(0);
        return config;
    }

    private static WorkerStatus statusWithUnlimitedKv() {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getTotalKvCacheTokens()).thenReturn(new AtomicLong());
        when(status.getAvailableKvCacheTokens()).thenReturn(new AtomicLong());
        return status;
    }

    private static PrefillEndpoint endpoint(WorkerStatus status) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        return endpoint;
    }

    private static BatcherContext context(FlexlbConfig config,
                                          PrefillEndpoint endpoint,
                                          DecisionGroupHandler handler,
                                          BatchItem item) {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.FIFO_QUEUE_ORDER);
        queue.add(item);
        return new BatcherContext("single-test", endpoint, config, handler,
                queue, mock(BatchSchedulerReporter.class));
    }

    private static BatchItem batchItem(FlexlbConfig config,
                                       long requestId,
                                       long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(config);
        balanceContext.setSchedulingMetadata(
                SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return new BatchItem(balanceContext, new CompletableFuture<>(), null,
                null, null, null, null, System.currentTimeMillis());
    }
}
