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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isA;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class SingleRequestBatcherAlgorithmTest {

    private static DecisionGroupHandler resolvingHandler() {
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        doAnswer(invocation -> {
            TestCapacityAdmission.complete(invocation.getArgument(0));
            return null;
        }).when(handler).onDecisionGroupAdmitted(
                any(AdmittedDecisionGroup.class),
                org.mockito.ArgumentMatchers.any());
        return handler;
    }

    @Test
    void computeFeasibilityChangeBeforeAdmissionKeepsActiveThenRejectsNextTick()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        WorkerStatus status = statusWithUnlimitedKv();
        // seqLen=100 fits the first strict limit, but not the final re-read.
        when(status.getMaxBatchTokensSize()).thenReturn(200L, 100L);
        PrefillEndpoint endpoint = endpoint(status);
        DecisionGroupHandler handler = resolvingHandler();
        BatchItem item = batchItem(config, 1L, 100L);
        BatcherContext context = context(config, endpoint, handler, item);

        SingleRequestBatcherAlgorithm algorithm = new SingleRequestBatcherAlgorithm();
        algorithm.processQueue(context);

        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class),
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
    void kvFeasibilityChangeBeforeAdmissionKeepsRequestActive()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getMaxBatchTokensSize()).thenReturn(1_000L);
        when(status.getTotalKvCacheTokens()).thenReturn(new AtomicLong(1_000L));
        when(status.getAvailableKvCacheTokens()).thenReturn(
                new AtomicLong(200L), new AtomicLong(50L));
        PrefillEndpoint endpoint = endpoint(status);
        DecisionGroupHandler handler = resolvingHandler();
        BatchItem item = batchItem(config, 2L, 100L);
        BatcherContext context = context(config, endpoint, handler, item);

        new SingleRequestBatcherAlgorithm().processQueue(context);

        verify(status, times(2)).getAvailableKvCacheTokens();
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class),
                org.mockito.ArgumentMatchers.any());
        assertEquals(1, context.size());
        assertSame(item, context.peek());
    }

    @Test
    void insufficientKvReturnsEventDrivenWaitWithoutPolling()
            throws InterruptedException {
        long nowMs = 2_000_000L;
        long expiresAtMs = 2_010_000L;
        FlexlbConfig config = singleBatchConfig();
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getMaxBatchTokensSize()).thenReturn(1_000L);
        when(status.getTotalKvCacheTokens()).thenReturn(new AtomicLong(1_000L));
        when(status.getAvailableKvCacheTokens()).thenReturn(new AtomicLong(99L));
        PrefillEndpoint endpoint = endpoint(status);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        DeliveryCapacityAdmission capacityAdmission =
                mock(DeliveryCapacityAdmission.class);
        BatchItem head = batchItem(
                config, 20L, 100L, 1_999_000L, expiresAtMs);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.FIFO_QUEUE_ORDER);
        queue.add(head);
        AtomicLong queueVersion = new AtomicLong(73L);
        ReentrantLock queueLock = new ReentrantLock();
        BatcherContext context = new BatcherContext(
                "single-kv-wait-test",
                endpoint,
                config,
                handler,
                capacityAdmission,
                queue,
                new AtomicInteger(1),
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
            context.incrementSchedulingInputVersion();
        } finally {
            queueLock.unlock();
        }

        BatcherCycleResult result =
                new SingleRequestBatcherAlgorithm().processQueue(context);

        BatcherCycleResult.AwaitingSchedulingChange waiting = assertInstanceOf(
                BatcherCycleResult.AwaitingSchedulingChange.class, result);
        assertSame(head, waiting.head());
        assertEquals(73L, waiting.queueVersion());
        assertEquals(2L, waiting.schedulingInputVersion());
        assertEquals(expiresAtMs, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY,
                waiting.reason());
        verify(status, times(1)).getAvailableKvCacheTokens();
        verifyNoInteractions(handler, capacityAdmission);
        assertEquals(1, context.size());
        assertSame(head, context.peek());
    }

    @Test
    void batchCapacityAdmissionBlocksAndKeepsRequestActive()
            throws InterruptedException {
        FlexlbConfig config = singleBatchConfig();
        SchedulingTestConfig.useBatchDispatcher(config)
                .setMaxInflightBatchesPerPrefillWorker(1);
        WorkerStatus status = statusWithUnlimitedKv();
        when(status.getMaxBatchTokensSize()).thenReturn(1_000L);
        PrefillEndpoint endpoint = endpoint(status);
        AtomicInteger batchAdmissionAttempts = new AtomicInteger();
        DeliveryCapacityAdmission available = TestCapacityAdmission.alwaysAvailable();
        DeliveryCapacityAdmission capacityAdmission = new DeliveryCapacityAdmission() {
            @Override
            public AdmissionResult tryReserveItemCapacity(BatchItem candidate) {
                return available.tryReserveItemCapacity(candidate);
            }

            @Override
            public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
                batchAdmissionAttempts.incrementAndGet();
                return new BatchCapacityUnavailable(
                        CapacityResource.PREFILL_BATCH, () -> false);
            }
        };
        DecisionGroupHandler handler = resolvingHandler();
        BatchItem item = batchItem(config, 3L, 100L);
        BatcherContext context = context(
                config, endpoint, handler, capacityAdmission, item);

        new SingleRequestBatcherAlgorithm().processQueue(context);

        assertEquals(1, batchAdmissionAttempts.get());
        verify(endpoint, never()).getInflightBatchCount();
        verify(handler, never()).onDecisionGroupAdmitted(any(AdmittedDecisionGroup.class),
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
        return context(
                config,
                endpoint,
                handler,
                TestCapacityAdmission.alwaysAvailable(),
                item);
    }

    private static BatcherContext context(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            DecisionGroupHandler handler,
            DeliveryCapacityAdmission capacityAdmission,
            BatchItem item) {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.FIFO_QUEUE_ORDER);
        queue.add(item);
        return new BatcherContext(
                "single-test",
                endpoint,
                config,
                handler,
                capacityAdmission,
                queue,
                new java.util.concurrent.atomic.AtomicInteger(queue.size()),
                new AtomicLong(),
                new java.util.concurrent.locks.ReentrantLock(),
                WorkerBatcher.FIFO_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
    }

    private static BatchItem batchItem(FlexlbConfig config,
                                       long requestId,
                                       long seqLen) {
        return batchItem(config, requestId, seqLen,
                System.currentTimeMillis(), Long.MAX_VALUE);
    }

    private static BatchItem batchItem(
            FlexlbConfig config,
            long requestId,
            long seqLen,
            long enqueuedAtMs,
            long expiresAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(config);
        balanceContext.setSchedulingMetadata(
                SchedulingMetadata.explicit(50, expiresAtMs));
        return new BatchItem(balanceContext, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }
}
