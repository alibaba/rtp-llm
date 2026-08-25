package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ImmediateNonBatchAlgorithmTest {

    @Test
    void routesLongSingletonAboveEngineBatchTokenBudget() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(128);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(1_048_576L);
        status.setMaxBatchTokensSize(409_600L);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.availableRequestSlots(anyInt())).thenReturn(1);

        BatchItem item = routeItem(910_537L);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        queue.add(item);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext context = new BatcherContext(
                "prefill-worker", endpoint, config, handler, queue,
                new AtomicInteger(1), new AtomicLong(), new ReentrantLock(),
                WorkerBatcher.PRIORITY_QUEUE_ORDER, mock(BatchSchedulerReporter.class));

        new ImmediateNonBatchAlgorithm().processQueue(context);

        ArgumentCaptor<List<BatchItem>> items = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> metadata =
                ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(items.capture(), metadata.capture());
        verify(handler, never()).onOfferFailure(any(), any());
        assertEquals(List.of(item), items.getValue());
        assertEquals("non_batch_immediate", metadata.getValue().reason());
        assertEquals(0, context.size());
    }

    private static BatchItem routeItem(long seqLen) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(seqLen);
        request.setPriority(50);

        FlexlbConfig requestConfig = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(requestConfig);
        SchedulingTestConfig.useNonBatchDispatcher(requestConfig);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setConfig(requestConfig);
        balanceContext.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return new BatchItem(balanceContext, new CompletableFuture<>(), null,
                null, null, null, null, System.currentTimeMillis());
    }
}
