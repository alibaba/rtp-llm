package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Phase 2 tests for {@link PrefillQueueManager} + {@link WorkerBatcher}:
 * PRIORITY queue order, wait estimation, and FIFO-order regression.
 *
 * <p>Uses fixed-window batching with priority ordering, so the batcher can be
 * built without a live {@code PrefillEndpoint}. The batcher is never started —
 * the queue is inspected/mutated directly through the manager facade.
 */
class PrefillQueueManagerTest {

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
    }

    private WorkerBatcher newBatcher() {
        return new WorkerBatcher("test-worker", null, config,
                mock(DecisionGroupHandler.class), mock(BatchSchedulerReporter.class));
    }

    // ==================== 8.1 queue order ====================

    @Test
    void priority_order_is_priority_desc_then_enqueue_fifo() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // Insertion order deliberately scrambled
        assertTrue(batcher.tryOffer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(batcher.tryOffer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(batcher.tryOffer(item(4, 50, now + 5_000, now - 100, 128)));

        PrefillQueueSnapshot snapshot = batcher.queueManager().snapshot();
        List<Long> order = snapshot.items().stream().map(QueuedRequestSnapshot::requestId).toList();

        // P70 first (priority desc); P50s preserve offer order. Neither the
        // supplied timestamp nor expiration changes same-priority FIFO.
        assertEquals(List.of(2L, 1L, 3L, 4L), order);
        assertEquals(4, snapshot.items().size());
        assertEquals(SchedulingTestConfig.useBatchDispatcher(config).getMaxWaitingRequestsPerPrefillWorker(), snapshot.queueCapacity());
        for (QueuedRequestSnapshot item : snapshot.items()) {
            assertEquals(QueuedRequestSnapshot.PREFILL_QUEUED, item.state());
        }
    }

    @Test
    void priority_order_uses_unique_enqueue_sequence_before_request_id() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // Same priority and supplied arrival timestamp still preserve the
        // actual offer sequence. requestId is only a defensive final tie-break
        // after the unique enqueue sequence.
        assertTrue(batcher.tryOffer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(batcher.tryOffer(item(3, 50, now + 9_000, now, 128)));

        List<Long> order = batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId).toList();
        assertEquals(List.of(1L, 2L, 4L, 3L), order);
    }

    @Test
    void fifo_order_ignores_priority() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // High priority arrives last: FIFO ordering must keep offer order.
        assertTrue(batcher.tryOffer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(batcher.tryOffer(item(3, 70, now + 100, now + 200, 128)));

        List<Long> order = batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId).toList();
        assertEquals(List.of(1L, 2L, 3L), order);
    }

    // ==================== 8.4 wait estimate ====================

    @Test
    void samePriorityPriorityAndFifoHaveTheSameCollectionDelay() {
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(200);
        long now = System.currentTimeMillis();
        WorkerBatcher priorityPartial = newBatcher();
        assertTrue(priorityPartial.tryOffer(item(1, 50, now, now, 128)));
        assertTrue(priorityPartial.tryOffer(item(2, 50, now, now, 128)));
        long priorityPartialWait = priorityPartial.queueManager().estimateWaitMs(50, 999);
        assertTrue(priorityPartial.tryOffer(item(3, 50, now, now, 128)));
        long priorityFullWait = priorityPartial.queueManager().estimateWaitMs(50, 1_000);

        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher fifoPartial = newBatcher();
        assertTrue(fifoPartial.tryOffer(item(4, 50, now, now, 128)));
        assertTrue(fifoPartial.tryOffer(item(5, 50, now, now, 128)));
        long fifoPartialWait = fifoPartial.queueManager().estimateWaitMs(50, 1_001);
        assertTrue(fifoPartial.tryOffer(item(6, 50, now, now, 128)));
        long fifoFullWait = fifoPartial.queueManager().estimateWaitMs(50, 1_002);

        assertEquals(200L, priorityPartialWait);
        assertEquals(priorityPartialWait, fifoPartialWait);
        assertEquals(0L, priorityFullWait);
        assertEquals(priorityFullWait, fifoFullWait);
    }

    @Test
    void singleDecisionAlwaysReportsZeroCollectionDelay() {
        SchedulingTestConfig.useSingleDecision(config);
        long now = System.currentTimeMillis();
        WorkerBatcher priority = newBatcher();
        assertTrue(priority.tryOffer(item(1, 90, now, now, 128)));
        assertTrue(priority.tryOffer(item(2, 10, now, now, 128)));
        assertEquals(0L, priority.queueManager().estimateWaitMs(50, 999));

        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher fifo = newBatcher();
        assertTrue(fifo.tryOffer(item(3, 90, now, now, 128)));
        assertTrue(fifo.tryOffer(item(4, 10, now, now, 128)));
        assertEquals(0L, fifo.queueManager().estimateWaitMs(50, 1_000));
    }

    @Test
    void highestPriorityProbeHasNoDelayWhenLaterMembersCanFillItsGroup() {
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(1, 30, now, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 20, now, now, 128)));
        assertTrue(batcher.tryOffer(item(3, 10, now, now, 128)));

        assertEquals(0L, batcher.queueManager().estimateWaitMs(100, 999));
    }

    @Test
    void highestPriorityProbePaysWindowWhenItsGroupRemainsPartial() {
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(4);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(1, 30, now, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 20, now, now, 128)));

        assertEquals(200L, batcher.queueManager().estimateWaitMs(100, 999));
    }

    // ==================== helpers ====================

    private BatchItem item(long requestId, int priority, long expiresAtMs,
                           long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, expiresAtMs));
        BatchItem item = new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
        return item;
    }

}
