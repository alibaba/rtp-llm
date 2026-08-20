package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.PriorityOrdering;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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
    void estimate_wait_counts_only_items_ahead_and_is_monotonic_in_priority() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        // Ancient arrivals zero out the head's remaining window for determinism
        assertTrue(batcher.tryOffer(item(1, 50, now, now - 100_000, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now, now - 100_000, 128)));

        PrefillQueueManager manager = batcher.queueManager();
        long waitP70 = manager.estimateWaitMs(70, 999);
        long waitP50 = manager.estimateWaitMs(50, 999);
        long waitP30 = manager.estimateWaitMs(30, 999);

        // P70 jumps ahead of both P50 items: 0 cycles ahead
        assertEquals(0, waitP70);
        // P50/P30 wait behind both: 2 cycles x avgDecisionIntervalMs
        // (no dispatch observed yet -> fixed_window fallback = fixedWaitMs)
        assertEquals(400, waitP50);
        assertEquals(400, waitP30);
        assertTrue(waitP70 <= waitP50 && waitP50 <= waitP30);
    }

    @Test
    void direct_wait_scan_matches_sorted_reference_for_random_active_and_ready_queues() {
        Random random = new Random(0x5CB7E9118L);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(anyInt())).thenReturn(0);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);

        for (int scenario = 0; scenario < 250; scenario++) {
            FlexlbConfig scenarioConfig = new FlexlbConfig();
            SchedulingTestConfig.usePriorityQueue(scenarioConfig);
            BatchDispatcherConfig scenarioDispatcher =
                    SchedulingTestConfig.useBatchDispatcher(scenarioConfig);
            scenarioDispatcher.setMaxCollectionWaitMs(1 + random.nextInt(500));
            scenarioDispatcher.setMaxRequests(1 + random.nextInt(64));

            int itemCount = random.nextInt(129);
            boolean forceOrderingTie = scenario % 3 == 0;
            if (forceOrderingTie) {
                itemCount = Math.max(3, itemCount);
            }
            int maxReadyCount = forceOrderingTie ? itemCount - 3 : itemCount;
            int readyCount = itemCount == 0
                    ? 0 : random.nextInt(maxReadyCount + 1);
            long arrivalMs = 1_700_000_000_000L + scenario * 10_000L;
            int incomingPriority = 1 + random.nextInt(100);
            long incomingRequestId = 10_000_000L + scenario * 1_000L + 500L;
            PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                    Math.max(11, itemCount), WorkerBatcher.PRIORITY_QUEUE_ORDER);
            List<BatchItem> items = new ArrayList<>(itemCount);
            for (int index = 0; index < itemCount; index++) {
                boolean tiedWithProbe = forceOrderingTie && index < 2;
                long itemRequestId = tiedWithProbe
                        ? incomingRequestId + (index == 0 ? -1L : 1L)
                        : 20_000_000L + scenario * 1_000L + index;
                int itemPriority = tiedWithProbe
                        ? incomingPriority : 1 + random.nextInt(100);
                long itemArrivalMs = tiedWithProbe
                        ? arrivalMs : arrivalMs - 1_000L + random.nextInt(2_001);
                BatchItem item = routeItem(
                        itemRequestId,
                        itemPriority,
                        itemArrivalMs,
                        128L + random.nextInt(8_192));
                queue.add(item);
                items.add(item);
            }

            BatcherContext context = new BatcherContext(
                    "random-" + scenario,
                    endpoint,
                    scenarioConfig,
                    handler,
                    queue,
                    new AtomicInteger(itemCount),
                    new AtomicLong(),
                    new ReentrantLock(),
                    WorkerBatcher.PRIORITY_QUEUE_ORDER,
                    reporter);
            if (readyCount > 0) {
                context.stageDecisionGroup(
                        items.subList(itemCount - readyCount, itemCount),
                        new DecisionGroupMetadata("random-ready", itemCount));
            }

            long expected = sortedReferenceEstimate(
                    context, incomingPriority, arrivalMs, incomingRequestId);
            long actual = context.estimateIncomingWaitMs(
                    incomingPriority, arrivalMs, incomingRequestId);
            assertEquals(expected, actual,
                    "direct scan diverged from sorted reference in scenario " + scenario);
        }
    }

    @Test
    void queue_wait_view_releases_queue_members_once_the_worker_is_idle() {
        BatcherContext context = contextWithActiveItems(512, "wait-view");

        context.estimateIncomingWaitMs(50, 1_700_000_000_000L, Long.MAX_VALUE);
        assertEquals(512, context.queueWaitViewRetainedItemsForTest());

        context.stopAndDrainTo(new ArrayList<>());
        context.estimateIncomingWaitMs(50, 1_700_000_000_000L, Long.MAX_VALUE);
        assertEquals(0, context.queueWaitViewRetainedItemsForTest());
    }

    @Test
    void direct_wait_scan_anchors_window_on_the_longest_waiting_member() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(200);
        long arrivalMs = 1_700_000_000_000L;
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem lowerPriorityEarlier = routeItem(
                1L, 20, arrivalMs - 150L, 128L);
        BatchItem higherPriorityLater = routeItem(
                2L, 80, arrivalMs - 50L, 128L);
        queue.add(lowerPriorityEarlier);
        queue.add(higherPriorityLater);
        BatcherContext context = new BatcherContext(
                "priority-ordered",
                mock(PrefillEndpoint.class),
                config,
                mock(DecisionGroupHandler.class),
                queue,
                new AtomicInteger(2),
                new AtomicLong(),
                new ReentrantLock(),
                WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));

        long expected = sortedReferenceEstimate(context, 100, arrivalMs, 999L);
        long actual = context.estimateIncomingWaitMs(100, arrivalMs, 999L);
        assertEquals(50L, expected,
                "the 150ms-old member leaves 50ms of window, whichever item sorts first");
        assertEquals(expected, actual);
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

    /** Admitted under NON_BATCH dispatch, so it stays a route decision. */
    private static BatchItem routeItem(long requestId, int priority,
                                       long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        FlexlbConfig admitted = new FlexlbConfig();
        SchedulingTestConfig.useNonBatchDispatcher(admitted);
        ctx.setConfig(admitted);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    /** Sorted-scan model of the wait estimate, independent of the cached view. */
    private static long sortedReferenceEstimate(BatcherContext context,
                                                int priority,
                                                long arrivalMs,
                                                long requestId) {
        int activeItemsAhead = 0;
        int readyItemsAhead = 0;
        long windowOpenedAtMs = Long.MAX_VALUE;
        context.queueLock().lock();
        try {
            for (BatchItem item : context.sortedQueuedItems()) {
                if (item.readyDeliveryReason() != null) {
                    readyItemsAhead++;
                    continue;
                }
                windowOpenedAtMs = Math.min(windowOpenedAtMs, item.enqueuedAtMs());
                if (PriorityOrdering.comesBefore(
                        item, item.requestId(), priority, arrivalMs, requestId)) {
                    activeItemsAhead++;
                }
            }
        } finally {
            context.queueLock().unlock();
        }
        long perCycleMs = context.avgDecisionIntervalMs();
        long readyDrainMs = (long) readyItemsAhead * perCycleMs;
        long activeDrainMs =
                (long) (activeItemsAhead / context.maxDecisionRequests()) * perCycleMs;
        long remainingWindowMs = 0L;
        if (windowOpenedAtMs != Long.MAX_VALUE) {
            long elapsedMs = Math.max(0L, arrivalMs - windowOpenedAtMs);
            remainingWindowMs = Math.max(0L, context.collectionWindowMs() - elapsedMs);
        }
        return readyDrainMs + activeDrainMs + remainingWindowMs;
    }

    private BatcherContext contextWithActiveItems(int itemCount, String key) {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                Math.max(11, itemCount), WorkerBatcher.PRIORITY_QUEUE_ORDER);
        long arrivalMs = 1_699_999_900_000L;
        for (int index = 0; index < itemCount; index++) {
            queue.add(routeItem(index + 1L, 1 + index % 100,
                    arrivalMs + index, 128L + index));
        }
        return new BatcherContext(
                key,
                mock(PrefillEndpoint.class),
                config,
                mock(DecisionGroupHandler.class),
                queue,
                new AtomicInteger(itemCount),
                new AtomicLong(),
                new ReentrantLock(),
                WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
    }
}
