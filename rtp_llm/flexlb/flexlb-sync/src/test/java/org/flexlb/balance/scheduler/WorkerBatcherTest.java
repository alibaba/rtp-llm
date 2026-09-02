package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link WorkerBatcher#queueSizeByPriority()}: per-priority
 * bucketing, the legacy priority-0 bucket, and the present-only empty-bucket
 * behavior (drained priorities disappear from the snapshot — same convention
 * as the batch wait-time-by-priority series).
 *
 * <p>Same construction pattern as {@link PrefillQueueManagerTest}: the
 * {@code fixed_window} algorithm needs no predictor, and the batcher is
 * never started so the queue content is fully deterministic.
 */
class WorkerBatcherTest {

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

    @Test
    void queue_size_by_priority_buckets_multiple_priorities() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));
        assertTrue(batcher.tryOffer(item(3, 50, now)));
        assertTrue(batcher.tryOffer(item(4, 30, now)));

        Map<Integer, Integer> buckets = batcher.queueSizeByPriority();
        assertEquals(Map.of(70, 1, 50, 2, 30, 1), buckets);
        // Bucket sum matches the global queue size
        assertEquals(batcher.queueSize(), buckets.values().stream().mapToInt(Integer::intValue).sum());
    }

    @Test
    void items_without_scheduling_metadata_fall_into_priority_zero_bucket() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(legacyItem(1, now)));
        assertTrue(batcher.tryOffer(legacyItem(2, now)));

        assertEquals(Map.of(0, 2), batcher.queueSizeByPriority());
    }

    @Test
    void empty_queue_returns_empty_map() {
        assertEquals(Map.of(), newBatcher().queueSizeByPriority());
    }

    @Test
    void drained_priorities_disappear_from_snapshot() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));

        // Drain the P70 item: its bucket drops out (present-only, no zero-fill
        // — same empty-bucket behavior as wait-time-by-priority)
        List<BatchItem> removed = batcher.tryRemove(List.of(1L), "test-drain");
        assertEquals(1, removed.size());

        assertEquals(Map.of(50, 1), batcher.queueSizeByPriority());

        // Fully drained queue reports no buckets at all
        assertEquals(1, batcher.tryRemove(List.of(2L), "test-drain").size());
        assertEquals(Map.of(), batcher.queueSizeByPriority());
    }

    @Test
    void decisionCallbackFailure_restoresOnlyStagedItemsWithoutDepthLeak() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem first = item(1, 50, 100);
        BatchItem second = item(2, 50, 200);
        queue.add(first);
        queue.add(second);
        AtomicInteger depth = new AtomicInteger(2);
        BatcherContext ctx = context(queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                throw new IllegalStateException("test callback failure");
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }

            @Override
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });

        assertThrows(IllegalStateException.class,
                () -> ctx.stageForDelivery(List.of(first, second), new DecisionGroupMetadata("test", 0)));

        assertEquals(2, depth.get());
        assertEquals(0, ctx.pendingDeliveryCount());
        List<BatchItem> restored = ctx.sortedItems();
        assertEquals(List.of(1L, 2L), restored.stream().map(BatchItem::requestId).toList());
        assertSame(first, restored.get(0));
        assertSame(second, restored.get(1));
    }

    @Test
    void routeCallbackFailureRestoresToReadyBacklog_andRemovalAndShutdownDoNotLeak() {
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem first = routeItem(1, 50, 100);
        BatchItem second = routeItem(2, 50, 200);
        queue.add(first);
        queue.add(second);
        AtomicInteger depth = new AtomicInteger(2);
        BatcherContext ctx = context(endpoint, queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                throw new IllegalStateException("route callback failure");
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }

            @Override
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });

        assertThrows(IllegalStateException.class, () -> ctx.stageDecisionGroup(
                List.of(first, second), new DecisionGroupMetadata("batch_full", 0)));

        assertTrue(ctx.isActiveEmpty());
        assertEquals(2, ctx.readyDeliveryCount());
        assertEquals(2, depth.get());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertEquals(List.of(1L, 2L), ctx.sortedQueuedItems().stream()
                .map(BatchItem::requestId).toList());

        // Lease timeout / preemption can still revoke an already-ready item.
        assertTrue(ctx.remove(first));
        assertEquals(1, depth.get());
        assertEquals(1, ctx.readyDeliveryCount());

        // Shutdown owns and drains the final ready member exactly once.
        List<BatchItem> drained = new java.util.ArrayList<>();
        ctx.stopAndDrainTo(drained);
        assertEquals(List.of(second), drained);
        assertEquals(0, depth.get());
        assertEquals(0, ctx.readyDeliveryCount());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedQueuedItems().isEmpty());
    }

    @Test
    void nonBatchRouteDoesNotApplyWorkerBatchTokenCapacity() {
        SchedulingTestConfig.useNonBatchDispatcher(config);
        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(2);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.availableRequestSlots(0)).thenReturn(1);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = routeItem(1, 50, System.currentTimeMillis());
        queue.add(item);
        AtomicReference<List<BatchItem>> delivered = new AtomicReference<>();
        BatcherContext ctx = context(endpoint, queue, new AtomicInteger(1), new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                delivered.set(items);
            }

            @Override
            public void onOfferFailure(BatchItem failed, Throwable error) {
                throw new AssertionError("route-only delivery must not use batch token capacity", error);
            }

            @Override
            public void onDeliveryFailure(BatchItem failed, Throwable error) {
                throw new AssertionError("route-only delivery must complete", error);
            }
        });

        new ImmediateNonBatchAlgorithm().processQueue(ctx);

        assertEquals(List.of(item), delivered.get());
        assertEquals(0, ctx.size());
    }

    @Test
    void readyBacklogRemainsVisibleAndRemovableThroughQueueManager() throws Exception {
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        AtomicInteger deliveryCalls = new AtomicInteger();
        AtomicReference<BatchItem> shutdownFailure = new AtomicReference<>();
        WorkerBatcher batcher = new WorkerBatcher(
                "ready-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override
                    public void onExpired(BatchItem head) {
                    }

                    @Override
                    public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                        deliveryCalls.incrementAndGet();
                    }

                    @Override
                    public void onOfferFailure(BatchItem item, Throwable error) {
                        shutdownFailure.set(item);
                    }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                    }
                }, mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(routeItem(1, 70, 100)));
        assertTrue(batcher.tryOffer(routeItem(2, 50, 200)));
        long offeredVersion = batcher.queueVersion();
        batcher.start();
        try {
            awaitTrue(() -> batcher.queueVersion() > offeredVersion);

            // Both requests have left the active decision queue and are held
            // behind the request cap, yet remain actionable eviction victims.
            assertEquals(List.of(1L, 2L), batcher.queueManager().snapshot().items().stream()
                    .map(item -> item.requestId()).toList());
            assertEquals(Map.of(70, 1, 50, 1), batcher.queueSizeByPriority());
            assertEquals(2, batcher.queueManager().estimateWaitMs(100, 99),
                    "NON_BATCH wait accounts for each pending request independently");
            assertEquals(0, deliveryCalls.get());

            batcher.queueManager().tryRemove(1L, "ready-lease-timeout");
            assertEquals(1, batcher.queueSize());
            assertEquals(List.of(2L), batcher.queueManager().snapshot().items().stream()
                    .map(item -> item.requestId()).toList());
        } finally {
            batcher.shutdown();
        }

        assertEquals(2L, shutdownFailure.get().requestId());
        assertEquals(0, batcher.queueSize());
        assertTrue(batcher.queueManager().snapshot().items().isEmpty());
    }

    @Test
    void priorityQueueEmptyWorkerWaitsOnConditionAndEnqueueWakesIt() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(0);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        CountDownLatch delivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "condition-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                        delivered.countDown();
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            TimeUnit.MILLISECONDS.sleep(30);
            assertTrue(batcher.isWaitingForSignal(),
                    "an empty AutoTPM worker must block, not wake on a 1ms poll");

            assertTrue(batcher.tryOffer(item(1, 50, System.currentTimeMillis())));
            assertTrue(delivered.await(2, TimeUnit.SECONDS));
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void routeSlotSignalWakesReadyOnlyWorkerWithoutPolling() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        AtomicInteger slots = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenAnswer(ignored -> slots.get());
        CountDownLatch delivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "slot-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                        delivered.countDown();
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(routeItem(1, 50, System.currentTimeMillis())));
        long offeredVersion = batcher.queueVersion();
        batcher.start();
        try {
            awaitTrue(() -> batcher.queueVersion() > offeredVersion
                    && batcher.isWaitingForSignal());
            assertEquals(1, batcher.queueSize());
            assertEquals(1, delivered.getCount());

            slots.set(1);
            batcher.signalDeliveryCapacityAvailable();
            assertTrue(delivered.await(2, TimeUnit.SECONDS));
            awaitTrue(() -> batcher.queueSize() == 0);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void fullRouteCapDoesNotHeadOfLineBlockLegacyBatchWork() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightBatchesPerPrefillWorker(0);
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        AtomicInteger routeDeliveries = new AtomicInteger();
        CountDownLatch batchDelivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "mixed-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                        if (items.get(0).deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
                            batchDelivered.countDown();
                        } else {
                            routeDeliveries.incrementAndGet();
                        }
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(routeItem(1, 70, System.currentTimeMillis())));
        long routeOfferVersion = batcher.queueVersion();
        batcher.start();
        try {
            awaitTrue(() -> batcher.queueVersion() > routeOfferVersion
                    && batcher.isWaitingForSignal());

            assertTrue(batcher.tryOffer(item(2, 50, System.currentTimeMillis())));
            assertTrue(batchDelivered.await(2, TimeUnit.SECONDS),
                    "BATCH_ENQUEUE work must pass a capacity-blocked route backlog");
            assertEquals(0, routeDeliveries.get());
            assertEquals(1, batcher.queueSize(),
                    "only the capacity-blocked route request remains charged");
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void successfulLegacyCallbackConsumesDistinctItemsSharingRequestId() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem first = item(0, 50, 100);
        BatchItem second = item(0, 50, 200);
        queue.add(first);
        queue.add(second);
        AtomicInteger depth = new AtomicInteger(2);
        AtomicInteger callbackMembers = new AtomicInteger();
        BatcherContext ctx = context(queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                callbackMembers.set(items.size());
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }

            @Override
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });

        ctx.stageForDelivery(List.of(first, second), new DecisionGroupMetadata("test", 0));

        assertEquals(2, callbackMembers.get());
        assertEquals(0, depth.get(),
                "a successful legacy callback consumes unclaimed staged members");
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void claimedDeliveryCompletesOnce_andFinallyCannotRequeueIt() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = item(7, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger dispatchCalls = new AtomicInteger();
        AtomicReference<BatcherContext> owner = new AtomicReference<>();
        BatcherContext ctx = context(queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                dispatchCalls.incrementAndGet();
                long stagedVersion = owner.get().queueVersionValue();
                assertEquals(BatcherContext.PendingClaimResult.CLAIMED,
                        owner.get().claimPendingDelivery(item));
                assertEquals(stagedVersion + 1, owner.get().queueVersionValue(),
                        "queue-to-delivery ownership must invalidate versioned plans");
                long claimedVersion = owner.get().queueVersionValue();
                assertTrue(owner.get().completePendingDelivery(item));
                assertEquals(claimedVersion + 1, owner.get().queueVersionValue(),
                        "releasing charged capacity must invalidate versioned offers");
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }

            @Override
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });
        owner.set(ctx);

        ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0));
        ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0));

        assertEquals(1, dispatchCalls.get(), "a claimed member must not be dispatched twice");
        assertEquals(0, depth.get());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void claimedCallbackFailure_usesDeliveryFailureWithoutPendingLeakOrRequeue() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = item(8, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger offerFailures = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        AtomicReference<BatcherContext> owner = new AtomicReference<>();
        BatcherContext ctx = context(queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                assertEquals(BatcherContext.PendingClaimResult.CLAIMED,
                        owner.get().claimPendingDelivery(item));
                throw new IllegalStateException("failed after claim");
            }

            @Override
            public void onOfferFailure(BatchItem failed, Throwable error) {
                offerFailures.incrementAndGet();
            }

            @Override
            public void onDeliveryFailure(BatchItem failed, Throwable error) {
                assertSame(item, failed);
                assertEquals("failed after claim", error.getMessage());
                deliveryFailures.incrementAndGet();
            }
        });
        owner.set(ctx);

        assertThrows(IllegalStateException.class,
                () -> ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0)));
        assertEquals(0, offerFailures.get());
        assertEquals(1, deliveryFailures.get());
        assertEquals(0, depth.get());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void shutdownDrainWinsStagedItemExactlyOnce() throws Exception {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = item(9, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger offerFailures = new AtomicInteger();
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch callbackMayReturn = new CountDownLatch(1);
        BatcherContext ctx = context(queue, depth, new DecisionGroupHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                callbackEntered.countDown();
                try {
                    assertTrue(callbackMayReturn.await(2, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                }
            }

            @Override
            public void onOfferFailure(BatchItem failed, Throwable error) {
                assertSame(item, failed);
                offerFailures.incrementAndGet();
            }

            @Override
            public void onDeliveryFailure(BatchItem failed, Throwable error) {
                throw new AssertionError("shutdown-drained item must not reach delivery failure");
            }
        });

        CompletableFuture<Void> dispatch = CompletableFuture.runAsync(() ->
                ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0)));
        assertTrue(callbackEntered.await(2, TimeUnit.SECONDS));
        long stagedVersion = ctx.queueVersionValue();
        List<BatchItem> drained = new java.util.ArrayList<>();
        ctx.stopAndDrainTo(drained);
        assertEquals(List.of(item), drained);
        assertEquals(stagedVersion + 1, ctx.queueVersionValue(),
                "shutdown releasing a staged capacity slot must invalidate versioned offers");
        callbackMayReturn.countDown();
        dispatch.get(2, TimeUnit.SECONDS);

        assertEquals(0, depth.get());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedItems().isEmpty());
        assertEquals(0, offerFailures.get(),
                "shutdown owns the drained item; callback finally must not deliver it twice");
    }

    // ==================== helpers ====================

    private static BatchItem item(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext ctx = newContext(requestId, priority);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    /** Missing scheduling metadata preserves the untrusted priority-zero sentinel. */
    private static BatchItem legacyItem(long requestId, long enqueuedAtMs) {
        return new BatchItem(newContext(requestId, 0), new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BatchItem routeItem(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext ctx = newContext(requestId, priority);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        SchedulingTestConfig.useNonBatchDispatcher(ctx.getConfig());
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BalanceContext newContext(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }

    private BatcherContext context(PriorityBlockingQueue<BatchItem> queue,
                                   AtomicInteger depth,
                                   DecisionGroupHandler handler) {
        return context(null, queue, depth, handler);
    }

    private BatcherContext context(PrefillEndpoint endpoint,
                                   PriorityBlockingQueue<BatchItem> queue,
                                   AtomicInteger depth,
                                   DecisionGroupHandler handler) {
        return new BatcherContext("test-worker", endpoint, config, handler, queue, depth,
                new AtomicLong(), new ReentrantLock(), WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
    }

    private static void awaitTrue(BooleanSupplier condition) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (!condition.getAsBoolean() && System.nanoTime() < deadlineNanos) {
            TimeUnit.MILLISECONDS.sleep(5);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true within 2 seconds");
    }
}
