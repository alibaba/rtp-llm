package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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
    void singleDecisionWithBatchDispatcherSendsSingletonGroups() throws Exception {
        assertDecisionAndDispatcherCombination(true, true, List.of(1, 1), "single_request");
    }

    @Test
    void singleDecisionWithNonBatchDispatcherSendsSingletonGroups() throws Exception {
        assertDecisionAndDispatcherCombination(true, false, List.of(1, 1), "single_request");
    }

    @Test
    void fixedWindowWithBatchDispatcherSendsOneFullGroup() throws Exception {
        assertDecisionAndDispatcherCombination(false, true, List.of(2), "batch_full");
    }

    @Test
    void fixedWindowWithNonBatchDispatcherSendsOneFullGroup() throws Exception {
        assertDecisionAndDispatcherCombination(false, false, List.of(2), "batch_full");
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
    void decodeBlockedReadyRouteDoesNotStarveActiveBatchWork() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        CountDownLatch batchDelivered = new CountDownLatch(1);
        AtomicInteger routeAttempts = new AtomicInteger();
        AtomicReference<String> callbackFailure = new AtomicReference<>();
        AtomicReference<WorkerBatcher> owner = new AtomicReference<>();
        WorkerBatcher batcher = new WorkerBatcher(
                "ready-fairness-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        BatchItem delivered = items.get(0);
                        if (delivered.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
                            routeAttempts.incrementAndGet();
                            BatcherContext.PendingClaimResult claim =
                                    owner.get().claimPendingDelivery(delivered);
                            if (claim != BatcherContext.PendingClaimResult.CLAIMED) {
                                callbackFailure.compareAndSet(null,
                                        "unexpected claim result: " + claim);
                                return;
                            }
                            BatcherContext.PendingRestoreResult restore =
                                    owner.get().restorePendingDelivery(delivered);
                            if (restore != BatcherContext.PendingRestoreResult.RESTORED) {
                                callbackFailure.compareAndSet(null,
                                        "unexpected restore result: " + restore);
                            }
                        } else {
                            batchDelivered.countDown();
                        }
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));
        owner.set(batcher);

        assertTrue(batcher.tryOffer(routeItem(
                1, 100, System.currentTimeMillis())));
        assertTrue(batcher.tryOffer(item(
                2, 1, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(batchDelivered.await(2, TimeUnit.SECONDS),
                    "a restored ready route must not starve unrelated active batch work");
            assertNull(callbackFailure.get(), callbackFailure::get);
            assertEquals(1, routeAttempts.get(),
                    "active batch work must bypass the deferred ready-route retry");
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void decodeBlockedReadyRouteRetriesAtBoundedRate() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        CountDownLatch firstAttempt = new CountDownLatch(1);
        CountDownLatch deliverySucceeded = new CountDownLatch(1);
        AtomicBoolean allowDelivery = new AtomicBoolean();
        AtomicInteger attempts = new AtomicInteger();
        AtomicReference<String> callbackFailure = new AtomicReference<>();
        AtomicReference<WorkerBatcher> owner = new AtomicReference<>();
        WorkerBatcher batcher = new WorkerBatcher(
                "ready-retry-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        BatchItem delivered = items.get(0);
                        attempts.incrementAndGet();
                        firstAttempt.countDown();
                        BatcherContext.PendingClaimResult claim =
                                owner.get().claimPendingDelivery(delivered);
                        if (claim != BatcherContext.PendingClaimResult.CLAIMED) {
                            callbackFailure.compareAndSet(null,
                                    "unexpected claim result: " + claim);
                            return;
                        }
                        if (allowDelivery.get()) {
                            if (!owner.get().completePendingDelivery(delivered)) {
                                callbackFailure.compareAndSet(null,
                                        "successful retry did not complete pending ownership");
                            }
                            deliverySucceeded.countDown();
                            return;
                        }
                        BatcherContext.PendingRestoreResult restore =
                                owner.get().restorePendingDelivery(delivered);
                        if (restore != BatcherContext.PendingRestoreResult.RESTORED) {
                            callbackFailure.compareAndSet(null,
                                    "unexpected restore result: " + restore);
                        }
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));
        owner.set(batcher);

        assertTrue(batcher.tryOffer(routeItem(
                1, 100, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(firstAttempt.await(2, TimeUnit.SECONDS));
            TimeUnit.MILLISECONDS.sleep(100);
            assertTrue(attempts.get() >= 2,
                    "a restored hard claim must remain retryable");
            assertTrue(attempts.get() <= 15,
                    "10ms hard-claim backoff must bound retries in a 100ms interval, attempts="
                            + attempts.get());
            allowDelivery.set(true);
            assertTrue(deliverySucceeded.await(2, TimeUnit.SECONDS),
                    "a transient hard-claim failure must eventually deliver");
            awaitTrue(() -> batcher.queueSize() == 0);
            assertNull(callbackFailure.get(), callbackFailure::get);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void callbackFailureRetriesAtBoundedRate() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        CountDownLatch firstAttempt = new CountDownLatch(1);
        CountDownLatch successfulCallback = new CountDownLatch(1);
        AtomicBoolean allowCallback = new AtomicBoolean();
        AtomicInteger attempts = new AtomicInteger();
        WorkerBatcher batcher = new WorkerBatcher(
                "callback-retry-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        attempts.incrementAndGet();
                        firstAttempt.countDown();
                        if (allowCallback.get()) {
                            successfulCallback.countDown();
                            return;
                        }
                        throw new IllegalStateException("synthetic callback failure");
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(item(
                1, 100, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(firstAttempt.await(2, TimeUnit.SECONDS));
            TimeUnit.MILLISECONDS.sleep(100);
            assertTrue(attempts.get() >= 2,
                    "a restored callback failure must remain retryable");
            assertTrue(attempts.get() <= 15,
                    "10ms callback-failure backoff must bound retries in a 100ms interval, attempts="
                            + attempts.get());
            assertEquals(1, batcher.queueSize(),
                    "failed callback must retain exactly one charged queue owner before recovery");
            allowCallback.set(true);
            assertTrue(successfulCallback.await(2, TimeUnit.SECONDS),
                    "a transient callback failure must eventually deliver");
            awaitTrue(() -> batcher.queueSize() == 0);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void partialVictimReplacementWakesNewlyExposedActiveHead() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        CountDownLatch batchDelivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "replace-wake-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        if (items.get(0).deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
                            batchDelivered.countDown();
                        }
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        BatchItem blockedRoute = routeItem(1, 100, System.currentTimeMillis());
        assertTrue(batcher.tryOffer(blockedRoute));
        assertTrue(batcher.tryOffer(item(2, 1, System.currentTimeMillis())));
        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            PrefillQueueManager.ReplaceOutcome outcome =
                    batcher.queueManager().tryReplaceVictimsPresent(
                            List.of(1L, 1L), item(3, 50, System.currentTimeMillis()));
            assertTrue(outcome.isPartialFailure());
            assertEquals(List.of(blockedRoute), outcome.removed());
            assertTrue(batchDelivered.await(2, TimeUnit.SECONDS),
                    "removing a blocked route head must wake the exposed batch head");
        } finally {
            batcher.shutdown();
        }
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
    void routeSlotSignalLetsSingleDecisionSendAfterItStayedActive() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        AtomicInteger slots = new AtomicInteger();
        AtomicInteger slotChecks = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenAnswer(ignored -> {
            slotChecks.incrementAndGet();
            return slots.get();
        });
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
            awaitTrue(batcher::isWaitingForSignal);
            int checksAtWait = slotChecks.get();
            TimeUnit.MILLISECONDS.sleep(30);
            assertTrue(batcher.isWaitingForSignal());
            assertEquals(checksAtWait, slotChecks.get(),
                    "capacity-blocked active route work must not poll every millisecond");
            assertEquals(offeredVersion, batcher.queueVersion(),
                    "capacity-blocked SINGLE work must remain in the active queue");
            assertEquals(1, batcher.queueSize());
            assertEquals(List.of(1L), batcher.queueManager().snapshot().items().stream()
                    .map(item -> item.requestId()).toList());
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
    void routeHeadExpirationBoundsCapacityConditionWait() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        CountDownLatch expired = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "expiration-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { expired.countDown(); }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) { }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(routeItem(1, 50, now, now + 200)));
        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            assertTrue(expired.await(2, TimeUnit.SECONDS),
                    "absolute expiration must wake a route-capacity condition wait");
            awaitTrue(() -> batcher.queueSize() == 0);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void cancelingCapacityBlockedRouteHeadWakesBatchWorkBehindIt() throws Exception {
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        CountDownLatch delivered = new CountDownLatch(1);
        AtomicLong deliveredRequestId = new AtomicLong();
        WorkerBatcher batcher = new WorkerBatcher(
                "mixed-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        deliveredRequestId.set(items.get(0).requestId());
                        delivered.countDown();
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(routeItem(1, 50, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 1)));
        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            assertEquals(List.of(1L), batcher.tryRemove(List.of(1L), "test-cancel")
                    .stream().map(BatchItem::requestId).toList());
            assertTrue(delivered.await(2, TimeUnit.SECONDS),
                    "removing a blocked route head must wake processable batch work");
            assertEquals(2L, deliveredRequestId.get());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void oversizedRouteHeadIsRejectedBeforeCapacityWait() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenReturn(0);
        CountDownLatch failed = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        WorkerBatcher batcher = new WorkerBatcher(
                "oversized-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) { }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) {
                        failure.set(error);
                        failed.countDown();
                    }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        BatchItem oversized = routeItem(1, 50, System.currentTimeMillis());
        oversized.ctx().getRequest().setSeqLen(
                config.getInternalRuntime().getFallbackBatchTokenCapacity());
        assertTrue(batcher.tryOffer(oversized));
        batcher.start();
        try {
            assertTrue(failed.await(2, TimeUnit.SECONDS),
                    "permanently oversized route work must not sleep behind a full route cap");
            assertTrue(failure.get() instanceof BatchTokenCapacityExceededException);
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void atomicRouteStageUsesOneCapacitySnapshot() {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        AtomicInteger slotChecks = new AtomicInteger();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(1)).thenAnswer(ignored ->
                slotChecks.incrementAndGet() == 1 ? 1 : 0);
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = routeItem(1, 50, System.currentTimeMillis());
        queue.add(item);
        AtomicInteger delivered = new AtomicInteger();
        BatcherContext ctx = context(endpoint, queue, new AtomicInteger(1),
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        delivered.addAndGet(items.size());
                    }
                    @Override public void onOfferFailure(BatchItem failed, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem failed, Throwable error) { }
                });

        assertTrue(ctx.stageDecisionGroupIfVersion(
                List.of(item), new DecisionGroupMetadata("single_request", 0),
                ctx.queueVersionValue(), null, 0L));
        assertEquals(1, slotChecks.get());
        assertEquals(1, delivered.get());
        assertEquals(0, ctx.size());
        assertEquals(0, ctx.readyDeliveryCount());
    }

    @Test
    void dropHeadReportsExpirationOnlyWhenRemovalWins() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        BatchItem item = item(1, 50, System.currentTimeMillis());
        queue.add(item);
        AtomicInteger expired = new AtomicInteger();
        BatcherContext ctx = context(queue, new AtomicInteger(1),
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) {
                        expired.incrementAndGet();
                    }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) { }
                    @Override public void onOfferFailure(BatchItem failed, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem failed, Throwable error) { }
                });

        ctx.dropHead(item);
        ctx.dropHead(item);

        assertEquals(1, expired.get());
        assertEquals(0, ctx.size());
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
        assertEquals(0, ctx.activeSize(),
                "callback-pending ownership must not count as active decision work");
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

    private void assertDecisionAndDispatcherCombination(
            boolean singleDecision,
            boolean batchDispatcher,
            List<Integer> expectedGroupSizes,
            String expectedReason) throws Exception {
        SchedulingTestConfig.useFifoQueue(config);
        if (singleDecision) {
            SchedulingTestConfig.useSingleDecision(config);
        } else {
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        }
        if (batchDispatcher) {
            SchedulingTestConfig.useBatchDispatcher(config);
        } else {
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.availableRequestSlots(0)).thenReturn(2);
        List<Integer> actualGroupSizes = new CopyOnWriteArrayList<>();
        List<String> actualReasons = new CopyOnWriteArrayList<>();
        CountDownLatch delivered = new CountDownLatch(expectedGroupSizes.size());
        WorkerBatcher batcher = new WorkerBatcher(
                "mode-worker", endpoint, config, new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(
                            List<BatchItem> items, DecisionGroupMetadata meta) {
                        actualGroupSizes.add(items.size());
                        actualReasons.add(meta.reason());
                        delivered.countDown();
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        BatchItem first = batchDispatcher
                ? item(1, 50, System.currentTimeMillis())
                : routeItem(1, 50, System.currentTimeMillis());
        BatchItem second = batchDispatcher
                ? item(2, 50, System.currentTimeMillis())
                : routeItem(2, 50, System.currentTimeMillis());
        assertTrue(batcher.tryOffer(first));
        assertTrue(batcher.tryOffer(second));
        batcher.start();
        try {
            assertTrue(delivered.await(2, TimeUnit.SECONDS));
            awaitTrue(() -> batcher.queueSize() == 0);
        } finally {
            batcher.shutdown();
        }

        assertEquals(expectedGroupSizes, actualGroupSizes);
        assertEquals(expectedGroupSizes.stream().map(ignored -> expectedReason).toList(),
                actualReasons);
    }

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
        return routeItem(requestId, priority, enqueuedAtMs, Long.MAX_VALUE);
    }

    private static BatchItem routeItem(long requestId, int priority,
                                       long enqueuedAtMs, long expiresAtMs) {
        BalanceContext ctx = newContext(requestId, priority);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, expiresAtMs));
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
