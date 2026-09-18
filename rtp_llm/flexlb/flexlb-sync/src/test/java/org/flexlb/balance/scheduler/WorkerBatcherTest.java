package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

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
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
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
    void enqueuePreservesQueueWaitReasonUntilQueueBecomesEmpty() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerBatcher batcher = new WorkerBatcher("test", endpoint, config,
                mock(DecisionGroupHandler.class), mock(BatchSchedulerReporter.class));
        assertNull(batcher.tryOffer(item(1, 50, System.currentTimeMillis())));
        batcher.setWaitReason("decode engine slots exhausted");
        assertNull(batcher.tryOffer(item(2, 70, System.currentTimeMillis())));
        assertEquals("decode engine slots exhausted", batcher.getWaitReason());
        org.mockito.Mockito.verifyNoInteractions(endpoint);

        batcher.tryRemove(List.of(1L, 2L), "test drained");
        assertNull(batcher.tryOffer(item(3, 50, System.currentTimeMillis())));
        assertNull(batcher.getWaitReason(), "a new queue episode must not inherit the previous wait");
    }

    @ParameterizedTest
    @MethodSource("fullQueueCases")
    void fullOfferUsesMaintainedPriorityCounts(int[] priorities, int capacity,
                                               StrategyErrorType code, AdmissionRejectReason reason) {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(priorities.length);
        WorkerBatcher batcher = newBatcher();
        BatchItem[] occupants = new BatchItem[priorities.length];
        for (int i = 0; i < priorities.length; i++) {
            occupants[i] = org.mockito.Mockito.spy(priorities[i] == 0
                    ? legacyItem(i + 1, 100) : item(i + 1, priorities[i], 100));
            assertNull(batcher.tryOffer(occupants[i]));
        }
        config.batchDispatcher().setMaxWaitingRequestsPerPrefillWorker(capacity);
        org.mockito.Mockito.clearInvocations((Object[]) occupants);
        BatchItem incoming = item(1000, 50, 200);
        for (int attempt = 0; attempt < 100; attempt++) {
            Response failure = batcher.tryOffer(incoming);
            assertEquals(code.getErrorCode(), failure.getCode());
            assertEquals(reason, failure.getAdmissionRejectReason());
        }
        org.mockito.Mockito.verifyNoInteractions((Object[]) occupants);
        assertEquals(priorities.length, batcher.queueSize());
    }

    static Stream<Arguments> fullQueueCases() {
        return Stream.of(
                Arguments.of(new int[]{70}, 1, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                        AdmissionRejectReason.HIGHER_PRIORITY_AHEAD),
                Arguments.of(new int[]{50}, 1, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                        AdmissionRejectReason.SAME_PRIORITY_AHEAD),
                Arguments.of(new int[]{30}, 1, StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED),
                Arguments.of(new int[]{0}, 1, StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED),
                Arguments.of(new int[]{70, 0}, 2, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                        AdmissionRejectReason.HIGHER_PRIORITY_AHEAD),
                Arguments.of(new int[]{70, 0}, 1, StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED),
                Arguments.of(new int[]{30, 0}, 2, StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED));
    }

    @Test
    void repeatedFullQueueRejectionDoesNotTraverseOrCopyRequests() {
        PriorityBlockingQueue<BatchItem> queue = org.mockito.Mockito.spy(new PriorityBlockingQueue<BatchItem>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER));
        queue.add(item(1, 70, 100));
        BatcherContext ctx = context(queue, new AtomicInteger(1), mock(DecisionGroupHandler.class));
        org.mockito.Mockito.clearInvocations(queue);

        ctx.queueLock().lock();
        try {
            for (int attempt = 0; attempt < 100; attempt++) {
                assertEquals(AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
                        ctx.queueFullResponse(50, 1).getAdmissionRejectReason());
                assertEquals(1, ctx.queueDiagnostics(50).get("higherPriorityCount"));
            }
        } finally {
            ctx.queueLock().unlock();
        }

        org.mockito.Mockito.verifyNoInteractions(queue);
    }

    @Test
    void removingOccupantsDoesNotLeaveStalePriorityCounts() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);
        WorkerBatcher batcher = newBatcher();
        for (int priority : new int[]{70, 50, 30}) {
            assertNull(batcher.tryOffer(item(priority, priority, 100)));
            assertEquals(1, batcher.tryRemove(List.of((long) priority), "removed").size());
        }
        assertNull(batcher.tryOffer(legacyItem(1, 100)));
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(),
                batcher.tryOffer(item(2, 50, 200)).getCode());
    }

    @Test
    void fullOfferCapturesCauseBeforeQueueChanges() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertNull(batcher.tryOffer(item(1, 70, now)));

        BatchItem incoming = item(2, 50, now);
        Response failure = batcher.tryOffer(incoming);
        batcher.tryRemove(List.of(1L), "test capacity released");

        assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(), failure.getCode());
        assertEquals(AdmissionRejectReason.HIGHER_PRIORITY_AHEAD, failure.getAdmissionRejectReason());
        assertEquals(0, batcher.queueSize());
        Map<String, Object> diagnostics = incoming.ctx().getSchedulingDiagnostics();
        assertEquals("prefill queue capacity exhausted", diagnostics.get("cause"));
        Map<?, ?> prefill = (Map<?, ?>) diagnostics.get("prefill");
        assertEquals(1, prefill.get("queueDepth"), "PV must retain the failure-time occupancy");
        assertEquals(1, prefill.get("higherPriorityCount"));
        assertEquals(1, prefill.get("aheadCountUpperBound"));
        assertThrows(UnsupportedOperationException.class, () -> diagnostics.clear());
        assertThrows(UnsupportedOperationException.class, () -> prefill.clear());
        assertNull(batcher.tryOffer(incoming));
        assertNull(incoming.ctx().getSchedulingDiagnostics(), "successful retry clears the rejected offer");
    }

    @Test
    void samePriorityOccupancyIsNotReportedAsAnExactQueuePosition() {
        WorkerBatcher batcher = newBatcher();
        BatchItem head = item(1, 50, 100);
        assertNull(batcher.tryOffer(head));
        assertNull(batcher.tryOffer(item(2, 50, 200)));
        batcher.recordFailureDiagnostics(head, "decode engine slots exhausted");
        Map<?, ?> prefill = (Map<?, ?>) head.ctx().getSchedulingDiagnostics().get("prefill");
        assertEquals(2, prefill.get("samePriorityCount"));
        assertEquals(2, prefill.get("aheadCountUpperBound"));
        assertFalse(prefill.containsKey("aheadCount"), "later arrivals are not known predecessors");
    }

    @Test
    void fullOfferIncludesChargedPendingDeliveryButDoesNotMakeItEvictable() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);
        BatchItem occupant = item(1, 70, System.currentTimeMillis());
        BatchItem incoming = item(2, 50, System.currentTimeMillis());
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        queue.add(occupant);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatcherContext ctx = context(queue, new AtomicInteger(1), handler);
        org.mockito.Mockito.doAnswer(invocation -> {
            assertTrue(ctx.sortedQueuedItems().isEmpty(), "pending member is not an eviction candidate");
            assertEquals(1, ctx.size(), "pending member still owns its capacity slot");
            Map<String, Object> diagnostics = ctx.queueDiagnostics(incoming.priority());
            assertEquals(1, diagnostics.get("queueDepth"));
            assertEquals(1, diagnostics.get("pendingCount"));
            assertEquals(1, diagnostics.get("higherPriorityCount"));
            Response failure = ctx.queueFullResponse(incoming.priority(), 1);
            assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(), failure.getCode());
            assertEquals(AdmissionRejectReason.HIGHER_PRIORITY_AHEAD, failure.getAdmissionRejectReason());
            return null;
        }).when(handler).onDecisionGroupReady(org.mockito.ArgumentMatchers.anyList(),
                org.mockito.ArgumentMatchers.any());

        ctx.stageForDelivery(List.of(occupant), new DecisionGroupMetadata("test", 0));

        assertEquals(0, ctx.size());
        assertOccupiedPriorities(ctx, Map.of());
    }

    @Test
    void stoppedOfferIsNotACapacityOrVictimConflict() {
        WorkerBatcher batcher = newBatcher();
        batcher.shutdown();
        Response failure = batcher.tryOffer(item(2, 50, System.currentTimeMillis()));
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), failure.getCode());
        assertFalse(StrategyErrorType.fromErrorCode(failure.getCode()).isCapacityRejection());
        PrefillQueueManager.ReplaceOutcome replacement = batcher.tryReplaceVictimsPresent(
                List.of(1L), item(3, 70, System.currentTimeMillis()));
        assertFalse(replacement.isVictimGone());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), replacement.failure().getCode());
        assertTrue(replacement.removed().isEmpty());
    }

    @Test
    void replacementExceptionRetainsRemovedVictimsForSettlement() {
        WorkerBatcher batcher = newBatcher();
        BatchItem victim = item(1, 30, System.currentTimeMillis());
        assertNull(batcher.tryOffer(victim));

        PrefillQueueManager.ReplaceOutcome result = batcher.tryReplaceVictimsPresent(List.of(1L), null);

        assertTrue(result.isPartialFailure());
        assertEquals(List.of(victim), result.removed());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), result.failure().getCode());
        assertEquals(0, batcher.queueSize());
    }

    @Test
    void queue_size_by_priority_buckets_multiple_priorities() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertNull(batcher.tryOffer(item(1, 70, now)));
        assertNull(batcher.tryOffer(item(2, 50, now)));
        assertNull(batcher.tryOffer(item(3, 50, now)));
        assertNull(batcher.tryOffer(item(4, 30, now)));

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

        assertNull(batcher.tryOffer(legacyItem(1, now)));
        assertNull(batcher.tryOffer(legacyItem(2, now)));

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

        assertNull(batcher.tryOffer(item(1, 70, now)));
        assertNull(batcher.tryOffer(item(2, 50, now)));

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
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });

        assertThrows(IllegalStateException.class,
                () -> ctx.stageForDelivery(List.of(first, second), new DecisionGroupMetadata("test", 0)));

        assertEquals(2, depth.get());
        assertOccupiedPriorities(ctx, Map.of(50, 2));
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
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });

        assertThrows(IllegalStateException.class, () -> ctx.stageDecisionGroup(
                List.of(first, second), new DecisionGroupMetadata("batch_full", 0)));

        assertTrue(ctx.isActiveEmpty());
        assertEquals(2, ctx.readyDeliveryCount());
        assertEquals(2, ctx.queueDiagnostics(70).get("aheadCountUpperBound"),
                "already-decided lower-priority requests may be ahead of the active queue");
        assertEquals(2, depth.get());
        assertOccupiedPriorities(ctx, Map.of(50, 2));
        assertEquals(0, ctx.pendingDeliveryCount());
        assertEquals(List.of(1L, 2L), ctx.sortedQueuedItems().stream()
                .map(BatchItem::requestId).toList());

        // Lease timeout / preemption can still revoke an already-ready item.
        assertTrue(ctx.remove(first));
        assertEquals(1, depth.get());
        assertEquals(1, ctx.readyDeliveryCount());
        assertOccupiedPriorities(ctx, Map.of(50, 1));

        // Shutdown owns and drains the final ready member exactly once.
        List<BatchItem> drained = new java.util.ArrayList<>();
        ctx.stopAndDrainTo(drained);
        assertEquals(List.of(second), drained);
        assertEquals(0, depth.get());
        assertOccupiedPriorities(ctx, Map.of());
        assertEquals(0, ctx.readyDeliveryCount());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedQueuedItems().isEmpty());
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
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        shutdownFailure.set(item);
                    }
                }, mock(BatchSchedulerReporter.class));

        assertNull(batcher.tryOffer(routeItem(1, 70, 100)));
        assertNull(batcher.tryOffer(routeItem(2, 50, 200)));
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

                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            TimeUnit.MILLISECONDS.sleep(30);
            assertTrue(batcher.isWaitingForSignal(),
                    "an empty AutoTPM worker must block, not wake on a 1ms poll");

            assertNull(batcher.tryOffer(item(1, 50, System.currentTimeMillis())));
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

                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        assertNull(batcher.tryOffer(routeItem(1, 50, System.currentTimeMillis())));
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

                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));

        assertNull(batcher.tryOffer(routeItem(1, 70, System.currentTimeMillis())));
        long routeOfferVersion = batcher.queueVersion();
        batcher.start();
        try {
            awaitTrue(() -> batcher.queueVersion() > routeOfferVersion
                    && batcher.isWaitingForSignal());

            assertNull(batcher.tryOffer(item(2, 50, System.currentTimeMillis())));
            assertTrue(batchDelivered.await(2, TimeUnit.SECONDS),
                    "BATCH_ENQUEUE work must pass a capacity-blocked route backlog");
            assertEquals(0, routeDeliveries.get());
            // The callback signals before deliverStaged's finally releases its queue slot.
            awaitTrue(() -> batcher.queueSize() == 1);
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
            public void onDeliveryFailure(BatchItem item, Throwable error) {
            }
        });
        owner.set(ctx);

        ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0));
        ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0));

        assertEquals(1, dispatchCalls.get(), "a claimed member must not be dispatched twice");
        assertEquals(0, depth.get());
        assertOccupiedPriorities(ctx, Map.of());
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
            public void onDeliveryFailure(BatchItem failed, Throwable error) {
                assertSame(item, failed);
                assertEquals("failed after claim", error.getMessage());
                deliveryFailures.incrementAndGet();
            }
        });
        owner.set(ctx);

        assertThrows(IllegalStateException.class,
                () -> ctx.stageForDelivery(List.of(item), new DecisionGroupMetadata("test", 0)));
        assertEquals(1, deliveryFailures.get());
        assertEquals(0, depth.get());
        assertOccupiedPriorities(ctx, Map.of());
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
        assertOccupiedPriorities(ctx, Map.of());
        assertEquals(0, ctx.pendingDeliveryCount());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    private static void assertOccupiedPriorities(BatcherContext ctx, Map<Integer, Integer> expected) {
        int[] counts = (int[]) org.springframework.test.util.ReflectionTestUtils.getField(
                ctx, "occupiedSlotsByPriority");
        Map<Integer, Integer> actual = new java.util.HashMap<>();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] != 0) {
                actual.put(priority, counts[priority]);
            }
        }
        assertEquals(expected, actual, "capacity attribution must follow ownership through delivery and cleanup");
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
