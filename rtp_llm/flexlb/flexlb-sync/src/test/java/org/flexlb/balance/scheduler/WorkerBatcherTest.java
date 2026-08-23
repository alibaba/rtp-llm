package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
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
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/** Queue ordering and capacity-first ownership tests for {@link WorkerBatcher}. */
class WorkerBatcherTest {

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
    }

    @Test
    void queueSizeByPriorityBucketsActiveRequests() {
        WorkerBatcher batcher = newIdleBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(batchItem(1, 70, now)));
        assertTrue(batcher.tryOffer(batchItem(2, 50, now)));
        assertTrue(batcher.tryOffer(batchItem(3, 50, now)));
        assertTrue(batcher.tryOffer(batchItem(4, 30, now)));

        assertEquals(Map.of(70, 1, 50, 2, 30, 1), batcher.queueSizeByPriority());
        assertEquals(4, batcher.queueSize());
    }

    @Test
    void requestWithoutSchedulingMetadataUsesPriorityZero() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher batcher = newIdleBatcher();
        assertTrue(batcher.tryOffer(itemWithoutSchedulingMetadata(1, 100)));
        assertTrue(batcher.tryOffer(itemWithoutSchedulingMetadata(2, 200)));
        assertEquals(Map.of(0, 2), batcher.queueSizeByPriority());
    }

    @Test
    void removingRequestsUpdatesPrioritySnapshot() {
        WorkerBatcher batcher = newIdleBatcher();
        assertTrue(batcher.tryOffer(batchItem(1, 70, 100)));
        assertTrue(batcher.tryOffer(batchItem(2, 50, 200)));

        assertEquals(1, batcher.tryRemove(List.of(1L), "test").size());
        assertEquals(Map.of(50, 1), batcher.queueSizeByPriority());
        assertEquals(1, batcher.tryRemove(List.of(2L), "test").size());
        assertEquals(Map.of(), batcher.queueSizeByPriority());
    }

    @Test
    void singleBatchDecisionPublishesSingletons() throws Exception {
        assertDecisionAndDeliveryMode(true, true, List.of(1, 1), "single_request");
    }

    @Test
    void singleRouteDecisionPublishesSingletons() throws Exception {
        assertDecisionAndDeliveryMode(true, false, List.of(1, 1), "single_request");
    }

    @Test
    void fixedWindowBatchDecisionPublishesFullGroup() throws Exception {
        assertDecisionAndDeliveryMode(false, true, List.of(2), "batch_full");
    }

    @Test
    void fixedWindowRouteDecisionPublishesFullGroup() throws Exception {
        assertDecisionAndDeliveryMode(false, false, List.of(2), "batch_full");
    }

    @Test
    void callbackFailureIsTerminalAndIsNeverRetried() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        AtomicInteger firstCallbacks = new AtomicInteger();
        AtomicInteger secondCallbacks = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        CountDownLatch firstFailed = new CountDownLatch(1);
        CountDownLatch secondDelivered = new CountDownLatch(1);
        OneBatchSlotCapacityAdmission capacity =
                new OneBatchSlotCapacityAdmission();
        WorkerBatcher batcher = new WorkerBatcher(
                "callback-failure-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        long requestId = group.requests().get(0).requestId();
                        if (requestId == 1L) {
                            firstCallbacks.incrementAndGet();
                            throw new IllegalStateException(
                                    "synthetic callback failure");
                        }
                        assertEquals(2L, requestId);
                        secondCallbacks.incrementAndGet();
                        TestCapacityAdmission.complete(group);
                        secondDelivered.countDown();
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        assertEquals(1L, item.requestId());
                        deliveryFailures.incrementAndGet();
                        firstFailed.countDown();
                    }
                },
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(batchItem(1, 50, System.currentTimeMillis())));
        assertTrue(batcher.tryOffer(batchItem(2, 50, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(firstFailed.await(2, TimeUnit.SECONDS));
            assertTrue(secondDelivered.await(2, TimeUnit.SECONDS),
                    "the worker must continue after terminalizing the failed callback");
            awaitTrue(() -> batcher.queueSize() == 0);
            TimeUnit.MILLISECONDS.sleep(50);
            assertEquals(1, firstCallbacks.get());
            assertEquals(1, secondCallbacks.get());
            assertEquals(1, deliveryFailures.get());
            assertEquals(2, capacity.reservationCount());
            assertEquals(1, capacity.releaseCount(),
                    "the callback failure must release its unregistered batch slot");
            assertEquals(1, capacity.registrationCount(),
                    "only the successful callback transfers a slot to batch lifecycle");
            assertEquals(0, batcher.callbackOwnedRequestCount());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void unexpectedCycleFailureStopsWorkerAndTerminatesEveryActiveRequestOnce()
            throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        IllegalStateException cycleFailure =
                new IllegalStateException("synthetic scheduling invariant failure");
        AtomicInteger schedulingInputReads = new AtomicInteger();
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        org.mockito.Mockito.doAnswer(invocation -> {
            schedulingInputReads.incrementAndGet();
            throw cycleFailure;
        }).when(prefill).getStatus();

        AtomicInteger firstTerminations = new AtomicInteger();
        AtomicInteger secondTerminations = new AtomicInteger();
        AtomicInteger deliveryCallbacks = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        AtomicReference<Throwable> firstTerminationCause = new AtomicReference<>();
        AtomicReference<Throwable> secondTerminationCause = new AtomicReference<>();
        CountDownLatch activeRequestsTerminated = new CountDownLatch(2);
        WorkerBatcher batcher = new WorkerBatcher(
                "cycle-failure-worker",
                prefill,
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        deliveryCallbacks.incrementAndGet();
                    }

                    @Override
                    public void onOfferFailure(BatchItem item, Throwable error) {
                        if (item.requestId() == 1L) {
                            firstTerminations.incrementAndGet();
                            firstTerminationCause.set(error);
                        } else if (item.requestId() == 2L) {
                            secondTerminations.incrementAndGet();
                            secondTerminationCause.set(error);
                        }
                        activeRequestsTerminated.countDown();
                    }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        deliveryFailures.incrementAndGet();
                    }
                },
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(batchItem(1, 50, System.currentTimeMillis())));
        assertTrue(batcher.tryOffer(batchItem(2, 50, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(activeRequestsTerminated.await(2, TimeUnit.SECONDS));
            assertEquals(0, batcher.queueSize());
            assertFalse(batcher.tryOffer(
                    batchItem(3, 50, System.currentTimeMillis())),
                    "an invariant-failed worker must reject new ACTIVE work");

            assertEquals(1, schedulingInputReads.get(),
                    "the failed cycle must not be retried");
            assertEquals(1, firstTerminations.get());
            assertEquals(1, secondTerminations.get());
            assertEquals(0, deliveryCallbacks.get());
            assertEquals(0, deliveryFailures.get());
            assertEquals(0, batcher.callbackOwnedRequestCount());
            assertInstanceOf(IllegalStateException.class,
                    firstTerminationCause.get());
            assertInstanceOf(IllegalStateException.class,
                    secondTerminationCause.get());
            assertSame(cycleFailure, firstTerminationCause.get().getCause());
            assertSame(cycleFailure, secondTerminationCause.get().getCause());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void blockedHeadRemainsActiveAndPreservesOrderedHeadOfLine() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        BatchItem blockedHead = routeItem(1, 100, System.currentTimeMillis());
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blockedHead, false);
        List<Long> delivered = new CopyOnWriteArrayList<>();
        CountDownLatch deliveries = new CountDownLatch(2);
        WorkerBatcher batcher = new WorkerBatcher(
                "ordered-capacity-worker",
                mock(PrefillEndpoint.class),
                config,
                resolvingHandler((group, metadata) -> {
                    delivered.add(group.requests().get(0).requestId());
                    deliveries.countDown();
                }),
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(blockedHead));
        assertTrue(batcher.tryOffer(routeItem(2, 1, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(capacity.awaitBlocked());
            awaitTrue(batcher::isWaitingForSignal);
            assertEquals(2, batcher.queueSize());
            assertTrue(delivered.isEmpty(),
                    "a lower-priority tail cannot bypass a capacity-blocked head");

            capacity.makeAvailable();
            batcher.signalDeliveryCapacityAvailable();
            assertTrue(deliveries.await(2, TimeUnit.SECONDS));
            assertEquals(List.of(1L, 2L), delivered);
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void higherPriorityArrivalSupersedesBlockedHeadThenOriginalOrderResumes()
            throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        BatchItem blocked = routeItem(1, 50, System.currentTimeMillis());
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blocked, false);
        List<Long> delivered = new CopyOnWriteArrayList<>();
        CountDownLatch highDelivered = new CountDownLatch(1);
        CountDownLatch allDelivered = new CountDownLatch(3);
        WorkerBatcher batcher = new WorkerBatcher(
                "priority-capacity-worker",
                mock(PrefillEndpoint.class),
                config,
                resolvingHandler((group, metadata) -> {
                    long requestId = group.requests().get(0).requestId();
                    delivered.add(requestId);
                    if (requestId == 3L) {
                        highDelivered.countDown();
                    }
                    allDelivered.countDown();
                }),
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(blocked));
        assertTrue(batcher.tryOffer(routeItem(2, 10, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(capacity.awaitBlocked());
            assertTrue(batcher.tryOffer(routeItem(3, 100, System.currentTimeMillis())));
            assertTrue(highDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(List.of(3L), delivered);

            capacity.makeAvailable();
            batcher.signalDeliveryCapacityAvailable();
            assertTrue(allDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(List.of(3L, 1L, 2L), delivered);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void capacityReleaseTransfersOneRequestToCallbackOwnershipExactlyOnce()
            throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        BatchItem item = routeItem(1, 50, System.currentTimeMillis());
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(item, false);
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch callbackMayComplete = new CountDownLatch(1);
        CountDownLatch callbackCompleted = new CountDownLatch(1);
        AtomicInteger callbacks = new AtomicInteger();
        AtomicReference<String> failure = new AtomicReference<>();
        WorkerBatcher batcher = new WorkerBatcher(
                "callback-ownership-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        callbacks.incrementAndGet();
                        AdmittedDecisionGroup.AdmittedItem admitted = group.members().get(0);
                        if (!admitted.transferCapacityToEndpointLifecycle()) {
                            failure.set("capacity ownership was lost");
                            callbackEntered.countDown();
                            return;
                        }
                        callbackEntered.countDown();
                        try {
                            if (!callbackMayComplete.await(2, TimeUnit.SECONDS)) {
                                failure.set("callback completion was not released");
                                return;
                            }
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            failure.set("callback was interrupted");
                            return;
                        }
                        if (!admitted.completeDeliveryHandoff()) {
                            failure.set("callback ownership was not resolved");
                        }
                        callbackCompleted.countDown();
                    }

                    @Override public void onOfferFailure(BatchItem request, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem request, Throwable error) {
                        failure.compareAndSet(null, "unexpected delivery failure: " + error);
                    }
                },
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(item));
        batcher.start();
        try {
            assertTrue(capacity.awaitBlocked());
            assertEquals(1, batcher.queueSize());
            capacity.makeAvailable();
            batcher.signalDeliveryCapacityAvailable();
            assertTrue(callbackEntered.await(2, TimeUnit.SECONDS));
            assertEquals(0, batcher.queueSize());
            assertEquals(1, batcher.callbackOwnedRequestCount());

            callbackMayComplete.countDown();
            assertTrue(callbackCompleted.await(2, TimeUnit.SECONDS));
            awaitTrue(() -> batcher.callbackOwnedRequestCount() == 0);
            assertEquals(1, callbacks.get());
            assertTrue(failure.get() == null, failure::get);
        } finally {
            callbackMayComplete.countDown();
            batcher.shutdown();
        }
    }

    @Test
    void removingBlockedDecodeHeadUnsubscribesCapacityListener() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        TrackingDecodeEndpoint decode = new TrackingDecodeEndpoint();
        BatchItem blocked = routeItem(
                1, 50, System.currentTimeMillis(), Long.MAX_VALUE, decode);
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blocked, false);
        WorkerBatcher batcher = new WorkerBatcher(
                "decode-listener-worker",
                mock(PrefillEndpoint.class),
                config,
                resolvingHandler((group, metadata) -> { }),
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(blocked));
        batcher.start();
        try {
            assertTrue(decode.awaitSubscribed());
            assertEquals(List.of(blocked), batcher.tryRemove(List.of(1L), "cancel"));
            assertTrue(decode.awaitUnsubscribed());
            assertFalse(decode.hasListener());
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void emptyWorkerWaitsAndOfferWakesIt() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        CountDownLatch delivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "empty-wait-worker",
                mock(PrefillEndpoint.class),
                config,
                resolvingHandler((group, metadata) -> delivered.countDown()),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));

        batcher.start();
        try {
            awaitTrue(batcher::isWaitingForSignal);
            assertTrue(batcher.tryOffer(batchItem(1, 50, System.currentTimeMillis())));
            assertTrue(delivered.await(2, TimeUnit.SECONDS));
            awaitTrue(() -> batcher.queueSize() == 0);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void capacityBlockedRequestExpiresWithoutEnteringCallback() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        long now = System.currentTimeMillis();
        BatchItem blocked = routeItem(1, 50, now, now + 100, null);
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blocked, false);
        CountDownLatch expired = new CountDownLatch(1);
        AtomicInteger callbacks = new AtomicInteger();
        WorkerBatcher batcher = new WorkerBatcher(
                "expiration-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { expired.countDown(); }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        callbacks.incrementAndGet();
                        TestCapacityAdmission.complete(group);
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                },
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(blocked));
        batcher.start();
        try {
            assertTrue(expired.await(2, TimeUnit.SECONDS));
            assertEquals(0, callbacks.get());
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void expiredCallbackFailureTerminatesOnlyThatItemAndWorkerContinues()
            throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        long now = System.currentTimeMillis();
        BatchItem expired = batchItem(1, 50, now, now - 1, 128);
        BatchItem next = batchItem(2, 50, now + 1);
        AtomicInteger expirationCallbacks = new AtomicInteger();
        AtomicInteger offerFailures = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        List<Long> deliveredRequests = new CopyOnWriteArrayList<>();
        CountDownLatch secondDelivered = new CountDownLatch(1);
        CountDownLatch thirdDelivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "expired-callback-failure-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override
                    public void onExpired(BatchItem item) {
                        assertSame(expired, item);
                        expirationCallbacks.incrementAndGet();
                        throw new IllegalStateException(
                                "synthetic expiration callback failure");
                    }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        TestCapacityAdmission.complete(group);
                        long requestId = group.requests().get(0).requestId();
                        deliveredRequests.add(requestId);
                        if (requestId == 2L) {
                            secondDelivered.countDown();
                        } else if (requestId == 3L) {
                            thirdDelivered.countDown();
                        }
                    }

                    @Override
                    public void onOfferFailure(BatchItem item, Throwable error) {
                        offerFailures.incrementAndGet();
                    }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        deliveryFailures.incrementAndGet();
                    }
                },
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(expired));
        assertTrue(batcher.tryOffer(next));
        batcher.start();
        try {
            assertTrue(secondDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(1, expirationCallbacks.get());
            assertEquals(List.of(2L), deliveredRequests);
            assertEquals(0, offerFailures.get());
            assertEquals(0, deliveryFailures.get());
            assertEquals(0, batcher.queueSize());

            BatchItem later = batchItem(3, 50, System.currentTimeMillis());
            assertTrue(batcher.tryOffer(later),
                    "an item-scoped callback failure must not stop the worker");
            assertTrue(thirdDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(List.of(2L, 3L), deliveredRequests);
            assertEquals(1, expirationCallbacks.get(),
                    "the removed expired item must not be retried");
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void batchTokenRejectionCallbackFailureTerminatesOnlyThatItemAndWorkerContinues()
            throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        WorkerStatus status = mock(WorkerStatus.class);
        org.mockito.Mockito.when(status.getMaxBatchTokensSize()).thenReturn(100L);
        org.mockito.Mockito.when(status.getTotalKvCacheTokens())
                .thenReturn(new AtomicLong(0L));
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        org.mockito.Mockito.when(prefill.getStatus()).thenReturn(status);

        long now = System.currentTimeMillis();
        BatchItem oversized = batchItem(1, 50, now, Long.MAX_VALUE, 100);
        BatchItem next = batchItem(2, 50, now + 1, Long.MAX_VALUE, 50);
        AtomicInteger rejectionCallbacks = new AtomicInteger();
        AtomicInteger expirationCallbacks = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        AtomicReference<Throwable> rejectionCause = new AtomicReference<>();
        List<Long> deliveredRequests = new CopyOnWriteArrayList<>();
        CountDownLatch secondDelivered = new CountDownLatch(1);
        CountDownLatch thirdDelivered = new CountDownLatch(1);
        WorkerBatcher batcher = new WorkerBatcher(
                "batch-token-callback-failure-worker",
                prefill,
                config,
                new DecisionGroupHandler() {
                    @Override
                    public void onExpired(BatchItem item) {
                        expirationCallbacks.incrementAndGet();
                    }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        TestCapacityAdmission.complete(group);
                        long requestId = group.requests().get(0).requestId();
                        deliveredRequests.add(requestId);
                        if (requestId == 2L) {
                            secondDelivered.countDown();
                        } else if (requestId == 3L) {
                            thirdDelivered.countDown();
                        }
                    }

                    @Override
                    public void onOfferFailure(BatchItem item, Throwable error) {
                        assertSame(oversized, item);
                        rejectionCause.set(error);
                        rejectionCallbacks.incrementAndGet();
                        throw new IllegalStateException(
                                "synthetic rejection callback failure");
                    }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        deliveryFailures.incrementAndGet();
                    }
                },
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(oversized));
        assertTrue(batcher.tryOffer(next));
        batcher.start();
        try {
            assertTrue(secondDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(1, rejectionCallbacks.get());
            assertInstanceOf(
                    BatchTokenCapacityExceededException.class,
                    rejectionCause.get());
            assertEquals(List.of(2L), deliveredRequests);
            assertEquals(0, expirationCallbacks.get());
            assertEquals(0, deliveryFailures.get());
            assertEquals(0, batcher.queueSize());

            BatchItem later = batchItem(
                    3, 50, System.currentTimeMillis(), Long.MAX_VALUE, 50);
            assertTrue(batcher.tryOffer(later),
                    "an item-scoped callback failure must not stop the worker");
            assertTrue(thirdDelivered.await(2, TimeUnit.SECONDS));
            assertEquals(List.of(2L, 3L), deliveredRequests);
            assertEquals(1, rejectionCallbacks.get(),
                    "the removed oversized item must not be retried");
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void capacityAdmissionFailureTerminatesWithoutCallback() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        AtomicInteger callbacks = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        CountDownLatch failed = new CountDownLatch(1);
        DeliveryCapacityAdmission failingAdmission = item ->
                new DeliveryCapacityAdmission.AdmissionFailed(
                        new IllegalStateException("capacity preparation failed"));
        WorkerBatcher batcher = new WorkerBatcher(
                "admission-failure-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        callbacks.incrementAndGet();
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        deliveryFailures.incrementAndGet();
                        failed.countDown();
                    }
                },
                failingAdmission,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(batchItem(1, 50, System.currentTimeMillis())));
        batcher.start();
        try {
            assertTrue(failed.await(2, TimeUnit.SECONDS));
            assertEquals(0, callbacks.get());
            assertEquals(1, deliveryFailures.get());
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void capacityFeasiblePrefixIsTheOnlyFinalDecision() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem first = batchItem(1, 50, 100);
        BatchItem second = batchItem(2, 50, 200);
        BatchItem blocked = batchItem(3, 50, 300);
        queue.addAll(List.of(first, second, blocked));
        AtomicInteger depth = new AtomicInteger(3);
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blocked, false);
        AtomicReference<List<BatchItem>> delivered = new AtomicReference<>();
        BatcherContext context = context(
                queue,
                depth,
                capacity,
                resolvingHandler((group, metadata) -> delivered.set(group.requests())));

        BatcherCycleResult result = context.admitAndDeliverCapacityFeasiblePrefix(
                List.of(first, second, blocked),
                new DecisionGroupMetadata("batch_full", 0),
                null,
                0);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class, result);
        assertEquals(List.of(first, second), admitted.items());
        assertEquals("delivery_capacity_prefix", admitted.metadata().reason());
        assertEquals(List.of(first, second), delivered.get());
        assertEquals(1, depth.get());
        assertSame(blocked, context.peek());
    }

    @Test
    void batchLoadPublicationFailureTerminatesReservedPrefixWithoutRetry()
            throws Exception {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem first = batchItem(1, 50, 100);
        BatchItem second = batchItem(2, 50, 200);
        BatchItem blocked = batchItem(3, 50, 300);
        List<BatchItem> originalSelection = List.of(first, second, blocked);
        queue.addAll(originalSelection);
        AtomicInteger depth = new AtomicInteger(3);
        IllegalStateException publicationFailure =
                new IllegalStateException("batch load publication failed");
        FailingBatchLoadPublicationAdmission capacity =
                new FailingBatchLoadPublicationAdmission(blocked, publicationFailure);
        AtomicInteger decisionCallbacks = new AtomicInteger();
        AtomicInteger firstFailures = new AtomicInteger();
        AtomicInteger secondFailures = new AtomicInteger();
        List<Long> failureOrder = new CopyOnWriteArrayList<>();
        CountDownLatch terminalFailures = new CountDownLatch(2);
        BatcherContext context = context(
                queue,
                depth,
                capacity,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        decisionCallbacks.incrementAndGet();
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        assertSame(publicationFailure, error);
                        failureOrder.add(item.requestId());
                        if (item == first) {
                            firstFailures.incrementAndGet();
                        } else if (item == second) {
                            secondFailures.incrementAndGet();
                        } else {
                            throw new AssertionError(
                                    "unreserved request received delivery failure request_id="
                                            + item.requestId());
                        }
                        terminalFailures.countDown();
                    }
                });

        BatcherCycleResult firstCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("batch_full", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, firstCycle);
        assertTrue(terminalFailures.await(0, TimeUnit.MILLISECONDS));
        assertEquals(0, decisionCallbacks.get());
        assertEquals(List.of(1L, 2L), failureOrder);
        assertEquals(1, firstFailures.get());
        assertEquals(1, secondFailures.get());
        assertEquals(List.of(1L, 2L, 3L), capacity.itemAdmissionAttempts());
        assertEquals(List.of(1L, 2L), capacity.publishedItems());
        assertEquals(List.of(1L, 2L), capacity.releasedItems());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
        assertEquals(0, capacity.batchLifecycleTransfers());
        assertEquals(1, context.size());
        assertEquals(1, depth.get());
        assertSame(blocked, context.peek());
        assertEquals(List.of(blocked), context.activeItemsInSchedulingOrder());
        assertEquals(0, context.callbackOwnedRequestCount());

        BatcherCycleResult staleCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("stale", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, staleCycle);
        assertEquals(List.of(1L, 2L, 3L), capacity.itemAdmissionAttempts());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
        assertEquals(List.of(1L, 2L), capacity.releasedItems());
        assertEquals(1, firstFailures.get());
        assertEquals(1, secondFailures.get());
        assertSame(blocked, context.peek());
    }

    @Test
    void publicationFailureKeepsTerminalAdmissionFailureCauseDistinct() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem reserved = batchItem(1, 50, 100);
        BatchItem admissionFailed = batchItem(2, 50, 200);
        List<BatchItem> originalSelection = List.of(reserved, admissionFailed);
        queue.addAll(originalSelection);
        AtomicInteger depth = new AtomicInteger(2);
        IllegalStateException publicationFailure =
                new IllegalStateException("batch load publication failed");
        IllegalStateException admissionFailure =
                new IllegalStateException("request admission failed");
        FailingBatchLoadPublicationAdmission capacity =
                new FailingBatchLoadPublicationAdmission(
                        admissionFailed,
                        new DeliveryCapacityAdmission.AdmissionFailed(admissionFailure),
                        publicationFailure);
        AtomicInteger decisionCallbacks = new AtomicInteger();
        List<Long> failedRequests = new CopyOnWriteArrayList<>();
        List<Throwable> failureCauses = new CopyOnWriteArrayList<>();
        BatcherContext context = context(
                queue,
                depth,
                capacity,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        decisionCallbacks.incrementAndGet();
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        failedRequests.add(item.requestId());
                        failureCauses.add(error);
                    }
                });

        BatcherCycleResult firstCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("batch_full", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, firstCycle);
        assertEquals(List.of(1L, 2L), failedRequests);
        assertEquals(2, failureCauses.size());
        assertSame(publicationFailure, failureCauses.get(0));
        assertSame(admissionFailure, failureCauses.get(1));
        assertEquals(0, decisionCallbacks.get());
        assertEquals(List.of(1L, 2L), capacity.itemAdmissionAttempts());
        assertEquals(List.of(1L), capacity.publishedItems());
        assertEquals(List.of(1L), capacity.releasedItems());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
        assertEquals(0, capacity.batchLifecycleTransfers());
        assertEquals(0, context.size());
        assertEquals(0, depth.get());
        assertTrue(context.activeItemsInSchedulingOrder().isEmpty());
        assertEquals(0, context.callbackOwnedRequestCount());

        BatcherCycleResult staleCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("stale", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, staleCycle);
        assertEquals(List.of(1L, 2L), failedRequests);
        assertEquals(2, failureCauses.size());
        assertEquals(List.of(1L, 2L), capacity.itemAdmissionAttempts());
        assertEquals(List.of(1L), capacity.releasedItems());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
    }

    @Test
    void publicationFailureRemovesOwnershipLostBoundaryWithoutFailureCallback() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem reserved = batchItem(1, 50, 100);
        BatchItem ownershipLost = batchItem(2, 50, 200);
        List<BatchItem> originalSelection = List.of(reserved, ownershipLost);
        queue.addAll(originalSelection);
        AtomicInteger depth = new AtomicInteger(2);
        IllegalStateException publicationFailure =
                new IllegalStateException("batch load publication failed");
        FailingBatchLoadPublicationAdmission capacity =
                new FailingBatchLoadPublicationAdmission(
                        ownershipLost,
                        DeliveryCapacityAdmission.OwnershipLost.INSTANCE,
                        publicationFailure);
        List<Long> failedRequests = new CopyOnWriteArrayList<>();
        List<Throwable> failureCauses = new CopyOnWriteArrayList<>();
        BatcherContext context = context(
                queue,
                depth,
                capacity,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        throw new AssertionError("publication failure entered callback");
                    }

                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem item, Throwable error) {
                        failedRequests.add(item.requestId());
                        failureCauses.add(error);
                    }
                });

        BatcherCycleResult firstCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("batch_full", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, firstCycle);
        assertEquals(List.of(1L), failedRequests,
                "ownership-lost boundary already has a terminal owner");
        assertEquals(1, failureCauses.size());
        assertSame(publicationFailure, failureCauses.get(0));
        assertEquals(List.of(1L, 2L), capacity.itemAdmissionAttempts());
        assertEquals(List.of(1L), capacity.publishedItems());
        assertEquals(List.of(1L), capacity.releasedItems());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
        assertEquals(0, capacity.batchLifecycleTransfers());
        assertEquals(0, context.size());
        assertEquals(0, depth.get());
        assertTrue(context.activeItemsInSchedulingOrder().isEmpty());
        assertEquals(0, context.callbackOwnedRequestCount());

        BatcherCycleResult staleCycle = context.admitAndDeliverCapacityFeasiblePrefix(
                originalSelection,
                new DecisionGroupMetadata("stale", 0),
                null,
                0);

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, staleCycle);
        assertEquals(List.of(1L), failedRequests);
        assertEquals(List.of(1L, 2L), capacity.itemAdmissionAttempts());
        assertEquals(List.of(1L), capacity.releasedItems());
        assertEquals(1, capacity.batchReservationAttempts());
        assertEquals(1, capacity.publicationAttempts());
        assertEquals(1, capacity.batchReleaseCalls());
    }

    @Test
    void successfulTypedCallbackKeepsDistinctRequestGenerations() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem first = batchItem(7, 50, 100);
        BatchItem second = batchItem(7, 50, 200);
        queue.addAll(List.of(first, second));
        AtomicReference<List<BatchItem>> delivered = new AtomicReference<>();
        BatcherContext context = context(
                queue,
                new AtomicInteger(2),
                TestCapacityAdmission.alwaysAvailable(),
                resolvingHandler((group, metadata) -> delivered.set(group.requests())));

        BatcherCycleResult result = context.admitAndDeliverCapacityFeasiblePrefix(
                List.of(first, second),
                new DecisionGroupMetadata("test", 0),
                null,
                0);

        assertInstanceOf(BatcherCycleResult.Admitted.class, result);
        assertEquals(2, delivered.get().size());
        assertSame(first, delivered.get().get(0));
        assertSame(second, delivered.get().get(1));
        assertEquals(0, context.size());
    }

    @Test
    void admittedRequestCannotBeDeliveredAgainFromStaleSelection() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem item = batchItem(1, 50, 100);
        queue.add(item);
        AtomicInteger callbacks = new AtomicInteger();
        BatcherContext context = context(
                queue,
                new AtomicInteger(1),
                TestCapacityAdmission.alwaysAvailable(),
                resolvingHandler((group, metadata) -> callbacks.incrementAndGet()));

        BatcherCycleResult first = context.admitAndDeliverCapacityFeasiblePrefix(
                List.of(item), new DecisionGroupMetadata("test", 0), null, 0);
        BatcherCycleResult second = context.admitAndDeliverCapacityFeasiblePrefix(
                List.of(item), new DecisionGroupMetadata("stale", 0), null, 0);

        assertInstanceOf(BatcherCycleResult.Admitted.class, first);
        assertSame(BatcherCycleResult.Outcome.NO_ACTION, second);
        assertEquals(1, callbacks.get());
        assertEquals(0, context.callbackOwnedRequestCount());
    }

    @Test
    void callbackThatCommitsThenThrowsTerminatesWithoutRequeue() {
        PriorityBlockingQueue<BatchItem> queue = queue();
        BatchItem item = batchItem(1, 50, 100);
        queue.add(item);
        OneBatchSlotCapacityAdmission capacity =
                new OneBatchSlotCapacityAdmission();
        AtomicInteger callbackInvocations = new AtomicInteger();
        AtomicInteger deliveryFailures = new AtomicInteger();
        BatcherContext context = context(
                queue,
                new AtomicInteger(1),
                capacity,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        callbackInvocations.incrementAndGet();
                        assertTrue(group.members().get(0)
                                .transferCapacityToEndpointLifecycle());
                        throw new IllegalStateException("failed after commit");
                    }

                    @Override public void onOfferFailure(BatchItem request, Throwable error) { }

                    @Override
                    public void onDeliveryFailure(BatchItem request, Throwable error) {
                        assertSame(item, request);
                        deliveryFailures.incrementAndGet();
                    }
                });

        BatcherCycleResult result = assertDoesNotThrow(() ->
                context.admitAndDeliverCapacityFeasiblePrefix(
                        List.of(item),
                        new DecisionGroupMetadata("test", 0),
                        null,
                        0));
        BatcherCycleResult staleReplay = assertDoesNotThrow(() ->
                context.admitAndDeliverCapacityFeasiblePrefix(
                        List.of(item),
                        new DecisionGroupMetadata("stale", 0),
                        null,
                        0));

        assertInstanceOf(BatcherCycleResult.Admitted.class, result);
        assertSame(BatcherCycleResult.Outcome.NO_ACTION, staleReplay);
        assertEquals(1, callbackInvocations.get());
        assertEquals(1, deliveryFailures.get());
        assertEquals(1, capacity.reservationCount());
        assertEquals(1, capacity.releaseCount());
        assertEquals(0, capacity.registrationCount());
        assertEquals(0, context.size());
        assertEquals(0, context.callbackOwnedRequestCount());
        assertTrue(context.activeItemsInSchedulingOrder().isEmpty());
    }

    @Test
    void shutdownDrainsCapacityBlockedActiveRequestExactlyOnce() throws Exception {
        SchedulingTestConfig.useSingleDecision(config);
        BatchItem blocked = routeItem(1, 50, System.currentTimeMillis());
        ExactItemCapacityGate capacity = new ExactItemCapacityGate(blocked, false);
        AtomicInteger callbackCount = new AtomicInteger();
        AtomicInteger shutdownFailures = new AtomicInteger();
        WorkerBatcher batcher = new WorkerBatcher(
                "shutdown-worker",
                mock(PrefillEndpoint.class),
                config,
                new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }

                    @Override
                    public void onDecisionGroupAdmitted(
                            AdmittedDecisionGroup group,
                            DecisionGroupMetadata metadata) {
                        callbackCount.incrementAndGet();
                    }

                    @Override
                    public void onOfferFailure(BatchItem item, Throwable error) {
                        shutdownFailures.incrementAndGet();
                    }

                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                },
                capacity,
                mock(BatchSchedulerReporter.class));

        assertTrue(batcher.tryOffer(blocked));
        batcher.start();
        assertTrue(capacity.awaitBlocked());
        batcher.shutdown();

        awaitTrue(() -> shutdownFailures.get() == 1);
        assertEquals(0, callbackCount.get());
        assertEquals(0, batcher.queueSize());
        assertEquals(0, batcher.callbackOwnedRequestCount());
    }

    private WorkerBatcher newIdleBatcher() {
        return new WorkerBatcher(
                "test-worker",
                null,
                config,
                mock(DecisionGroupHandler.class),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));
    }

    private void assertDecisionAndDeliveryMode(
            boolean singleDecision,
            boolean batchDelivery,
            List<Integer> expectedGroupSizes,
            String expectedReason) throws Exception {
        SchedulingTestConfig.useFifoQueue(config);
        if (singleDecision) {
            SchedulingTestConfig.useSingleDecision(config);
        } else {
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
            SchedulingTestConfig.useFixedWindowDecision(config)
                    .setMaxCollectionWaitMs(60_000);
        }
        if (batchDelivery) {
            SchedulingTestConfig.useBatchDispatcher(config);
        } else {
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }

        List<Integer> actualGroupSizes = new CopyOnWriteArrayList<>();
        List<String> actualReasons = new CopyOnWriteArrayList<>();
        CountDownLatch delivered = new CountDownLatch(expectedGroupSizes.size());
        WorkerBatcher batcher = new WorkerBatcher(
                "mode-worker",
                mock(PrefillEndpoint.class),
                config,
                resolvingHandler((group, metadata) -> {
                    actualGroupSizes.add(group.requests().size());
                    actualReasons.add(metadata.reason());
                    delivered.countDown();
                }),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));

        BatchItem first = batchDelivery
                ? batchItem(1, 50, System.currentTimeMillis())
                : routeItem(1, 50, System.currentTimeMillis());
        BatchItem second = batchDelivery
                ? batchItem(2, 50, System.currentTimeMillis())
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
        assertEquals(
                expectedGroupSizes.stream().map(ignored -> expectedReason).toList(),
                actualReasons);
    }

    private static DecisionGroupHandler resolvingHandler(GroupObserver observer) {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }

            @Override
            public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group,
                    DecisionGroupMetadata metadata) {
                observer.accept(group, metadata);
                TestCapacityAdmission.complete(group);
            }

            @Override public void onOfferFailure(BatchItem item, Throwable error) { }

            @Override
            public void onDeliveryFailure(BatchItem item, Throwable error) {
                throw new AssertionError("unexpected delivery failure", error);
            }
        };
    }

    private BatcherContext context(
            PriorityBlockingQueue<BatchItem> queue,
            AtomicInteger depth,
            DeliveryCapacityAdmission capacityAdmission,
            DecisionGroupHandler handler) {
        return new BatcherContext(
                "test-worker",
                null,
                config,
                handler,
                capacityAdmission,
                queue,
                depth,
                new AtomicLong(),
                new ReentrantLock(),
                WorkerBatcher.PRIORITY_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
    }

    private static PriorityBlockingQueue<BatchItem> queue() {
        return new PriorityBlockingQueue<>(11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
    }

    private static BatchItem batchItem(long requestId, int priority, long enqueuedAtMs) {
        return batchItem(
                requestId, priority, enqueuedAtMs, Long.MAX_VALUE, 128);
    }

    private static BatchItem batchItem(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen) {
        BalanceContext context = newContext(requestId, priority);
        context.getRequest().setSeqLen(seqLen);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, expiresAtMs));
        return new BatchItem(context, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BatchItem itemWithoutSchedulingMetadata(
            long requestId, long enqueuedAtMs) {
        return new BatchItem(newContext(requestId, 0), new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BatchItem routeItem(long requestId, int priority, long enqueuedAtMs) {
        return routeItem(requestId, priority, enqueuedAtMs, Long.MAX_VALUE, null);
    }

    private static BatchItem routeItem(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            DecodeEndpoint decodeEndpoint) {
        BalanceContext context = newContext(requestId, priority);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, expiresAtMs));
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        return new BatchItem(context, new CompletableFuture<>(), null,
                null, null, null, decodeEndpoint, enqueuedAtMs);
    }

    private static BalanceContext newContext(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return context;
    }

    private static void awaitTrue(BooleanSupplier condition) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (!condition.getAsBoolean() && System.nanoTime() < deadlineNanos) {
            TimeUnit.MILLISECONDS.sleep(5);
        }
        assertTrue(condition.getAsBoolean(),
                "condition did not become true within two seconds");
    }

    @FunctionalInterface
    private interface GroupObserver {
        void accept(AdmittedDecisionGroup group, DecisionGroupMetadata metadata);
    }

    private static final class ExactItemCapacityGate implements DeliveryCapacityAdmission {
        private final BatchItem blockedItem;
        private final AtomicBoolean available;
        private final CountDownLatch blocked = new CountDownLatch(1);

        private ExactItemCapacityGate(BatchItem blockedItem, boolean initiallyAvailable) {
            this.blockedItem = blockedItem;
            this.available = new AtomicBoolean(initiallyAvailable);
        }

        @Override
        public AdmissionResult tryReserveItemCapacity(BatchItem item) {
            if (item == blockedItem && !available.get()) {
                blocked.countDown();
                DecodeEndpoint decodeEndpoint = item.decodeEp();
                return new CapacityUnavailable(
                        CapacityResource.DECODE_ENGINE,
                        new CapacityAvailability() {
                            @Override
                            public boolean isAvailable() {
                                return available.get();
                            }

                            @Override
                            public void addListener(Runnable listener) {
                                if (decodeEndpoint != null) {
                                    decodeEndpoint.addEngineDispatchCapacityListener(listener);
                                }
                            }

                            @Override
                            public void removeListener(Runnable listener) {
                                if (decodeEndpoint != null) {
                                    decodeEndpoint.removeEngineDispatchCapacityListener(listener);
                                }
                            }
                        });
            }
            return TestCapacityAdmission.alwaysAvailable().tryReserveItemCapacity(item);
        }

        @Override
        public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
            return TestCapacityAdmission.alwaysAvailable()
                    .tryReserveBatchCapacity(head);
        }

        private boolean awaitBlocked() throws InterruptedException {
            return blocked.await(2, TimeUnit.SECONDS);
        }

        private void makeAvailable() {
            available.set(true);
        }
    }

    /** One real group slot used to prove callback failure cleanup unblocks the worker. */
    private static final class OneBatchSlotCapacityAdmission
            implements DeliveryCapacityAdmission {
        private final DeliveryCapacityAdmission itemCapacity =
                TestCapacityAdmission.alwaysAvailable();
        private final AtomicBoolean slotAvailable = new AtomicBoolean(true);
        private final AtomicInteger reservations = new AtomicInteger();
        private final AtomicInteger releases = new AtomicInteger();
        private final AtomicInteger registrations = new AtomicInteger();

        @Override
        public AdmissionResult tryReserveItemCapacity(BatchItem item) {
            return itemCapacity.tryReserveItemCapacity(item);
        }

        @Override
        public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
            if (!slotAvailable.compareAndSet(true, false)) {
                return new BatchCapacityUnavailable(
                        CapacityResource.PREFILL_BATCH, slotAvailable::get);
            }
            reservations.incrementAndGet();
            return new BatchCapacityReserved(new BatchCapacityReservation() {
                private boolean resolved;

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
                public synchronized BatchDispatcher.SubmissionPermit
                        transferToBatchLifecycle(
                        long batchId, long predictedMs, List<BatchItem> requests) {
                    if (resolved) {
                        throw new IllegalStateException(
                                "batch slot was already resolved");
                    }
                    resolved = true;
                    registrations.incrementAndGet();
                    return new BatchDispatcher.SubmissionPermit() {
                        @Override
                        public void submit(
                                List<BatchItem> submittedItems,
                                PrefillEndpoint prefillEndpoint,
                                long submittedBatchId,
                                long submittedPredictedMs,
                                String reason,
                                DispatchCallback callback) {
                        }

                        @Override
                        public void release() {
                        }
                    };
                }

                @Override
                public synchronized void completeDeliveryHandoff() {
                    if (!resolved) {
                        throw new IllegalStateException(
                                "batch slot was not registered before handoff completion");
                    }
                }

                @Override
                public synchronized void release() {
                    if (resolved) {
                        return;
                    }
                    resolved = true;
                    releases.incrementAndGet();
                    slotAvailable.set(true);
                }
            });
        }

        int reservationCount() {
            return reservations.get();
        }

        int releaseCount() {
            return releases.get();
        }

        int registrationCount() {
            return registrations.get();
        }
    }

    /**
     * Reserves one batch slot and an ordered item prefix, then reports a typed
     * failure while establishing callback-owned batch load.
     */
    private static final class FailingBatchLoadPublicationAdmission
            implements DeliveryCapacityAdmission {
        private final BatchItem terminalBoundaryItem;
        private final AdmissionResult terminalBoundaryResult;
        private final Throwable publicationFailure;
        private final List<Long> itemAdmissionAttempts = new CopyOnWriteArrayList<>();
        private final List<Long> publishedItems = new CopyOnWriteArrayList<>();
        private final List<Long> releasedItems = new CopyOnWriteArrayList<>();
        private final AtomicInteger batchReservationAttempts = new AtomicInteger();
        private final AtomicInteger publicationAttempts = new AtomicInteger();
        private final AtomicInteger batchReleaseCalls = new AtomicInteger();
        private final AtomicInteger batchLifecycleTransfers = new AtomicInteger();

        private FailingBatchLoadPublicationAdmission(
                BatchItem blockedItem,
                Throwable publicationFailure) {
            this(
                    blockedItem,
                    new CapacityUnavailable(
                            CapacityResource.DECODE_ENGINE, () -> false),
                    publicationFailure);
        }

        private FailingBatchLoadPublicationAdmission(
                BatchItem terminalBoundaryItem,
                AdmissionResult terminalBoundaryResult,
                Throwable publicationFailure) {
            this.terminalBoundaryItem = terminalBoundaryItem;
            this.terminalBoundaryResult = terminalBoundaryResult;
            this.publicationFailure = publicationFailure;
        }

        @Override
        public AdmissionResult tryReserveItemCapacity(BatchItem item) {
            itemAdmissionAttempts.add(item.requestId());
            if (item == terminalBoundaryItem) {
                return terminalBoundaryResult;
            }
            return new CapacityReserved(new ItemCapacityReservation() {
                @Override
                public BatchItem item() {
                    return item;
                }

                @Override
                public boolean transferToEndpointLifecycle() {
                    throw new AssertionError(
                            "publication failure must precede item lifecycle transfer");
                }

                @Override
                public void release() {
                    releasedItems.add(item.requestId());
                }
            });
        }

        @Override
        public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
            batchReservationAttempts.incrementAndGet();
            return new BatchCapacityReserved(new BatchCapacityReservation() {
                @Override
                public BatchItem head() {
                    return head;
                }

                @Override
                public BatchLoadPublicationResult establishBatchLoadPublication(
                        List<BatchItem> requests) {
                    publicationAttempts.incrementAndGet();
                    publishedItems.addAll(
                            requests.stream().map(BatchItem::requestId).toList());
                    return new BatchLoadPublicationFailed(publicationFailure);
                }

                @Override
                public BatchDispatcher.SubmissionPermit transferToBatchLifecycle(
                        long batchId,
                        long predictedMs,
                        List<BatchItem> requests) {
                    batchLifecycleTransfers.incrementAndGet();
                    throw new AssertionError(
                            "publication failure must precede batch lifecycle transfer");
                }

                @Override
                public void completeDeliveryHandoff() {
                    throw new AssertionError(
                            "untransferred batch must not complete delivery handoff");
                }

                @Override
                public void release() {
                    batchReleaseCalls.incrementAndGet();
                }
            });
        }

        List<Long> itemAdmissionAttempts() {
            return List.copyOf(itemAdmissionAttempts);
        }

        List<Long> publishedItems() {
            return List.copyOf(publishedItems);
        }

        List<Long> releasedItems() {
            return List.copyOf(releasedItems);
        }

        int batchReservationAttempts() {
            return batchReservationAttempts.get();
        }

        int publicationAttempts() {
            return publicationAttempts.get();
        }

        int batchReleaseCalls() {
            return batchReleaseCalls.get();
        }

        int batchLifecycleTransfers() {
            return batchLifecycleTransfers.get();
        }
    }

    private static final class TrackingDecodeEndpoint extends DecodeEndpoint {
        private final AtomicReference<Runnable> listener = new AtomicReference<>();
        private final CountDownLatch subscribed = new CountDownLatch(1);
        private final CountDownLatch unsubscribed = new CountDownLatch(1);

        private TrackingDecodeEndpoint() {
            super(decodeWorkerStatus());
        }

        @Override
        public void addEngineDispatchCapacityListener(Runnable newListener) {
            super.addEngineDispatchCapacityListener(newListener);
            if (listener.compareAndSet(null, newListener)) {
                subscribed.countDown();
            }
        }

        @Override
        public void removeEngineDispatchCapacityListener(Runnable oldListener) {
            super.removeEngineDispatchCapacityListener(oldListener);
            if (listener.compareAndSet(oldListener, null)) {
                unsubscribed.countDown();
            }
        }

        private boolean awaitSubscribed() throws InterruptedException {
            return subscribed.await(2, TimeUnit.SECONDS);
        }

        private boolean awaitUnsubscribed() throws InterruptedException {
            return unsubscribed.await(2, TimeUnit.SECONDS);
        }

        private boolean hasListener() {
            return listener.get() != null;
        }

        private static WorkerStatus decodeWorkerStatus() {
            WorkerStatus status = new WorkerStatus();
            status.setIp("127.0.0.9");
            status.setPort(8009);
            status.setGrpcPort(9009);
            return status;
        }
    }
}
