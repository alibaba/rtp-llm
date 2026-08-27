package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryContext;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Canonical queue-order contract exposed by {@link PrefillGenerationRuntime}. */
class PrefillGenerationQueueTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();
    private FlexlbConfig config;
    private PrefillEndpoint prefillEndpoint;
    private BlockingDeliveryStrategy deliveryStrategy;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        prefillEndpoint = stablePrefillEndpoint();
        deliveryStrategy = new BlockingDeliveryStrategy();
    }

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void priorityOrderIsPriorityDescendingThenActualOfferFifo() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(runtime.offer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 5_000, now - 100, 128)));

        PrefillGenerationRuntime.QueueSnapshot snapshot =
                runtime.captureQueueSnapshot();
        assertEquals(List.of(2L, 1L, 3L, 4L), requestIds(snapshot.items()));
        assertEquals(4, snapshot.items().size());
        assertEquals(SchedulingTestConfig.useQueueCapacity(config)
                .getMaxWaitingRequestsPerPrefillWorker(), snapshot.queueCapacity());
        assertEquals("test-worker", snapshot.endpointId());
    }

    @Test
    void priorityOrderUsesUniqueOfferSequenceBeforeRequestIdOrCallerTimestamp() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 9_000, now, 128)));

        assertEquals(List.of(1L, 2L, 4L, 3L), requestIds(
                runtime.captureQueueSnapshot().items()));
    }

    @Test
    void fifoOrderIgnoresPriorityAndCallerTimestamp() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 70, now + 100, now + 200, 128)));

        assertEquals(List.of(1L, 2L, 3L), requestIds(
                runtime.captureQueueSnapshot().items()));
    }

    @Test
    void snapshotsExposeExactCanonicalIdentitiesAndMonotonicMutationVersion() {
        WorkerBatcher runtime = runningRuntime();
        BatchItem first = item(11, 70, Long.MAX_VALUE, 1, 128);
        BatchItem second = item(12, 50, Long.MAX_VALUE, 2, 128);

        PrefillGenerationRuntime.QueueSnapshot empty =
                runtime.captureQueueSnapshot();
        assertTrue(runtime.offer(first));
        assertTrue(runtime.offer(second));
        PrefillGenerationRuntime.QueueSnapshot offered =
                runtime.captureQueueSnapshot();

        assertTrue(offered.queueVersion() > empty.queueVersion());
        assertSame(first, offered.items().get(0));
        assertSame(second, offered.items().get(1));
        assertEquals(2, runtime.queueSize());
        assertEquals(java.util.Map.of(70, 1, 50, 1),
                runtime.queueSizeByPriority());

        assertTrue(runtime.removeQueued(first, "test exact removal"));
        PrefillGenerationRuntime.QueueSnapshot removed =
                runtime.captureQueueSnapshot();
        assertTrue(removed.queueVersion() > offered.queueVersion());
        assertEquals(List.of(second), removed.items());
    }

    @Test
    void removalRequiresTheExactQueuedIdentityEvenForSameRequestId() {
        WorkerBatcher runtime = runningRuntime();
        BatchItem canonical = item(21, 50, Long.MAX_VALUE, 1, 128);
        BatchItem lookalike = item(21, 50, Long.MAX_VALUE, 1, 128);
        assertTrue(runtime.offer(canonical));

        assertFalse(runtime.removeQueued(lookalike, "stale identity"));
        assertEquals(List.of(canonical),
                runtime.captureQueueSnapshot().items());
        assertTrue(runtime.removeQueued(canonical, "canonical identity"));
    }

    @Test
    void preparedOfferHoldsBothSeatsAndCloseSignalsFreedCapacity() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        SchedulingTestConfig.useQueueCapacity(config)
                .setMaxWaitingRequestsPerPrefillWorker(1);
        WorkerBatcher runtime = runningRuntime();
        CapacityBoundary.Availability availability = runtime.offerAvailability();
        AtomicInteger signals = new AtomicInteger();
        availability.addListener(signals::incrementAndGet);

        long before = runtime.offerCapacityEpoch();
        try (PrefillGenerationRuntime.PreparedOffer prepared =
                     runtime.prepareOffer(31L, 50)) {
            assertEquals(1L, runtime.pendingRequestCount());
            assertFalse(availability.isAvailable());
            PrefillGenerationRuntime.QueueSnapshot held =
                    runtime.captureQueueSnapshot();
            assertEquals(1L, held.waitingCount());
            assertTrue(held.items().isEmpty());
            assertNull(runtime.prepareOffer(32L, 50));
            assertFalse(runtime.offer(item(
                    33L, 50, Long.MAX_VALUE,
                    System.currentTimeMillis(), 128L)));
        }

        assertTrue(availability.isAvailable());
        assertTrue(runtime.offerCapacityEpoch() > before);
        assertEquals(1, signals.get());
        assertSame(availability, runtime.offerAvailability());
    }

    @Test
    void fullProbeRefreshesExternalPendingPressureBeforeReleaseSignal() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        CapacityBoundary.Availability availability = runtime.offerAvailability();
        AtomicInteger signals = new AtomicInteger();
        availability.addListener(signals::incrementAndGet);
        long before = runtime.offerCapacityEpoch();

        assertTrue(availability.isAvailable());
        try (PrefillWorkLedger.DirectRegistration registration =
                     runtime.ownedLedger().tryRegisterDirect(35L, 1L)) {
            assertNotNull(registration);
            assertNull(runtime.prepareOffer(36L, 50));
            assertFalse(availability.isAvailable(),
                    "the failed capacity probe must publish the external ledger pressure");
            assertEquals(before, runtime.offerCapacityEpoch(),
                    "capacity consumption is not a release");
        }

        assertTrue(availability.isAvailable());
        assertTrue(runtime.offerCapacityEpoch() > before);
        assertEquals(1, signals.get());
    }

    @Test
    void preparedOfferCommitIsExactOneShotAndCapacityNeutral() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        BatchItem exact = item(
                41L, 50, Long.MAX_VALUE,
                System.currentTimeMillis(), 128L);
        PrefillGenerationRuntime.PreparedOffer prepared =
                runtime.prepareOffer(41L, 50);
        long heldEpoch = runtime.offerCapacityEpoch();

        assertTrue(prepared.seal());
        assertTrue(prepared.seal());
        prepared.commit(exact);

        assertEquals(heldEpoch, runtime.offerCapacityEpoch());
        assertEquals(1L, runtime.pendingRequestCount());
        assertEquals(List.of(exact), runtime.captureQueueSnapshot().items());
        assertThrows(IllegalStateException.class, () -> prepared.commit(exact));
        prepared.close();
        assertEquals(1L, runtime.pendingRequestCount());
        assertTrue(runtime.removeQueued(exact, "test committed hold release"));
        assertEquals(0L, runtime.pendingRequestCount());
    }

    @Test
    void preparedOfferRejectsAnotherRequestWithoutConsumingItsHold() {
        WorkerBatcher runtime = runningRuntime();
        try (PrefillGenerationRuntime.PreparedOffer prepared =
                     runtime.prepareOffer(51L, 50)) {
            BatchItem wrong = item(
                    52L, 50, Long.MAX_VALUE,
                    System.currentTimeMillis(), 128L);
            assertTrue(prepared.seal());
            assertThrows(IllegalArgumentException.class,
                    () -> prepared.commit(wrong));
            assertEquals(1L, runtime.pendingRequestCount());
        }
        assertEquals(0L, runtime.pendingRequestCount());
    }

    @Test
    void priorityFullReplacesLatestLowestOpenHoldWithoutCapacitySignal() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(2L);
        SchedulingTestConfig.useQueueCapacity(config)
                .setMaxWaitingRequestsPerPrefillWorker(2);
        WorkerBatcher runtime = runningRuntime();
        AtomicInteger signals = new AtomicInteger();
        runtime.offerAvailability().addListener(signals::incrementAndGet);
        PrefillGenerationRuntime.PreparedOffer olderLow =
                runtime.prepareOffer(101L, 20);
        PrefillGenerationRuntime.PreparedOffer latestLow =
                runtime.prepareOffer(102L, 20);
        long fullEpoch = runtime.offerCapacityEpoch();

        PrefillGenerationRuntime.PreparedOffer incoming =
                runtime.prepareOffer(103L, 80);

        assertNotNull(olderLow);
        assertNotNull(latestLow);
        assertNotNull(incoming);
        assertTrue(olderLow.seal());
        assertFalse(latestLow.seal());
        assertTrue(incoming.seal());
        assertEquals(2L, runtime.pendingRequestCount());
        assertEquals(2L, runtime.captureQueueSnapshot().waitingCount());
        assertEquals(fullEpoch, runtime.offerCapacityEpoch());
        assertEquals(0, signals.get());

        latestLow.close();
        assertEquals(2L, runtime.pendingRequestCount());
        olderLow.close();
        incoming.close();
        assertEquals(0L, runtime.pendingRequestCount());
        assertEquals(2, signals.get());
    }

    @Test
    void fifoFullDoesNotReplaceOpenHold() {
        SchedulingTestConfig.useFifoQueue(config);
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        try (PrefillGenerationRuntime.PreparedOffer first =
                     runtime.prepareOffer(111L, 20)) {
            assertNull(runtime.prepareOffer(112L, 80));
            assertTrue(first.seal());
        }
    }

    @Test
    void priorityFullDoesNotReplaceEqualPriorityOpenHold() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        try (PrefillGenerationRuntime.PreparedOffer first =
                     runtime.prepareOffer(121L, 50)) {
            assertNull(runtime.prepareOffer(122L, 50));
            assertTrue(first.seal());
        }
    }

    @Test
    void priorityFullDoesNotReplaceSealedHold() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        try (PrefillGenerationRuntime.PreparedOffer first =
                     runtime.prepareOffer(131L, 20)) {
            assertTrue(first.seal());
            assertNull(runtime.prepareOffer(132L, 80));
            assertTrue(first.seal());
        }
    }

    @Test
    void preparedOfferCommitRequiresSealAndPreservesOpenHold() {
        WorkerBatcher runtime = runningRuntime();
        BatchItem exact = item(
                141L, 50, Long.MAX_VALUE,
                System.currentTimeMillis(), 128L);
        PrefillGenerationRuntime.PreparedOffer prepared =
                runtime.prepareOffer(141L, 50);

        assertThrows(IllegalStateException.class, () -> prepared.commit(exact));
        assertEquals(1L, runtime.pendingRequestCount());
        assertTrue(prepared.seal());
        prepared.commit(exact);
        assertFalse(prepared.seal());
        assertEquals(List.of(exact), runtime.captureQueueSnapshot().items());
        assertTrue(runtime.removeQueued(exact, "test sealed commit release"));
    }

    @Test
    void pendingCapCanReplaceTheExactWorstActiveWithStrictlyHigherPriority() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        BatchItem victim = item(61L, 50, Long.MAX_VALUE, 1L, 128L);
        BatchItem incoming = item(62L, 80, Long.MAX_VALUE, 2L, 128L);
        assertTrue(runtime.offer(victim));

        PrefillGenerationRuntime.QueueSnapshot snapshot =
                runtime.captureQueueSnapshot();
        assertEquals(1L, snapshot.pendingCount());
        assertEquals(1L, snapshot.maxPendingRequests());
        assertTrue(snapshot.items().size() < snapshot.queueCapacity());
        assertEquals(PrefillGenerationRuntime.QueueReplacementStatus.SUCCESS,
                runtime.replaceQueued(List.of(victim), incoming).status());
        assertEquals(List.of(incoming), runtime.captureQueueSnapshot().items());
    }

    @Test
    void pendingCapDoesNotReplaceEqualPriority() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        BatchItem victim = item(71L, 50, Long.MAX_VALUE, 1L, 128L);
        BatchItem incoming = item(72L, 50, Long.MAX_VALUE, 2L, 128L);
        assertTrue(runtime.offer(victim));

        assertEquals(PrefillGenerationRuntime.QueueReplacementStatus.DECLINED,
                runtime.replaceQueued(List.of(victim), incoming).status());
        assertEquals(List.of(victim), runtime.captureQueueSnapshot().items());
    }

    @Test
    void pendingCapWithoutAnActiveVictimDeclinesReplacement() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(1L);
        WorkerBatcher runtime = runningRuntime();
        try (PrefillWorkLedger.DirectRegistration registration =
                     runtime.ownedLedger().tryRegisterDirect(81L, 1L)) {
            registration.commit();
        }

        assertEquals(1L, runtime.pendingRequestCount());
        assertTrue(runtime.captureQueueSnapshot().items().isEmpty());
        assertEquals(PrefillGenerationRuntime.QueueReplacementStatus.DECLINED,
                runtime.replaceQueued(
                        List.of(),
                        item(82L, 80, Long.MAX_VALUE, 2L, 128L)).status());
    }

    private WorkerBatcher runningRuntime() {
        WorkerBatcher runtime = new WorkerBatcher(
                "test-worker",
                prefillEndpoint,
                config,
                deliveryStrategy,
                mock(DeliveryLifecyclePort.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private BatchItem item(long requestId, int priority, long expiresAtMs,
                           long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, expiresAtMs));
        return new BatchItem(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                prefillEndpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static List<Long> requestIds(List<DeliveryItem> items) {
        return items.stream().map(DeliveryItem::requestId).toList();
    }

    private static PrefillEndpoint stablePrefillEndpoint() {
        PrefillTimePredictor.Evaluator evaluator =
                mock(PrefillTimePredictor.Evaluator.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "test",
                "127.0.0.1",
                8080,
                8090,
                "test-site"));
        return endpoint;
    }

    /** Holds every exact head ACTIVE so snapshots can exercise live ordering. */
    private static final class BlockingDeliveryStrategy
            implements DeliveryStrategy {

        private final AtomicInteger attempts = new AtomicInteger();
        private final CapacityBoundary.Availability availability =
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return false;
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                };

        @Override
        public <R> R admitAndDeliver(
                List<DeliveryItem> candidates,
                DeliveryMetadata metadata,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction,
                DeliveryContext<R> context) {
            attempts.incrementAndGet();
            return context.commitBoundary(
                    new DeliveryContext.SelectionBoundary(
                            candidates.getFirst(),
                            new CapacityBoundary.Unavailable(
                                    availability,
                                    new RouteProjection.AdmissionBlockSemantics(
                                            "TEST_QUEUE_BLOCK",
                                            RouteProjection.AfterProbeAdmission.BLOCKED,
                                            "TEST_QUEUE_BLOCK"))));
        }

        @Override
        public double projectGroupDurationMs(
                List<DeliveryItem> items,
                PrefillTimePredictor.Evaluator evaluator) {
            return 0.0;
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }
    }
}
