package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bind;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bindRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Executable documentation for the RequestSlot delivery lock boundary. */
class RequestLifecycleDeliveryLockContractTest {

    private FlexlbConfig config;
    private RequestRegistry lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestRegistry(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (lifecycle.closeAdmissionAndAwaitMutations()) {
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
        }
    }

    @Test
    void itemPublicationDoesNotOwnTheExactSlotMonitor()
            throws Exception {
        Registered registered = registerItem(101L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        AdmissionMutation admission =
                lifecycle.claimAdmissionMutation(
                        registered.item().requestId(), registered.future());
        assertNotNull(admission);
        CountDownLatch actionEntered = new CountDownLatch(1);
        CountDownLatch releaseAction = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (slot) {
                contenderEntered.countDown();
            }
        }, "commit-inflight-slot-contender");

        try {
            Future<Boolean> committed = owner.submit(() ->
                    lifecycle.commitItemForPublication(
                            registered.item(),
                            () -> {
                                actionEntered.countDown();
                                assertFalse(Thread.holdsLock(slot));
                                await(releaseAction);
                                return true;
                            }));

            assertTrue(actionEntered.await(5, TimeUnit.SECONDS));
            contender.start();
            assertTrue(contenderStarted.await(5, TimeUnit.SECONDS));
            assertTrue(contenderEntered.await(5, TimeUnit.SECONDS),
                    "slot operations must proceed during endpoint publication");
            synchronized (slot) {
                assertSame(registered.item(), slot.activeItem());
            }
            assertTrue(lifecycle.prepareIfOwned(
                            registered.item(), () -> Boolean.TRUE)
                    .orElseThrow(),
                    "queue publication must never expose an unready slot");

            releaseAction.countDown();
            assertTrue(committed.get(5, TimeUnit.SECONDS));
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
        } finally {
            releaseAction.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
            admission.close();
        }
    }

    @Test
    void itemPublicationRequiresAnExactAdmissionMutation() {
        Registered registered = registerItem(106L);
        boolean[] publicationCalled = new boolean[1];

        assertFalse(lifecycle.commitItemForPublication(
                registered.item(), () -> {
                    publicationCalled[0] = true;
                    return true;
                }));
        assertFalse(publicationCalled[0]);
    }

    @Test
    void declinedPublicationClearsTheProvisionalSlotBinding() {
        Registered registered = registerItem(111L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertFalse(lifecycle.commitItemForPublication(
                    registered.item(), () -> false));
            synchronized (slot) {
                assertNull(slot.activeItem());
            }
        }
    }

    @Test
    void throwingPublicationPreservesTheFailureAndClearsTheBinding() {
        Registered registered = registerItem(121L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        IllegalStateException expected =
                new IllegalStateException("queue publication failed");

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            IllegalStateException actual = assertThrows(
                    IllegalStateException.class,
                    () -> lifecycle.commitItemForPublication(
                            registered.item(), () -> {
                                throw expected;
                            }));
            assertSame(expected, actual);
            synchronized (slot) {
                assertNull(slot.activeItem());
            }
        }
    }

    @Test
    void slotIsDeliveryReadyBeforeQueuePublicationCanBecomeVisible() {
        Registered registered = registerItem(131L);
        boolean[] preparedBeforeResolution = new boolean[1];

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), () -> {
                        preparedBeforeResolution[0] = lifecycle.prepareIfOwned(
                                registered.item(), () -> Boolean.TRUE)
                                .isPresent();
                        return true;
                    }));
            assertTrue(preparedBeforeResolution[0],
                    "a queue-visible item must never race an unready slot");
            assertTrue(lifecycle.prepareIfOwned(
                            registered.item(), () -> Boolean.TRUE)
                    .orElseThrow());
        }
    }

    @Test
    void cancellationEntersTheSlotButWaitsForPublicationResolution()
            throws Exception {
        Registered registered = registerItem(141L);
        CountDownLatch publicationEntered = new CountDownLatch(1);
        CountDownLatch releasePublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        lifecycle.commitItemForPublication(
                                registered.item(), () -> {
                                    publicationEntered.countDown();
                                    await(releasePublication);
                                    return false;
                                }));
                assertTrue(publicationEntered.await(5, TimeUnit.SECONDS));

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancelRequest(
                                registered.item().requestId(),
                                0L,
                                CancelReason.CLIENT_CANCELLED));
                RequestState pending =
                        cancellation.get(5, TimeUnit.SECONDS);
                assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                        pending.state());
                assertFalse(publication.isDone(),
                        "cancellation must not resolve endpoint publication");

                releasePublication.countDown();
                assertFalse(publication.get(5, TimeUnit.SECONDS));
            } finally {
                releasePublication.countDown();
                operations.shutdownNow();
            }
        }

        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(
                        registered.item().requestId(), 0L).state());
    }

    @Test
    void cancellationAfterQueuePointOfNoReturnRemovesTheExactItemOnce()
            throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(151L, prefill);
        CountDownLatch queuePublished = new CountDownLatch(1);
        CountDownLatch returnFromPublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        lifecycle.commitItemForPublication(
                                registered.item(), () -> {
                                    queuePublished.countDown();
                                    await(returnFromPublication);
                                    return true;
                                }));
                assertTrue(queuePublished.await(5, TimeUnit.SECONDS));

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancelRequest(
                                registered.item().requestId(),
                                0L,
                                CancelReason.CLIENT_CANCELLED));
                assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                        cancellation.get(5, TimeUnit.SECONDS).state());

                returnFromPublication.countDown();
                assertTrue(publication.get(5, TimeUnit.SECONDS));
            } finally {
                returnFromPublication.countDown();
                operations.shutdownNow();
            }
        }

        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        verify(prefill).removeQueued(
                eq(registered.item()), anyString());
    }

    @Test
    void deliveryClaimKeepsEndpointHandoffAndSlotClaimAtomic()
            throws Exception {
        Registered registered = registerItem(201L);
        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), () -> true));
        }
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        CountDownLatch transferEntered = new CountDownLatch(1);
        CountDownLatch releaseTransfer = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (slot) {
                contenderEntered.countDown();
            }
        }, "try-commit-slot-contender");

        try {
            Future<RequestRegistry.DeliveryClaim> committed = owner.submit(() ->
                    RequestLifecycleTestSupport.claimRoute(lifecycle,
                            registered.item(),
                            () -> {
                                assertTrue(Thread.holdsLock(slot));
                                transferEntered.countDown();
                                await(releaseTransfer);
                                return true;
                            }));

            assertTrue(transferEntered.await(5, TimeUnit.SECONDS));
            contender.start();
            assertTrue(contenderStarted.await(5, TimeUnit.SECONDS));
            awaitCondition(() -> contender.getState() == Thread.State.BLOCKED
                    || contenderEntered.getCount() == 0L);

            assertEquals(Thread.State.BLOCKED, contender.getState());
            assertEquals(1L, contenderEntered.getCount(),
                    "another slot operation must not enter during endpoint transfer");

            releaseTransfer.countDown();
            RequestRegistry.DeliveryClaim claim = committed.get(5, TimeUnit.SECONDS);
            assertNotNull(claim);
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
            assertEquals(0L, contenderEntered.getCount());

            lifecycle.complete(
                    claim, DeliveryResult.delivered());
            assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        } finally {
            releaseTransfer.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
        }
    }

    @Test
    void failedEndpointTransferLeavesTheSlotUnclaimed() {
        Registered rejected = registerItem(202L);
        bind(lifecycle, rejected);

        assertThrows(IllegalStateException.class, () -> RequestLifecycleTestSupport.claimRoute(lifecycle,
                rejected.item(),
                () -> false));
        assertQueuedWithoutClaim(rejected.item().requestId());

        Registered failed = registerItem(203L);
        bind(lifecycle, failed);
        IllegalStateException expected = new IllegalStateException(
                "synthetic endpoint failure");
        assertSame(expected, assertThrows(IllegalStateException.class,
                () -> RequestLifecycleTestSupport.claimRoute(lifecycle,
                        failed.item(),
                        () -> {
                            throw expected;
                        })));
        assertQueuedWithoutClaim(failed.item().requestId());
    }

    @Test
    void batchClaimIsCompleteWhenTryCommitReturns() {
        Registered registered = registerItem(204L);
        bind(lifecycle, registered);

        RequestRegistry.DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(lifecycle,
                registered.item(),
                701L,
                () -> true);

        assertNotNull(claim);
        RequestState snapshot = lifecycle.getRequestState(
                registered.item().requestId(), 0L);
        assertEquals(RequestState.Phase.DISPATCHING, snapshot.state());
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE,
                snapshot.deliveryClaimKind());
        assertEquals(701L, snapshot.batchId());
        assertTrue(lifecycle.requestSlot(registered.item().requestId())
                .getBatchEnqueueStartedAtMs() > 0L);
    }

    @Test
    void terminalEngineEvidenceMakesLateDeliveryCallbacksHarmless() throws Exception {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        Registered registered = registerItem(207L, endpoint);
        bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(
                lifecycle, registered.item(), 703L, () -> true);
        assertNotNull(claim);
        RequestSlot original = lifecycle.requestSlot(207L);

        new EndpointEventProjector(lifecycle).onPrefillStatus(endpoint, RoleType.PDFUSION,
                List.of(PrefillState.WorkerStatusFact.terminal(
                        registered.item(), PrefillState.WorkerStatusFact.Kind.COMPLETED, 0L)));
        Response terminal = registered.future().get(5L, TimeUnit.SECONDS);
        assertTrue(terminal.isSuccess());

        lifecycle.beginDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        lifecycle.complete(claim, DeliveryResult.delivered());
        assertSame(terminal, registered.future().join());
        assertTrue(lifecycle.removeExactTombstone(original, Long.MAX_VALUE));

        Registered replacement = registerItem(207L, endpoint);
        bind(lifecycle, replacement);
        lifecycle.beginDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        lifecycle.complete(claim, DeliveryResult.failed(new IllegalStateException("late RPC failure")));
        assertFalse(replacement.future().isDone());
        assertQueuedWithoutClaim(207L);
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void deliveryStartReconcilesEarlyDecodeAcceptanceWithoutFabricatingBatchAck(
            boolean batch, boolean acceptanceBeforeClaim)
            throws Exception {
        if (!batch) {
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation = new DecodeEndpoint.ReservationHandle(1L, 208L, 1L);
        when(decode.isReservationAccepted(reservation)).thenReturn(true);
        BalanceContext context = context(208L);
        CompletableFuture<Response> future = lifecycle.register(context);
        ScheduledRequest item = new ScheduledRequest(context, future, new Response(), null, null,
                null, decode, reservation, System.currentTimeMillis());
        bind(lifecycle, new Registered(item, future));
        if (acceptanceBeforeClaim) {
            lifecycle.onDecodeAccepted(decode, reservation);
        }
        RequestRegistry.DeliveryClaim claim = batch
                ? lifecycle.tryClaimBatchDelivery(item, 704L, () -> true)
                : lifecycle.tryClaimRouteDelivery(item, () -> true);
        assertNotNull(claim);
        WorkSnapshot precedingWork = new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);

        if (batch) {
            lifecycle.beginDelivery(claim, precedingWork, 30_000L);
            assertFalse(future.isDone(), "Decode acceptance cannot replace the EnqueueBatch ACK");
        } else {
            lifecycle.beginRouteDelivery(claim, precedingWork, 30_000L);
        }
        RequestSlot slot = lifecycle.requestSlot(208L);
        synchronized (slot) {
            assertTrue(slot.decodeOwnsRequest());
            assertTrue(slot.decisionDeadlineAtMs().isEmpty(), "accepted Decode needs no observation deadline");
        }
        if (batch) {
            lifecycle.complete(claim, DeliveryResult.delivered());
        }
        assertTrue(future.get(5L, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void schedulingDeadlineCannotCancelAfterBatchDeliveryPointOfNoReturn() {
        Registered registered = registerItem(206L);
        bind(lifecycle, registered);

        RequestRegistry.DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(lifecycle,
                registered.item(),
                702L,
                () -> true);
        assertNotNull(claim);

        RequestState afterDeadline = lifecycle.cancelRequest(
                registered.item().requestId(),
                0L,
                CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestState.Phase.DISPATCHING, afterDeadline.state(),
                "the committed delivery claim must own the deadline race");

        lifecycle.complete(
                claim, DeliveryResult.delivered());
        assertTrue(registered.future().join().isSuccess());
    }

    @Test
    void acknowledgedDeliveryKeepsDecisionObservationDeadline()
            throws Exception {
        Registered registered = registerItem(
                205L, null, mock(DecodeEndpoint.class));
        bindRoute(lifecycle, registered);

        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(),
                () -> true);
        assertNotNull(claim);
        lifecycle.beginRouteDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 10L);

        assertTrue(registered.future().join().isSuccess());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState(205L, 0L).state());
        // The observation window includes the 10-second handoff grace and is
        // independent of worker-status RPC timing and request inactivity.
        assertTimeoutPreemptively(Duration.ofSeconds(15), () -> {
            while (!lifecycle.getRequestState(205L, 0L).detail().contains("SUSPECTED_LOST")) {
                Thread.sleep(5L);
            }
        });
    }

    private void assertQueuedWithoutClaim(long requestId) {
        RequestState snapshot = lifecycle.getRequestState(
                requestId, 0L);
        assertEquals(RequestState.Phase.QUEUED, snapshot.state());
        assertEquals(DeliveryClaimKind.NONE, snapshot.deliveryClaimKind());
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, null);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefillEndpoint) {
        return registerItem(requestId, prefillEndpoint, null);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefillEndpoint,
            DecodeEndpoint decodeEndpoint) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context);
        ScheduledRequest item = new ScheduledRequest(
                context,
                future,
                new Response(),
                null,
                null,
                prefillEndpoint,
                decodeEndpoint,
                null,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }
}
