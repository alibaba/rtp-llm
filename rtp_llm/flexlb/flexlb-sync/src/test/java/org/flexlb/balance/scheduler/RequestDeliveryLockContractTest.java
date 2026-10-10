package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.balance.scheduler.RequestProtocolTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.await;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.bind;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.bindRoute;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Executable documentation for the RequestContext delivery lock boundary. */
class RequestDeliveryLockContractTest {

    private FlexlbConfig config;
    private AbstractRequestScheduler lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                configService,
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
            lifecycle.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
        }
    }

    @Test
    void firstBatchMemberPreparesAtomicallyAgainstConcurrentCancellation() throws Exception {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        Registered registered = registerItem(809L, endpoint);
        bind(lifecycle, registered);
        var reservation = mock(PrefillState.BatchReservation.class);
        var submission = mock(DefaultBatchDispatcher.PreparedSubmission.class);
        CountDownLatch initializing = new CountDownLatch(1);
        CountDownLatch cancellationStarted = new CountDownLatch(1);
        ExecutorService operations = Executors.newSingleThreadExecutor();
        Future<RequestState> cancellation = operations.submit(() -> {
            await(initializing);
            cancellationStarted.countDown();
            return lifecycle.cancel(809L, 0L, CancelReason.CLIENT_CANCELLED);
        });
        org.mockito.Mockito.when(endpoint.reserveBatch(org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.anyInt()))
                .thenAnswer(invocation -> {
                    assertTrue(Thread.holdsLock(registered.item().ctx()));
                    assertFalse(cancellation.isDone());
                    return new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, reservation);
                });
        BatchDeliveryStrategy strategy = new BatchDeliveryStrategy(() -> {
            assertTrue(Thread.holdsLock(registered.item().ctx()));
            initializing.countDown();
            await(cancellationStarted);
            return org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted(submission);
        }, () -> 901L, mock(DeliveryMetricsReporter.class));
        try {
            try (var transaction = strategy.prepare(List.of(registered.item()),
                    DeliveryStrategyTestSupport.EVALUATOR, java.util.OptionalLong.empty())) {
                assertEquals(List.of(registered.item()), transaction.items());
                assertEquals(RequestState.Phase.CANCEL_REQUESTED, cancellation.get(5, TimeUnit.SECONDS).state());
                assertNull(lifecycle.claimDelivery(registered.item(), DeliveryClaimKind.BATCH_ENQUEUE,
                        901L, RequestProtocolTestSupport.handoff(() -> { throw new AssertionError("cancelled request transferred resources"); })));
            }
            verify(endpoint).rollbackReservation(reservation);
            verify(submission).close();
            org.mockito.Mockito.verify(submission, org.mockito.Mockito.never()).submit(org.mockito.ArgumentMatchers.any());
        } finally {
            initializing.countDown();
            operations.shutdownNow();
        }
    }

    @Test
    void itemPublicationDoesNotOwnTheExactContextMonitor()
            throws Exception {
        Registered registered = registerItem(101L);
        RequestContext requestContext = lifecycle.findRequestContext(registered.item().requestId());
        AdmissionHandle admission =
                lifecycle.claimAdmissionHandle(
                        registered.item().requestId(), registered.future());
        assertNotNull(admission);
        CountDownLatch actionEntered = new CountDownLatch(1);
        CountDownLatch releaseAction = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (requestContext) {
                contenderEntered.countDown();
            }
        }, "commit-inflight-context-contender");

        try {
            Future<Boolean> committed = owner.submit(() ->
                    (lifecycle.commitRoute(
                            registered.item(), RequestProtocolTestSupport.publication(() -> {
                                actionEntered.countDown();
                                assertFalse(Thread.holdsLock(requestContext));
                                await(releaseAction);
                                return true;
                            })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));

            assertTrue(actionEntered.await(5, TimeUnit.SECONDS));
            contender.start();
            assertTrue(contenderStarted.await(5, TimeUnit.SECONDS));
            assertTrue(contenderEntered.await(5, TimeUnit.SECONDS),
                    "context operations must proceed during endpoint publication");
            synchronized (requestContext) {
                assertSame(registered.item(), requestContext.activeRoute());
            }
            assertFalse(RequestProtocolTestSupport.prepareMember(lifecycle, registered.item()),
                    "ROUTING cannot be claimed during queue publication");

            releaseAction.countDown();
            assertTrue(committed.get(5, TimeUnit.SECONDS));
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
        } finally {
            releaseAction.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
            admission.finish();
        }
    }

    @Test
    void itemPublicationRequiresAnExactAdmissionHandle() {
        Registered registered = registerItem(106L);
        boolean[] publicationCalled = new boolean[1];

        assertFalse((lifecycle.commitRoute(
                registered.item(), RequestProtocolTestSupport.publication(() -> {
                    publicationCalled[0] = true;
                    return true;
                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        assertFalse(publicationCalled[0]);
    }

    @Test
    void declinedPublicationClearsTheProvisionalContextBinding() {
        Registered registered = registerItem(111L);
        RequestContext requestContext = lifecycle.findRequestContext(registered.item().requestId());

        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertFalse((lifecycle.commitRoute(
                    registered.item(), RequestProtocolTestSupport.publication(() -> false)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
            synchronized (requestContext) {
                assertNull(requestContext.activeRoute());
            }
        }
    }

    @Test
    void throwingPublicationPreservesTheFailureAndClearsTheBinding() {
        Registered registered = registerItem(121L);
        RequestContext requestContext = lifecycle.findRequestContext(registered.item().requestId());
        IllegalStateException expected =
                new IllegalStateException("queue publication failed");

        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            IllegalStateException actual = assertThrows(
                    IllegalStateException.class,
                    () -> lifecycle.commitRoute(
                            registered.item(), RequestProtocolTestSupport.publication(() -> {
                                throw expected;
                            })));
            assertSame(expected, actual);
            synchronized (requestContext) {
                assertNull(requestContext.activeRoute());
            }
        }
    }

    @Test
    void queueVisibleItemWaitsForRouteCommitBeforeDelivery() {
        Registered registered = registerItem(131L);
        boolean[] preparedBeforeResolution = new boolean[1];

        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((lifecycle.commitRoute(
                    registered.item(), RequestProtocolTestSupport.publication(() -> {
                        preparedBeforeResolution[0] = RequestProtocolTestSupport.prepareMember(lifecycle, registered.item());
                        return true;
                    })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
            assertFalse(preparedBeforeResolution[0],
                    "a ROUTING item cannot be delivered before route commit");
            assertTrue(RequestProtocolTestSupport.prepareMember(lifecycle, registered.item()));
        }
    }

    @Test
    void cancellationEntersTheContextButWaitsForPublicationResolution()
            throws Exception {
        Registered registered = registerItem(141L);
        CountDownLatch publicationEntered = new CountDownLatch(1);
        CountDownLatch releasePublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion4 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        (lifecycle.commitRoute(
                                registered.item(), RequestProtocolTestSupport.publication(() -> {
                                    publicationEntered.countDown();
                                    await(releasePublication);
                                    return false;
                                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                assertTrue(publicationEntered.await(5, TimeUnit.SECONDS));

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancel(
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
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(
                        registered.item().requestId(), 0L).state());
    }

    @Test
    void cancellationAfterQueuePointOfNoReturnRemovesTheExactItemOnce()
            throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(151L, prefill);
        PrefillState state = new PrefillState(new java.util.concurrent.locks.ReentrantLock(),
                org.flexlb.balance.endpoint.PrefillActiveIndex.ordered(1, WorkerBatcher.PRIORITY_QUEUE_ORDER));
        java.util.concurrent.atomic.AtomicInteger releases = new java.util.concurrent.atomic.AtomicInteger();
        when(prefill.releaseRequest(registered.item())).thenAnswer(invocation -> {
            assertFalse(Thread.holdsLock(registered.item().ctx()),
                    "resource release must run outside the request monitor");
            assertFalse(state.ownershipLock().isHeldByCurrentThread(),
                    "resource release must enter outside the queue ownership lock");
            boolean released = org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(state, registered.item());
            if (released) { releases.incrementAndGet(); }
            assertFalse(state.ownershipLock().isHeldByCurrentThread());
            return released;
        });
        CountDownLatch queuePublished = new CountDownLatch(1);
        CountDownLatch returnFromPublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion5 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        (lifecycle.commitRoute(
                                registered.item(), RequestProtocolTestSupport.publication(() -> {
                                    state.ownershipLock().lock();
                                    try {
                                        assertTrue(state.enqueueActiveLocked(registered.item(), 0L));
                                    } finally { state.ownershipLock().unlock(); }
                                    queuePublished.countDown();
                                    await(returnFromPublication);
                                    return true;
                                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                assertTrue(queuePublished.await(5, TimeUnit.SECONDS));
                assertEquals(List.of(registered.item()), state.captureQueue(1).items());
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancel(
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
        awaitCondition(() -> SchedulerTestSupport.repository(lifecycle).getRequestState(
                registered.item().requestId(), 0L).state() == RequestState.Phase.CANCELLED);
        verify(prefill).releaseRequest(registered.item());
        assertEquals(1, releases.get());
        assertTrue(state.captureQueue(1).items().isEmpty());
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(state, registered.item()),
                "late exact cleanup must be idempotent");
    }

    @Test
    void deliveryClaimKeepsEndpointHandoffAndContextClaimAtomic()
            throws Exception {
        Registered registered = registerItem(201L);
        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(),
                             registered.future()); var admissionCompletion6 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((lifecycle.commitRoute(
                    registered.item(), RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
        RequestContext requestContext = lifecycle.findRequestContext(registered.item().requestId());
        CountDownLatch transferEntered = new CountDownLatch(1);
        CountDownLatch releaseTransfer = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (requestContext) {
                contenderEntered.countDown();
            }
        }, "try-commit-context-contender");

        try {
            Future<DeliveryClaim> committed = owner.submit(() ->
                    RequestProtocolTestSupport.claimRouteWithoutPrediction(lifecycle,
                            registered.item(),
                            () -> {
                                assertTrue(Thread.holdsLock(requestContext));
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
                    "another context operation must not enter during endpoint transfer");

            releaseTransfer.countDown();
            DeliveryClaim claim = committed.get(5, TimeUnit.SECONDS);
            assertNotNull(claim);
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
            assertEquals(0L, contenderEntered.getCount());

            lifecycle.publishRoute(claim,
                    new WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L), 30_000L);
            assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        } finally {
            releaseTransfer.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
        }
    }

    @Test
    void failedEndpointTransferLeavesTheContextUnclaimed() {
        Registered rejected = registerItem(202L);
        bind(lifecycle, rejected);

        assertThrows(IllegalStateException.class, () -> RequestProtocolTestSupport.claimRoute(lifecycle,
                rejected.item(),
                () -> false));
        assertQueuedWithoutClaim(rejected.item().requestId());

        Registered failed = registerItem(203L);
        bind(lifecycle, failed);
        IllegalStateException expected = new IllegalStateException(
                "synthetic endpoint failure");
        assertSame(expected, assertThrows(IllegalStateException.class,
                () -> RequestProtocolTestSupport.claimRoute(lifecycle,
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

        DeliveryClaim claim = RequestProtocolTestSupport.claimBatch(lifecycle,
                registered.item(),
                701L,
                () -> true);

        assertNotNull(claim);
        RequestState snapshot = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(
                registered.item().requestId(), 0L);
        assertEquals(RequestState.Phase.DISPATCHING, snapshot.state());
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE,
                snapshot.deliveryClaimKind());
        assertEquals(701L, snapshot.batchId());
        assertTrue(((Long) org.springframework.test.util.ReflectionTestUtils.getField(
                lifecycle.findRequestContext(registered.item().requestId()), "batchEnqueueStartedAtMs")) > 0L);
    }

    @Test
    void anItemTransfersDeliveryOwnershipOnlyOnce() {
        Registered registered = registerItem(805L);
        bind(lifecycle, registered);
        DeliveryClaim first = RequestProtocolTestSupport.claimBatchWithoutPrediction(
                lifecycle, registered.item(), 701L, () -> true);
        assertNotNull(first);

        assertNull(lifecycle.claimDelivery(registered.item(), DeliveryClaimKind.BATCH_ENQUEUE, 702L, RequestProtocolTestSupport.handoff(() -> { throw new AssertionError("second batch transfer"); })));
        assertNull(lifecycle.claimDelivery(registered.item(), DeliveryClaimKind.ROUTE_DECISION, 0L, RequestProtocolTestSupport.handoff(() -> { throw new AssertionError("route transfer after batch claim"); })));
        assertEquals(701L, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(805L, 701L).batchId());
        assertFalse(registered.future().isDone());

        lifecycle.completeDelivery(first, DeliveryResult.delivered());
        assertTrue(registered.future().join().isSuccess());
    }

    @Test
    void terminalEngineEvidenceMakesLateDeliveryCallbacksHarmless() throws Exception {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        Registered registered = registerItem(207L, endpoint);
        bind(lifecycle, registered);
        DeliveryClaim claim = RequestProtocolTestSupport.claimBatch(
                lifecycle, registered.item(), 703L, () -> true);
        assertNotNull(claim);
        RequestContext original = lifecycle.findRequestContext(207L);

        lifecycle.onPrefillStatus(original, endpoint, RoleType.PDFUSION, PrefillState.PrefillRequestStatus.terminal(
                        registered.item(), PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L));
        Response terminal = registered.future().get(5L, TimeUnit.SECONDS);
        assertTrue(terminal.isSuccess());
        assertEquals(RequestState.Phase.COMPLETED, original.snapshot().state());

        lifecycle.setDeliveryPrediction(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
        assertSame(terminal, registered.future().join());
        assertEquals(RequestState.Phase.COMPLETED, original.snapshot().state());
        lifecycle.runtime.continuations().awaitIdle();
        RequestState retiredRecord0 = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(original.getRequestId(), 0L);
        assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).removeExactTerminal(org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(lifecycle, retiredRecord0), Long.MAX_VALUE));

        Registered replacement = registerItem(207L, endpoint);
        bind(lifecycle, replacement);
        lifecycle.setDeliveryPrediction(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        assertThrows(IllegalStateException.class, () -> claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.notSent(new IllegalStateException("late RPC failure"))));
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
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 208L, 1L);
        when(decode.isAcceptedByEngine(reservation)).thenReturn(true);
        RequestContext context = context(208L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                null, decode, reservation, System.currentTimeMillis());
        bind(lifecycle, new Registered(item, future));
        if (acceptanceBeforeClaim) {
            RequestProtocolTestSupport.applyDecodeStatus(lifecycle, decode, DecodeResources.DecodeRequestStatus.active(reservation));
        }
        DeliveryClaim claim = batch
                ? RequestProtocolTestSupport.claimBatchWithoutPrediction(lifecycle, item, 704L, () -> true)
                : RequestProtocolTestSupport.claimRouteWithoutPrediction(lifecycle, item, () -> true);
        assertNotNull(claim);
        WorkSnapshot precedingWork = new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);

        if (batch) {
            lifecycle.setDeliveryPrediction(claim, precedingWork, 30_000L);
            assertFalse(future.isDone(), "Decode acceptance cannot replace the EnqueueBatch ACK");
        } else {
            lifecycle.publishRoute(claim, precedingWork, 30_000L);
        }
        RequestContext requestContext = lifecycle.findRequestContext(208L);
        synchronized (requestContext) {
            assertTrue(requestContext.decodeAccepted());
            assertTrue(requestContext.decisionDeadlineAtMs().isEmpty(), "accepted Decode needs no observation deadline");
        }
        if (batch) {
            claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
        }
        assertTrue(future.get(5L, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void schedulingDeadlineCannotCancelAfterBatchDeliveryPointOfNoReturn() {
        Registered registered = registerItem(206L);
        bind(lifecycle, registered);

        DeliveryClaim claim = RequestProtocolTestSupport.claimBatch(lifecycle,
                registered.item(),
                702L,
                () -> true);
        assertNotNull(claim);

        RequestState afterDeadline = lifecycle.cancel(
                registered.item().requestId(),
                0L,
                CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestState.Phase.DISPATCHING, afterDeadline.state(),
                "the committed delivery claim must own the deadline race");

        claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
        assertTrue(registered.future().join().isSuccess());
        assertThrows(IllegalStateException.class,
                () -> claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.notSent(new IllegalStateException("duplicate callback"))));
        assertTrue(registered.future().join().isSuccess());
    }

    @Test
    void acknowledgedDeliveryKeepsDecisionObservationDeadline()
            throws Exception {
        Registered registered = registerItem(
                205L, null, RequestProtocolTestSupport.decodeEndpoint());
        bindRoute(lifecycle, registered);

        DeliveryClaim claim = RequestProtocolTestSupport.claimRouteWithoutPrediction(lifecycle,
                registered.item(),
                () -> true);
        assertNotNull(claim);
        assertThrows(IllegalStateException.class, () -> claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered()));
        assertFalse(registered.future().isDone(), "a batch callback cannot publish a route");
        WorkSnapshot precedingWork = new WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L);
        lifecycle.publishRoute(claim, precedingWork, 10L);
        assertThrows(IllegalStateException.class, () -> lifecycle.publishRoute(claim, precedingWork, 10L));

        assertTrue(registered.future().join().isSuccess());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(205L, 0L).state());
        // The observation window includes the 10-second handoff grace and is
        // independent of worker-status RPC timing and request inactivity.
        assertTimeoutPreemptively(Duration.ofSeconds(15), () -> {
            while (!org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(205L, 0L).detail().contains("SUSPECTED_LOST")) {
                Thread.sleep(5L);
            }
        });
    }

    private void assertQueuedWithoutClaim(long requestId) {
        RequestState snapshot = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(
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
        RequestContext context = context(requestId);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context),
                new Response(),
                null,
                null,
                prefillEndpoint,
                decodeEndpoint,
                null,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private RequestContext context(long requestId) {
        return RequestProtocolTestSupport.context(config, requestId);
    }
}
