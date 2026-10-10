package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.commitRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Canonical request-generation ownership tests, independent of the facade. */
class RequestRegistryTest {

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
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        BalanceContext context = context(101L);
        CompletableFuture<Response> canonical = lifecycle.register(context);

        CompletableFuture<Response> duplicate = lifecycle.register(context(101L));

        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicate.join().getCode());
        assertSame(canonical, lifecycle.requestSlot(101L).future());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void terminalRecordPreservesIdentityWithoutRetainingRequestContext() throws Exception {
        WeakReference<BalanceContext> contextReference = cancelAndReferenceContext(103L);
        RequestSlot terminal = lifecycle.requestSlot(103L);
        assertEquals(RequestState.Phase.CANCELLED, terminal.snapshot().state());

        for (int attempt = 0; attempt < 20 && !contextReference.refersTo(null); attempt++) {
            System.gc();
            Thread.sleep(50L);
        }

        assertTrue(contextReference.refersTo(null), "terminal identity must not retain the request payload");
        assertSame(terminal, lifecycle.requestSlot(103L));
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                lifecycle.register(context(103L)).join().getCode());
    }

    @Test
    void globalWaitingRequestsHaveNoQuantityAdmissionLimit() {
        var low = context(1L);
        low.setSchedulingMetadata(SchedulingMetadata.explicit(10, Long.MAX_VALUE));
        var waiting = lifecycle.register(low);
        for (long id = 2; id <= 1001; id++) {
            var high = context(id);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            assertFalse(lifecycle.register(high).isDone());
        }
        assertFalse(waiting.isDone(), "higher priority arrivals must not evict waiting requests");
        assertEquals(1001, lifecycle.liveRequestCount());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                lifecycle.register(context(1L)).join().getCode());
        for (long id = 1; id <= 1001; id++) {
            lifecycle.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        }
        assertEquals(0, lifecycle.liveRequestCount());
    }

    @Test
    void publicQueriesOwnTheirLockAndPrivateDecisionsStillRequireIt() {
        lifecycle.register(context(102L));
        RequestSlot slot = lifecycle.requestSlot(102L);
        assertNull(slot.activeItem());
        assertTrue(slot.isOpen());
        assertTrue(slot.isLiveGeneration());
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                        slot, "recordCancellationLocked", CancelReason.CLIENT_CANCELLED, "client cancelled"));
        assertTrue(failure.getMessage().contains("requires slot lock"));
        assertEquals(RequestState.Phase.QUEUED, slot.snapshot().state());
    }

    @Test
    void deliveredRequestsHaveNoExtraGlobalQuantityGate() {
        for (long id = 201; id <= 401; id++) {
            Registered registered = registerItem(id);
            assertEquals(PlacementResult.Status.SUCCESS,
                    commitRoute(lifecycle, registered));
            assertNotNull(RequestLifecycleTestSupport.claimRoute(
                    lifecycle, registered.item(), () -> true));
        }
        assertEquals(201, lifecycle.liveRequestCount());
    }

    @Test
    void admissionHandleDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L));
        AdmissionHandle scope =
                lifecycle.claimAdmissionHandle(301L, future);
        assertNotNull(scope);

        RequestState requested = lifecycle.cancelRequest(
                301L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(301L, 0L).state());
    }

    @Test
    void repeatedCancellationDuringAdmissionKeepsTheFirstCause() throws Exception {
        CompletableFuture<Response> future = lifecycle.register(context(303L));
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(303L, future);
        assertNotNull(admission);

        lifecycle.cancelRequest(303L, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancelRequest(303L, 0L, CancelReason.DEADLINE_EXCEEDED);
        assertFalse(future.isDone());

        admission.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(303L, 0L).state());
        assertEquals(0, lifecycle.liveRequestCount());
    }

    @Test
    void admissionFailurePreservesAnEarlierCancellation() throws Exception {
        CompletableFuture<Response> future = lifecycle.register(context(304L));
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(304L, future);
        assertNotNull(admission);

        lifecycle.cancelRequest(304L, 0L, CancelReason.CLIENT_CANCELLED);
        admission.terminate(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        admission.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(304L, 0L).state());
        assertEquals(0, lifecycle.liveRequestCount());
    }

    @Test
    void queueDecisionResponsePublishesOutsideTheDecisionCaller() throws Exception {
        CompletableFuture<Response> future = lifecycle.register(context(302L));
        CountDownLatch published = new CountDownLatch(1);
        AtomicReference<String> callbackThread = new AtomicReference<>();
        future.thenAccept(response -> {
            callbackThread.set(Thread.currentThread().getName());
            published.countDown();
        });

        Response rejection = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        assertTrue(lifecycle.publishDecisionResponseAsync(
                302L, future, rejection));
        assertTrue(published.await(5, TimeUnit.SECONDS));
        assertNotEquals(Thread.currentThread().getName(), callbackThread.get());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void shutdownGateWaitsForTheExactAdmissionHandleAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                lifecycle.register(context(401L));
        AdmissionHandle held =
                lifecycle.claimAdmissionHandle(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(lifecycle::closeAdmissionAndAwaitMutations);
            awaitCondition(lifecycle::isShuttingDown);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    lifecycle.register(context(402L));
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.close();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = lifecycle.register(context(501L));

        assertNull(lifecycle.cancelRequest(
                999L, 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancelRequest(
                501L, 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState exact = lifecycle.cancelRequest(
                501L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void publishedQueueDeadlineReleasesLocalReservationWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered));
        RequestSlot slot = lifecycle.requestSlot(602L);
        synchronized (slot) {
            org.springframework.test.util.ReflectionTestUtils.<RequestSlot.EngineObservation>invokeMethod(slot, "applyPrefillStatusLocked", registered.item().prefillEp(), org.flexlb.dao.route.RoleType.PREFILL,
                    org.flexlb.balance.endpoint.PrefillState.WorkerStatusFact.active(registered.item()), System.currentTimeMillis());
        }
        lifecycle.cancelRequest(602L, 0L, CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(602L, 0L).state());
        verify(registered.item().decodeEp()).release(
                registered.item().decodeReservation(),
                DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void expiredArrivalDoesNotDisturbTheWaitingRequests() throws Exception {
        var low = lifecycle.register(context(1));
        var expired = context(2);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1L));
        // QUEUE expiry before placement is admission capacity exhaustion (8431).
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                lifecycle.register(expired).get(5, TimeUnit.SECONDS).getCode());
        assertFalse(low.isDone());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void concurrentGlobalAdmissionRetainsEveryUniqueRequest() throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<CompletableFuture<Response>>> futures = new ArrayList<>();
            for (long id = 1; id <= 128; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> lifecycle.register(context(requestId))));
            }
            for (var future : futures) {
                assertFalse(future.get(5, TimeUnit.SECONDS).isDone());
            }
            assertEquals(128, lifecycle.liveRequestCount());
        }
    }

    @Test
    void oldDeliveryAndPreemptionCapabilitiesCannotReachAReusedRequestId() {
        Registered registered = registerItem(703L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        RequestSlot old = lifecycle.requestSlot(703L);
        DeliveryClaim delivery = RequestLifecycleTestSupport.claimBatchWithoutPrediction(lifecycle, registered.item(), 17L, () -> true);
        assertNotNull(delivery);
        PreemptionRegistration preemption = lifecycle.tryClaim(703L, 1L, 19L, "victim").orElseThrow();

        old.expireInactiveRequest(
                RequestLifecycleTestSupport.<Long>inspect(old, "inactivityExpiresAtMsLocked"));
        registered.future().join();
        assertTrue(lifecycle.removeExactTerminalRecord(old, Long.MAX_VALUE));
        CompletableFuture<Response> replacement = lifecycle.register(context(703L));

        delivery.complete(org.flexlb.balance.delivery.DeliveryResult.delivered());
        assertFalse(preemption.completePreemption("late engine cancellation"));
        assertFalse(preemption.release());
        assertNull(old.cancelRequest(0L, CancelReason.CLIENT_CANCELLED));
        assertFalse(replacement.isDone());
        assertEquals(RequestState.Phase.QUEUED, lifecycle.getRequestState(703L, 0L).state());
    }

    @Test
    void invalidBatchIdentityCannotTransferEndpointOwnership() {
        Registered registered = registerItem(704L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var transaction = mock(BatchDeliveryStrategy.BatchTransaction.class);
        when(transaction.batchId()).thenReturn(0L);

        assertThrows(IllegalArgumentException.class,
                () -> lifecycle.claimBatchDelivery(registered.item(), transaction));

        org.mockito.Mockito.verify(transaction, org.mockito.Mockito.never()).transferToEndpoint(registered.item());
        assertEquals(RequestState.Phase.QUEUED, lifecycle.getRequestState(704L, 0L).state());
        assertFalse(registered.future().isDone());
    }

    @Test
    void invalidResultDoesNotConsumeDeliveryAndDuplicateResultCannotChangeItsOutcome() throws Exception {
        Registered registered = registerItem(705L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(lifecycle, registered.item(), 23L, () -> true);
        assertNotNull(claim);

        assertThrows(NullPointerException.class, () -> claim.complete(null));
        assertEquals(RequestState.Phase.DISPATCHING, lifecycle.getRequestState(705L, 23L).state());
        claim.complete(org.flexlb.balance.delivery.DeliveryResult.delivered());
        assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertThrows(IllegalStateException.class, () -> claim.complete(
                org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("duplicate failure"))));

        assertEquals(RequestState.Phase.ACKNOWLEDGED, lifecycle.getRequestState(705L, 23L).state());
        org.mockito.Mockito.verify(registered.item().decodeEp(), org.mockito.Mockito.never())
                .release(registered.item().decodeReservation(), DecodeEndpoint.ReleaseReason.NOT_SENT);
    }

    private WeakReference<BalanceContext> cancelAndReferenceContext(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context);
        lifecycle.cancelRequest(requestId, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
        return new WeakReference<>(context);
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        return new Registered(
                new ScheduledRequest(
                        context,
                        future,
                        new Response(),
                        null,
                        null,
                        null,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
    }
}
