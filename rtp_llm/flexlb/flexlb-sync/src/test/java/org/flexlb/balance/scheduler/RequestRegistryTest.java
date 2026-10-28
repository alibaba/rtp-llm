package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
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
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Canonical request-generation ownership tests, independent of the facade. */
class RequestRegistryTest {

    private FlexlbConfig config;
    private RequestRegistry lifecycle;
    private EngineCancelChannel engineCancelChannel;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        engineCancelChannel = mock(EngineCancelChannel.class);
        lifecycle = new RequestRegistry(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class),
                engineCancelChannel);
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
        CompletableFuture<Response> canonical = lifecycle.register(context, 8);

        CompletableFuture<Response> duplicate = lifecycle.register(context(101L), 8);

        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicate.join().getCode());
        assertSame(canonical, lifecycle.requestSlot(101L).future());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void overloadRejectionsDoNotCreateRetainedGenerations() {
        assertFalse(lifecycle.register(context(1L), 1).isDone());
        for (long id = 2; id <= 1001; id++) {
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                    lifecycle.register(context(id), 1).join().getCode());
        }
        assertEquals(1, lifecycle.snapshotSlots().size());
        assertEquals(1, lifecycle.liveRequestCount());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                lifecycle.register(context(1L), 1).join().getCode());
    }

    @Test
    void slotLockContractIsEnforcedWithoutJvmAssertions() {
        lifecycle.register(context(102L), 8);
        RequestSlot slot = lifecycle.requestSlot(102L);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class, slot::activeItem);

        assertTrue(failure.getMessage().contains("requires slot lock"));
    }

    @Test
    void globalOutstandingPermitIsAtomicAndReusableAfterLocalTerminal() {
        CompletableFuture<Response> first = lifecycle.register(context(201L), 1);
        CompletableFuture<Response> rejected = lifecycle.register(context(202L), 1);

        assertFalse(first.isDone());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                rejected.join().getCode());

        RequestState cancellation = lifecycle.cancelRequest(
                201L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(cancellation);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                first.join().getCode());

        CompletableFuture<Response> admittedAgain =
                lifecycle.register(context(203L), 1);
        assertFalse(admittedAgain.isDone(),
                "the exact terminal must release its one outstanding permit");
    }

    @Test
    void decodeAcceptanceLimitIsAtomicAndReusableAfterLocalTerminal() {
        Registered first = registerItem(211L);
        Registered second = registerItem(212L);

        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, first, 1, 30_000L));
        assertEquals(
                PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, second, 1, 30_000L));
        assertEquals(0, lifecycle.decodeAcceptanceCount(), "queued requests own no delivery guard");
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, first);
        var blocked = lifecycle.prepareDecodeAcceptance(second.item());
        assertFalse(blocked.accepted());
        assertTrue(blocked.boundary().unavailable());
        assertEquals(1, lifecycle.decodeAcceptanceCount());
        AtomicInteger wakes = new AtomicInteger();
        Runnable listener = wakes::incrementAndGet;
        blocked.boundary().availability().addListener(listener);

        lifecycle.cancelRequest(211L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(0, lifecycle.decodeAcceptanceCount());
        assertEquals(1, wakes.get());
        assertTrue(blocked.boundary().availability().isAvailable());
        blocked.boundary().availability().removeListener(listener);
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, second);
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest(212L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
        assertEquals(1, wakes.get(), "detached waiters receive no later capacity callbacks");
    }

    @Test
    void failedAcceptanceTransferRetainsCleanupUntilOutsideTheDeliverySlotLock() {
        Registered registered = registerItem(215L);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered, 2, 30_000L));
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, registered);
        var duplicate = lifecycle.prepareDecodeAcceptance(registered.item()).value();
        assertNotNull(duplicate);
        var blocked = lifecycle.prepareDecodeAcceptance(registered.item());
        assertFalse(blocked.accepted());
        var slot = lifecycle.requestSlot(215L);
        AtomicReference<Boolean> notifiedUnderSlotLock = new AtomicReference<>();
        Runnable listener = () -> notifiedUnderSlotLock.set(Thread.holdsLock(slot));
        blocked.boundary().availability().addListener(listener);
        try (duplicate) {
            assertThrows(IllegalStateException.class, () -> lifecycle.tryClaimRouteDelivery(
                    registered.item(), () -> duplicate.transferTo(registered.item())));
            assertEquals(2, lifecycle.decodeAcceptanceCount(),
                    "failed transfer leaves the prepared permit with its transaction owner");
            assertNull(notifiedUnderSlotLock.get());
        }
        assertEquals(Boolean.FALSE, notifiedUnderSlotLock.get());
        assertEquals(1, lifecycle.decodeAcceptanceCount());
        blocked.boundary().availability().removeListener(listener);
    }

    @Test
    void zeroDecodeAcceptanceLimitKeepsTheGuardUnbounded() {
        Registered first = registerItem(221L);
        Registered second = registerItem(222L);

        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, first, 0, 30_000L));
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, second, 0, 30_000L));
        assertEquals(0, lifecycle.decodeAcceptanceCount());
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, first);
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, second);
        assertEquals(2, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest(221L, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancelRequest(222L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void admissionMutationDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L), 4);
        AdmissionMutation scope =
                lifecycle.claimAdmissionMutation(301L, future);
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
    void queueDecisionResponsePublishesOutsideTheDecisionCaller() throws Exception {
        CompletableFuture<Response> future = lifecycle.register(context(302L), 4);
        CountDownLatch published = new CountDownLatch(1);
        AtomicReference<String> callbackThread = new AtomicReference<>();
        future.thenAccept(response -> {
            callbackThread.set(Thread.currentThread().getName());
            published.countDown();
        });

        Response rejection = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        assertTrue(lifecycle.publishQueueDecisionResponseAsync(
                302L, future, rejection));
        assertTrue(published.await(5, TimeUnit.SECONDS));
        assertNotEquals(Thread.currentThread().getName(), callbackThread.get());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void shutdownGateWaitsForTheExactAdmissionMutationAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                lifecycle.register(context(401L), 4);
        AdmissionMutation held =
                lifecycle.claimAdmissionMutation(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(lifecycle::closeAdmissionAndAwaitMutations);
            awaitCondition(lifecycle::isShuttingDown);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    lifecycle.register(context(402L), 4);
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.close();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = lifecycle.register(context(501L), 4);

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
    void workerActivityExtendsOnlyTheInactiveMaintenanceTtl() {
        Registered registered = registerItem(601L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestSlot slot = lifecycle.requestSlot(601L);
        long ttlMs = 300_000L;
        long heartbeatAtMs = slot.createdAtMs() + ttlMs + 1_000L;
        synchronized (slot) {
            slot.observeWorkerStatus(heartbeatAtMs);
        }

        assertFalse(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs - 1L, ttlMs));
        assertEquals(RequestState.Phase.QUEUED,
                lifecycle.getRequestState(601L, 0L).state());

        assertTrue(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs + 1L, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(601L, 0L).state());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void staleDeliveredRequestReclaimsLocalOwnershipWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered, 0, 30_000L));
        RequestSlot slot = lifecycle.requestSlot(602L);
        long ttlMs = 300_000L;

        assertTrue(lifecycle.reduceStale(
                slot, slot.createdAtMs() + ttlMs + 1L, ttlMs));

        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(602L, 0L).state());
        verify(registered.item().decodeEp()).releaseReservationExact(
                registered.item().decodeReservation());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void acceptedCancelWithoutPriorityVictimConsumesPrefillTerminal() throws Exception {
        Registered registered = registerBatchItem(701L);
        DecodeEndpoint.EngineFenceLease lease = mock(DecodeEndpoint.EngineFenceLease.class);
        when(registered.item().decodeEp().beginEngineFenceProtection(
                registered.item().decodeReservation())).thenReturn(lease);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        assertEquals(1, lifecycle.decodeAcceptanceCount());
        lifecycle.complete(claim, DeliveryResult.uncertain(new RuntimeException("RPC deadline")));

        assertFalse(registered.future().isDone());
        verify(engineCancelChannel).cancel(any(), eq(701L), anyLong());
        publishPrefillCanceled(registered);

        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.FAILED, lifecycle.getRequestState(701L, 0L).state());
        assertEquals(0, lifecycle.liveRequestCount());
        assertEquals(0, lifecycle.decodeAcceptanceCount());
        publishPrefillCanceled(registered);
        verify(lease, times(1)).close();
        verify(registered.item().decodeEp(), times(1)).releaseLocalShadowIfExact(
                registered.item().decodeReservation());
    }

    @Test
    void canceledWorkerPreservesClientCancellationCause() throws Exception {
        Registered registered = registerBatchItem(702L);
        claimBatch(registered);
        lifecycle.cancelRequest(702L, 0L, CancelReason.CLIENT_CANCELLED);
        publishPrefillCanceled(registered);

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED, lifecycle.getRequestState(702L, 0L).state());
        assertEquals(0, lifecycle.liveRequestCount());
    }

    @Test
    void decodeAcceptanceDoesNotConfirmAnUncertainBatch() throws Exception {
        Registered registered = registerBatchItem(703L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        publishDecodeAccepted(registered);
        lifecycle.complete(claim, DeliveryResult.uncertain(new RuntimeException("RPC deadline")));

        assertFalse(registered.future().isDone(), "D allocation does not publish the P fetch context");
        assertEquals(RequestState.Phase.DISPATCHING, lifecycle.getRequestState(703L, 0L).state());
        verify(engineCancelChannel).cancel(any(), eq(703L), anyLong());
        publishPrefillCanceled(registered);
        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void decodeAcceptanceDoesNotConfirmATimedOutBatch() {
        Registered registered = registerBatchItem(704L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        publishDecodeAccepted(registered);
        lifecycle.complete(claim, DeliveryResult.timedOut(new RuntimeException("RPC deadline")));

        assertFalse(registered.future().isDone());
        verify(engineCancelChannel).cancel(any(), eq(704L), anyLong());
    }

    @Test
    void decodeAcceptanceDoesNotConfirmAFailedBatch() {
        Registered registered = registerBatchItem(705L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        publishDecodeAccepted(registered);
        when(registered.item().decodeEp().settleDefiniteDispatchRejection(
                registered.item().decodeReservation()))
                .thenReturn(DecodeEndpoint.DispatchRejectionSettlement.ENGINE_ACCEPTED);
        lifecycle.complete(claim, DeliveryResult.failed(new RuntimeException("rejected")));

        assertFalse(registered.future().isDone());
        verify(engineCancelChannel).cancel(any(), eq(705L), anyLong());
    }

    @Test
    void endpointAcceptanceRacingBatchFailureCannotPublishFetch() {
        Registered registered = registerBatchItem(706L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        when(registered.item().decodeEp().settleDefiniteDispatchRejection(
                registered.item().decodeReservation()))
                .thenReturn(DecodeEndpoint.DispatchRejectionSettlement.ENGINE_ACCEPTED);
        lifecycle.complete(claim, DeliveryResult.failed(new RuntimeException("rejected")));

        assertFalse(registered.future().isDone());
        verify(engineCancelChannel).cancel(any(), eq(706L), anyLong());
    }

    @Test
    void decodeAcceptanceCannotReleaseAnUnsentBatchFence() throws Exception {
        Registered registered = registerBatchItem(707L);
        claimBatch(registered);
        RequestSlot slot = lifecycle.requestSlot(707L);
        RequestSlot.FenceReduction fence;
        synchronized (slot) {
            fence = slot.requestDeliveryFence("RPC deadline");
        }
        assertEquals(RequestSlot.FenceReduction.Status.START, fence.status());
        publishDecodeAccepted(registered);

        assertFalse(registered.future().isDone());
        synchronized (slot) {
            assertEquals(RequestSlot.FenceReduction.Status.NONE,
                    slot.applyFenceUpdate(fence.fence(), RequestSlot.FenceUpdate.CANCEL_STARTED).status());
        }
        publishPrefillCanceled(registered);
        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void batchSuccessAckAfterDecodeAcceptancePublishesFetch() throws Exception {
        Registered registered = registerBatchItem(708L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        publishDecodeAccepted(registered);
        assertFalse(registered.future().isDone());
        lifecycle.complete(claim, DeliveryResult.delivered());

        Response response = registered.future().get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(RequestState.Phase.ACKNOWLEDGED, lifecycle.getRequestState(708L, 0L).state());
        verify(engineCancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void canceledFactFromAnotherPrefillCannotSettleTheRequest() {
        Registered registered = registerBatchItem(709L);
        claimBatch(registered);
        new EndpointEventProjector(lifecycle).onPrefillStatus(
                mock(PrefillEndpoint.class), RoleType.PREFILL,
                List.of(PrefillState.WorkerStatusFact.terminal(registered.item(),
                        PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED, 8429L)));
        assertFalse(registered.future().isDone());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void priorityVictimStillSettlesItsExactPreemptionOwner() throws Exception {
        Registered registered = registerBatchItem(710L);
        claimBatch(registered);
        PreemptionRegistration preemption = lifecycle.tryClaim(710L, 1L, 77L, "priority victim")
                .orElseThrow();
        when(registered.item().decodeEp().settlePriorityCanceled(
                77L, registered.item().decodeReservation())).thenReturn(true);

        publishPrefillCanceled(registered);

        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        assertTrue(preemption.terminalObservation().toCompletableFuture().isDone());
        assertEquals(0, lifecycle.liveRequestCount());
        verify(registered.item().decodeEp()).settlePriorityCanceled(
                77L, registered.item().decodeReservation());
    }

    @Test
    void releasingPriorityOwnerKeepsUncertainBatchFencedAfterDecodeAcceptance() {
        Registered registered = registerBatchItem(711L);
        RequestRegistry.DeliveryClaim claim = claimBatch(registered);
        PreemptionRegistration preemption = lifecycle.tryClaim(711L, 1L, 78L, "priority victim")
                .orElseThrow();
        publishDecodeAccepted(registered);
        lifecycle.complete(claim, DeliveryResult.uncertain(new RuntimeException("RPC deadline")));

        assertTrue(lifecycle.tryReleasePreemption(preemption));
        assertEquals(RequestState.Phase.DISPATCHING, lifecycle.getRequestState(711L, 0L).state());
        assertFalse(registered.future().isDone());
        verify(engineCancelChannel).cancel(any(), eq(711L), anyLong());
    }

    private Registered registerBatchItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(25351);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        when(engineCancelChannel.cancel(any(), eq(requestId), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
        Registered registered = new Registered(new ScheduledRequest(
                context, future, new Response(), prefill, null,
                mock(PrefillEndpoint.class), decode, reservation, System.currentTimeMillis()), future);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, registered);
        return registered;
    }

    private RequestRegistry.DeliveryClaim claimBatch(Registered registered) {
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(), 81L, () -> true);
        assertNotNull(claim);
        return claim;
    }

    private void publishDecodeAccepted(Registered registered) {
        new EndpointEventProjector(lifecycle).onDecodeStatus(registered.item().decodeEp(),
                List.of(DecodeEndpoint.WorkerStatusFact.accepted(registered.item().decodeReservation())));
    }

    private void publishPrefillCanceled(Registered registered) {
        new EndpointEventProjector(lifecycle).onPrefillStatus(registered.item().prefillEp(), RoleType.PREFILL,
                List.of(PrefillState.WorkerStatusFact.terminal(registered.item(),
                        PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED, 8429L)));
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
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
