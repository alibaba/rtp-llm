package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryRejection;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
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
        assertSame(canonical, lifecycle.requestSlot("101").future());
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
        RequestSlot slot = lifecycle.requestSlot("102");

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
                "201", 0L, CancelReason.CLIENT_CANCELLED);
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

        lifecycle.cancelRequest("211", 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(0, lifecycle.decodeAcceptanceCount());
        assertEquals(1, wakes.get());
        assertTrue(blocked.boundary().availability().isAvailable());
        blocked.boundary().availability().removeListener(listener);
        RequestLifecycleTestSupport.prepareAcceptance(lifecycle, second);
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest("212", 0L, CancelReason.CLIENT_CANCELLED);
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
        var slot = lifecycle.requestSlot("215");
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

        lifecycle.cancelRequest("221", 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancelRequest("222", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void admissionMutationDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L), 4);
        AdmissionMutation scope =
                lifecycle.claimAdmissionMutation("301", future);
        assertNotNull(scope);

        RequestState requested = lifecycle.cancelRequest(
                "301", 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState("301", 0L).state());
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
                "302", future, rejection));
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
                lifecycle.claimAdmissionMutation("401", heldFuture);
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
                "999", 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancelRequest(
                "501", 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState exact = lifecycle.cancelRequest(
                "501", 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void retryDeadlineAfterDefiniteBatchRejectionsPublishesBatchSloTimeout() {
        Registered registered = registerItem(5011L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        when(registered.item().decodeEp().settleDefiniteDispatchRejection(
                registered.item().decodeReservation())).thenReturn(
                        DecodeEndpoint.DispatchRejectionSettlement.RELEASED);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(), 901L, () -> true);
        assertNotNull(claim);

        lifecycle.complete(
                claim,
                DeliveryResult.retryDeadlineExceeded(
                        new DeliveryRejection(
                                registered.item().requestId(),
                                13L,
                                "transient engine rejection")));

        Response response = registered.future().join();
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                response.getCode());
        assertTrue(response.getErrorMessage().contains(
                "request deadline exceeded"));
        assertFalse(response.getErrorMessage().contains("error_code=13"));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("5011", 901L).state());
        verify(registered.item().decodeEp())
                .settleDefiniteDispatchRejection(
                        registered.item().decodeReservation());
    }

    @Test
    void batchDecodeTerminalReleasesExactPrefillCounterpartOnlyAtTerminal() {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        Registered registered = registerItem(
                502L, prefill, decode, 1L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(), 702L, () -> true);
        assertNotNull(claim);
        lifecycle.complete(claim, DeliveryResult.delivered());
        assertEquals(200, registered.future().join().getCode());
        EndpointEventProjector projector = new EndpointEventProjector(lifecycle);

        projector.onDecodeStatus(decode, List.of(
                DecodeEndpoint.WorkerStatusFact.active(
                        registered.item().decodeReservation()),
                DecodeEndpoint.WorkerStatusFact.accepted(
                        registered.item().decodeReservation())));

        verify(prefill, never()).releaseCommittedItem(registered.item());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState("502", 702L).state());

        projector.onDecodeStatus(decode, List.of(
                DecodeEndpoint.WorkerStatusFact.terminal(
                        registered.item().decodeReservation(), 0L)));

        verify(prefill).releaseCommittedItem(registered.item());
        assertEquals(RequestState.Phase.COMPLETED,
                lifecycle.getRequestState("502", 702L).state());
    }

    @Test
    void staleDecodeTerminalCannotReleaseReplacementPrefillGeneration() {
        PrefillEndpoint oldPrefill = mock(PrefillEndpoint.class);
        PrefillEndpoint replacementPrefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        Registered old = registerItem(503L, oldPrefill, decode, 1L);
        RequestLifecycleTestSupport.bind(lifecycle, old);
        RequestSlot oldSlot = lifecycle.requestSlot("503");

        lifecycle.cancelRequest("503", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                old.future().join().getCode());
        assertTrue(lifecycle.removeExactTombstone(
                oldSlot, Long.MAX_VALUE));

        Registered replacement = registerItem(
                503L, replacementPrefill, decode, 2L);
        RequestLifecycleTestSupport.bind(lifecycle, replacement);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                replacement.item(), 703L, () -> true);
        assertNotNull(claim);
        lifecycle.complete(claim, DeliveryResult.delivered());
        assertEquals(200, replacement.future().join().getCode());

        new EndpointEventProjector(lifecycle).onDecodeStatus(decode, List.of(
                DecodeEndpoint.WorkerStatusFact.terminal(
                        old.item().decodeReservation(), 0L)));

        verify(replacementPrefill, never())
                .releaseCommittedItem(replacement.item());
        assertSame(replacement.future(), lifecycle.requestSlot("503").future());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState("503", 703L).state());
    }

    @Test
    void schedulingDeadlineProtectsBlockedGlobalRequestFromInactiveTtl() {
        BalanceContext context = context(600L);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        RequestSlot slot = lifecycle.requestSlot("600");
        long ttlMs = 30_000L;
        long maintenanceAtMs = slot.createdAtMs() + ttlMs + 1L;

        synchronized (slot) {
            assertNull(slot.activeItem(),
                    "Decode-capacity blocking happens before item publication");
            assertTrue(slot.ownsSchedulingDeadline());
        }
        assertTrue(maintenanceAtMs < context.getRequestExpiresAtMs());
        assertFalse(lifecycle.reduceStale(
                slot, maintenanceAtMs, ttlMs));

        assertFalse(future.isDone());
        assertEquals(RequestState.Phase.QUEUED,
                lifecycle.getRequestState("600", 0L).state());
    }

    @Test
    void schedulingDeadlineProtectsDispatchUntilDeliveryAcknowledgement() {
        Registered registered = registerItem(6001L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(), () -> true);
        assertNotNull(claim);
        RequestSlot slot = lifecycle.requestSlot("6001");
        long ttlMs = 30_000L;

        assertFalse(lifecycle.reduceStale(
                slot, slot.createdAtMs() + ttlMs + 1L, ttlMs));
        assertFalse(registered.future().isDone());
        assertEquals(RequestState.Phase.DISPATCHING,
                lifecycle.getRequestState("6001", 0L).state());

        lifecycle.complete(claim, DeliveryResult.delivered());
        assertEquals(200, registered.future().join().getCode());
        long inactiveSince;
        synchronized (slot) {
            inactiveSince = slot.lastWorkerStatusAtMs();
        }

        assertTrue(lifecycle.reduceStale(
                slot, inactiveSince + ttlMs + 1L, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("6001", 0L).state());
    }

    @Test
    void inactivePreAckEngineFenceCanBeReclaimedBeforeLongSchedulingDeadline() {
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(8090);
        Registered registered = registerItem(6002L, prefill);
        DecodeEndpoint.EngineFenceLease fenceLease =
                mock(DecodeEndpoint.EngineFenceLease.class);
        when(registered.item().decodeEp().beginEngineFenceProtection(
                registered.item().decodeReservation())).thenReturn(fenceLease);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(), () -> true);
        assertNotNull(claim);
        RequestSlot slot = lifecycle.requestSlot("6002");
        long ttlMs = 1_000L;
        long inactiveSince;

        synchronized (slot) {
            assertTrue(slot.ownsSchedulingDeadline());
            RequestSlot.FenceReduction fence = slot.requestDeliveryFence(
                    "pre_ack_delivery_outcome_unknown");
            assertEquals(RequestSlot.FenceReduction.Status.START,
                    fence.status());
            assertEquals(RequestSlot.FenceReduction.Status.NONE,
                    slot.applyFenceUpdate(
                            fence.fence(),
                            RequestSlot.FenceUpdate.CANCEL_STARTED).status());
            assertEquals(RequestSlot.FenceReduction.Status.NONE,
                    slot.applyFenceUpdate(
                            fence.fence(),
                            RequestSlot.FenceUpdate.AWAIT_TERMINAL).status());
            assertTrue(slot.ownsReclaimableInactiveFence(registered.item()));
            inactiveSince = slot.lastWorkerStatusAtMs();
        }
        long maintenanceAtMs = inactiveSince + ttlMs + 1L;
        assertTrue(maintenanceAtMs
                < registered.item().ctx().getRequestExpiresAtMs());

        assertTrue(lifecycle.reduceStale(
                slot, maintenanceAtMs, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("6002", 0L).state());
        verify(fenceLease).close();
    }

    @Test
    void deliveryAcknowledgementRebasesInactiveTtlBeforeFirstWorkerStatus()
            throws Exception {
        Registered registered = registerItem(6003L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(), () -> true);
        assertNotNull(claim);
        RequestSlot slot = lifecycle.requestSlot("6003");
        long ttlMs = 10L;
        long preAckBaseline;
        synchronized (slot) {
            assertTrue(slot.ownsSchedulingDeadline());
            preAckBaseline = slot.lastWorkerStatusAtMs();
        }
        awaitCondition(() -> System.currentTimeMillis()
                > preAckBaseline + ttlMs);

        lifecycle.complete(claim, DeliveryResult.delivered());
        assertEquals(200, registered.future().join().getCode());
        RequestState acknowledged = lifecycle.getRequestState("6003", 0L);
        assertEquals(RequestState.Phase.ACKNOWLEDGED, acknowledged.state());
        long postAckBaseline;
        synchronized (slot) {
            assertFalse(slot.ownsSchedulingDeadline());
            postAckBaseline = slot.lastWorkerStatusAtMs();
        }
        assertTrue(acknowledged.updatedAtMs() - preAckBaseline > ttlMs);

        assertFalse(lifecycle.reduceStale(
                slot, acknowledged.updatedAtMs(), ttlMs));
        assertEquals(acknowledged.updatedAtMs(), postAckBaseline,
                "delivery ACK atomically starts a fresh inactivity interval");
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState("6003", 0L).state());

        assertTrue(lifecycle.reduceStale(
                slot, postAckBaseline + ttlMs + 1L, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("6003", 0L).state());
    }

    @Test
    void workerActivityExtendsOnlyTheInactiveMaintenanceTtl() {
        Registered registered = registerItem(601L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        acknowledgeRoute(registered);
        RequestSlot slot = lifecycle.requestSlot("601");
        long ttlMs = 300_000L;
        long heartbeatAtMs = slot.createdAtMs() + ttlMs + 1_000L;
        synchronized (slot) {
            slot.observeWorkerStatus(heartbeatAtMs);
        }

        assertFalse(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs - 1L, ttlMs));
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState("601", 0L).state());

        assertTrue(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs + 1L, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("601", 0L).state());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void staleDeliveredRequestReclaimsLocalOwnershipWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered, 0, 30_000L));
        acknowledgeRoute(registered);
        RequestSlot slot = lifecycle.requestSlot("602");
        long ttlMs = 300_000L;
        long inactiveSince;
        synchronized (slot) {
            inactiveSince = slot.lastWorkerStatusAtMs();
        }

        assertTrue(lifecycle.reduceStale(
                slot, inactiveSince + ttlMs + 1L, ttlMs));

        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("602", 0L).state());
        verify(registered.item().decodeEp()).releaseReservationExact(
                registered.item().decodeReservation());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void staleDeliveryUncertaintyFenceRetiresSchedulerOwnership() {
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(8090);
        Registered registered = registerItem(603L, prefill);
        DecodeEndpoint.EngineFenceLease fenceLease =
                mock(DecodeEndpoint.EngineFenceLease.class);
        when(registered.item().decodeEp().beginEngineFenceProtection(
                registered.item().decodeReservation())).thenReturn(fenceLease);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered, 0, 30_000L));
        acknowledgeRoute(registered);
        RequestSlot slot = lifecycle.requestSlot("603");
        RequestSlot.FenceReduction fence;
        synchronized (slot) {
            fence = slot.requestDeliveryFence(
                    "post_delivery_acceptance_timeout");
            assertEquals(RequestSlot.FenceReduction.Status.START,
                    fence.status());
            assertEquals(RequestSlot.FenceReduction.Status.NONE,
                    slot.applyFenceUpdate(
                            fence.fence(),
                            RequestSlot.FenceUpdate.CANCEL_STARTED).status());
            assertEquals(RequestSlot.FenceReduction.Status.NONE,
                    slot.applyFenceUpdate(
                            fence.fence(),
                            RequestSlot.FenceUpdate.AWAIT_TERMINAL).status());
        }
        long ttlMs = 300_000L;
        long inactiveSince;
        synchronized (slot) {
            inactiveSince = slot.lastWorkerStatusAtMs();
        }

        assertTrue(lifecycle.reduceStale(
                slot, inactiveSince + ttlMs + 1L, ttlMs));

        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState("603", 0L).state());
        verify(fenceLease).close();
        verify(registered.item().decodeEp(), never())
                .releaseReservationExact(registered.item().decodeReservation());
        synchronized (slot) {
            assertEquals(RequestSlot.FenceReduction.Status.STALE,
                    slot.applyFenceUpdate(
                            fence.fence(),
                            RequestSlot.FenceUpdate.TOMBSTONED).status());
        }
    }

    @Test
    void staleCancellationFencePreservesClientCancellationFirstCause() {
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(8090);
        Registered registered = registerItem(6031L, prefill);
        DecodeEndpoint.EngineFenceLease fenceLease =
                mock(DecodeEndpoint.EngineFenceLease.class);
        when(registered.item().decodeEp().beginEngineFenceProtection(
                registered.item().decodeReservation())).thenReturn(fenceLease);
        when(engineCancelChannel.cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.eq("6031"),
                org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelAck.NOT_FOUND));
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(), 7031L, () -> true);
        assertNotNull(claim);
        lifecycle.complete(claim, DeliveryResult.delivered());

        RequestState requested = lifecycle.cancelRequest(
                "6031", 7031L, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        RequestSlot slot = lifecycle.requestSlot("6031");
        long ttlMs = 300_000L;
        long inactiveSince;
        synchronized (slot) {
            assertTrue(slot.isLiveGeneration());
            assertTrue(slot.hasCancellationFirstCause());
            assertTrue(slot.ownsReclaimableInactiveFence(registered.item()));
            inactiveSince = slot.lastWorkerStatusAtMs();
        }

        assertTrue(lifecycle.reduceStale(
                slot, inactiveSince + ttlMs + 1L, ttlMs));

        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState("6031", 7031L).state());
        verify(fenceLease).close();
        verify(registered.item().decodeEp(), never())
                .releaseReservationExact(registered.item().decodeReservation());
    }

    @Test
    void clientCancellationAcceptsPrefillPriorityTerminalAsAuthoritativeProof() {
        ServerStatus prefillStatus = new ServerStatus();
        prefillStatus.setServerIp("127.0.0.1");
        prefillStatus.setGrpcPort(8090);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, "604", 1L);
        DecodeEndpoint.EngineFenceLease fenceLease =
                mock(DecodeEndpoint.EngineFenceLease.class);
        when(decode.beginEngineFenceProtection(reservation))
                .thenReturn(fenceLease);
        BalanceContext context = context(604L);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        Registered registered = new Registered(
                new ScheduledRequest(
                        context,
                        future,
                        new Response(),
                        prefillStatus,
                        null,
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(), 704L, () -> true);
        assertNotNull(claim);
        lifecycle.complete(claim, DeliveryResult.delivered());

        RequestState requested = lifecycle.cancelRequest(
                "604", 704L, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());

        RequestSlot slot = lifecycle.requestSlot("604");
        Runnable work;
        synchronized (slot) {
            work = lifecycle.materializePostLockActionLocked(
                    slot,
                    slot.reducePriorityCanceled(prefill, registered.item()),
                    null);
        }
        lifecycle.runPostLock(work);

        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState("604", 704L).state());
        verify(fenceLease).settleAuthoritativeTerminal();
        verify(fenceLease).close();
        verify(decode).releaseLocalShadowIfExact(reservation);
        verify(decode, never()).reconcilePriorityVictimActive(
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.any());
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, null);
    }

    private Registered registerItem(
            long requestId, ServerStatus prefill) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(
                        1L, Long.toString(requestId), 1L);
        return new Registered(
                new ScheduledRequest(
                        context,
                        future,
                        new Response(),
                        prefill,
                        null,
                        null,
                        decode,
                        reservation,
                System.currentTimeMillis()),
                future);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefill,
            DecodeEndpoint decode,
            long reservationToken) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(
                        1L, Long.toString(requestId), reservationToken);
        return new Registered(
                new ScheduledRequest(
                        context,
                        future,
                        new Response(),
                        null,
                        null,
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
    }

    private void acknowledgeRoute(Registered registered) {
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(), () -> true);
        assertNotNull(claim);
        lifecycle.complete(claim, DeliveryResult.delivered());
        assertEquals(200, registered.future().join().getCode());
    }

}
