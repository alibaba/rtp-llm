package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.scheduler.SchedulerTestSupport.Stage;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.SchedulerTestSupport.stage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestTerminalSettlementTest {

    private static final DecodeResources.ReservationHandle RESERVATION =
            new DecodeResources.ReservationHandle(1L, String.valueOf(2L), 3L);

    @Test
    void latePlacementDiagnosticsCannotOverwriteQueueTimeoutEvidence() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
        var timeoutEvidence =
                java.util.Map.<String, Object>of("cause", "DECODE placement unavailable");
        RequestLifecycle requestOwner =
                RequestProtocolTestSupport.initialize(
                        mock(ResponseCompletionExecutor.class),
                        context,
                        mock(ExpirationTimer.class));
        var admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        admission.recordDiagnostics(java.util.Map.of("cause", "earlier placement"));
        assertEquals("earlier placement", context.data().getSchedulingDiagnostics().get("cause"));
        context.data().setSchedulingDiagnostics(timeoutEvidence);
        requestOwner.cancel(context, 0L, CancelReason.DEADLINE_EXCEEDED);
        admission.recordDiagnostics(java.util.Map.of("cause", "late placement"));
        assertEquals(timeoutEvidence, context.data().getSchedulingDiagnostics());
        assertFalse(
                context.data().getFuture().isDone(),
                "cancellation must still wait for the active admission");
    }

    @Test
    void oldDecodeRetirementCannotRemoveANewSchedulingGeneration() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4202L);
        RequestLifecycle requestOwner =
                RequestProtocolTestSupport.initialize(
                        mock(ResponseCompletionExecutor.class),
                        context,
                        mock(ExpirationTimer.class));
        RequestProtocolTestSupport.decodeRetired(
                requestOwner,
                context,
                RequestProtocolTestSupport.decodeEndpoint(),
                new DecodeResources.ReservationHandle(1L, String.valueOf(4202L), 7L));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requestOwner));
        assertTrue((context.snapshot().phase() == Phase.OPEN));
        assertFalse(context.data().getFuture().isDone());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementRejectsAnotherSourceOrReservation(boolean wrongSource) {
        Fixture f = fixture(true, true);
        var source =
                wrongSource ? RequestProtocolTestSupport.decodeEndpoint() : f.item().decodeEp();
        var reservation =
                wrongSource
                        ? RESERVATION
                        : new DecodeResources.ReservationHandle(
                                RESERVATION.endpointGenerationId(),
                                String.valueOf(RESERVATION.requestId()),
                                RESERVATION.reservationToken() + 1);

        RequestProtocolTestSupport.decodeRetired(
                f.scheduler(), f.requestContext(), source, reservation);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));

        assertTrue((f.requestContext().snapshot().phase() == Phase.OPEN));
        assertFalse(f.requestContext().data().getFuture().isDone());
        assertSame(f.item(), f.requestContext().snapshot().route());
        verify(f.item().prefillEp(), never()).releaseRequest(any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementPublishesTheOriginalPreemptionResolution(boolean cancelAlreadyUnknown) {
        Fixture f = fixture(true);
        PreemptionRegistration claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 40L).orElse(null);
        assertNotNull(claim);
        if (cancelAlreadyUnknown) {
            assertTrue(
                    RequestProtocolTestSupport.preemptionPhase(
                            f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            assertTrue(
                    RequestProtocolTestSupport.preemptionPhase(
                            f.scheduler(), claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        }
        assertNull(RequestProtocolTestSupport.resolved(claim));

        RequestProtocolTestSupport.decodeRetired(
                f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));

        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
        assertNotNull(
                RequestProtocolTestSupport.resolved(claim),
                "retirement must notify the detached original claim");
        assertEquals(
                org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                RequestProtocolTestSupport.resolved(claim).outcome());
        assertEquals(Stage.FINISHED, stage(f.requestContext()));
        assertFalse(
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                        .isSuccess());
        verify(f.item().decodeEp(), never()).updatePreemption(anyLong(), any());
    }

    @Test
    void unknownCancelOutcomeKeepsTheAcknowledgedCancellationVisible() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 4L).orElse(null);
        assertNotNull(claim);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertEquals(
                RequestState.Phase.CANCEL_REQUESTED,
                RequestState.from(f.requestContext().snapshot()).state());
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertEquals(
                RequestState.Phase.CANCEL_REQUESTED,
                RequestState.from(f.requestContext().snapshot()).state());
        assertEquals(Stage.READY_TO_DELIVER, stage(f.requestContext()));
        assertFalse(f.requestContext().data().getFuture().isDone());
    }

    @Test
    void queuedCancellationDoesNotWaitForAClaimWithoutPhysicalOwnership() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 4L).orElse(null);
        assertNotNull(claim);
        assertEquals(
                RequestState.Phase.CANCELLED,
                f.scheduler()
                        .cancel(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED)
                        .state());
        assertFalse(RequestProtocolTestSupport.releasePreemption(f.scheduler(), claim));
        assertFalse(
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                        .isSuccess());
        assertEquals(
                RequestState.Phase.CANCELLED,
                RequestState.from(f.requestContext().snapshot()).state());
        assertFalse(
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                        .isSuccess());
    }

    @Test
    void queuedDecodeRetirementFreezesTheResultBeforeAsyncCleanup() throws Exception {
        Fixture f = fixture(true, true);
        AtomicReference<String> cleanupThread = new AtomicReference<>();
        when(f.item().prefillEp().releaseRequest(any()))
                .thenAnswer(
                        call -> {
                            cleanupThread.set(Thread.currentThread().getName());
                            return true;
                        });
        {
            RequestProtocolTestSupport.decodeRetired(
                    f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
            assertFalse((f.requestContext().snapshot().phase() == Phase.OPEN));
            assertEquals(
                    RequestState.Phase.FAILED,
                    RequestState.from(f.requestContext().snapshot()).state());
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
            assertNotNull(cleanupThread.get());
            assertNotEquals(Thread.currentThread().getName(), cleanupThread.get());
        }
    }

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        Fixture f = fixture(true);
        var sender =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        assertTrue(sender.owner().data().getLifecycle().tryStartSend(sender));
        RequestProtocolTestSupport.transport(
                sender.owner().data().getLifecycle(),
                sender,
                Delivery.Result.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 4L).orElse(null);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        RequestProtocolTestSupport.applyDecodeStatus(
                f.scheduler(),
                f.requestContext(),
                f.item().decodeEp(),
                DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        verify(f.item().decodeEp(), never())
                .updatePreemption(
                        anyLong(),
                        argThat(
                                update ->
                                        update.kind()
                                                == DecodeResources.PreemptionUpdate.Kind.FINISHED));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryAckDuringPreemptionResumesWithTheOriginalBatchIdentity(boolean notificationFails) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        Delivery.Member delivery =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        assertNotNull(delivery);
        PreemptionRegistration preemption =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 9L).orElse(null);
        assertNotNull(preemption);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), preemption, PreemptionCancelPhase.CANCEL_IN_FLIGHT));

        RequestProtocolTestSupport.transport(
                delivery.owner().data().getLifecycle(), delivery, Delivery.Result.delivered());
        assertSame(preemption, RequestProtocolTestSupport.activePreemption(f.requestContext()));
        assertFalse(f.requestContext().data().getFuture().isDone());
        assertNull(
                SchedulerTestSupport.repository(f.scheduler())
                        .getRequestState(RESERVATION.requestId(), 8L));
        assertEquals(
                7L,
                SchedulerTestSupport.repository(f.scheduler())
                        .getRequestState(RESERVATION.requestId(), 7L)
                        .batchId());

        when(f.item()
                        .decodeEp()
                        .updatePreemption(9L, DecodeResources.PreemptionUpdate.active(RESERVATION)))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        var notificationFailure = new IllegalStateException("capacity listener failed");
        doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(f.requestContext()));
                            assertNull(
                                    RequestProtocolTestSupport.activePreemption(
                                            f.requestContext()));
                            assertFalse(preemption.completion().toCompletableFuture().isDone());
                            if (notificationFails) {
                                throw notificationFailure;
                            }
                            return null;
                        })
                .when(assignedDecode)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        if (notificationFails) {
            assertSame(
                    notificationFailure,
                    org.flexlb.balance.scheduler.SchedulerTestServices.failure(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                    f.scheduler())));
        }
        assertTrue(
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                        .isSuccess());
        assertEquals(
                org.flexlb.balance.preemption.VictimResolution.Outcome.DELIVERY_RESUMED,
                RequestProtocolTestSupport.awaitResult(
                                preemption.completion().toCompletableFuture())
                        .outcome());
        assertEquals(
                RequestState.Phase.ACKNOWLEDGED,
                SchedulerTestSupport.repository(f.scheduler())
                        .getRequestState(RESERVATION.requestId(), 7L)
                        .state());
        assertNull(
                SchedulerTestSupport.repository(f.scheduler())
                        .getRequestState(RESERVATION.requestId(), 8L));
        // The new protocol commits the exact reconciliation once; no redundant proof replay.
        verify(f.item().decodeEp(), times(1))
                .updatePreemption(9L, DecodeResources.PreemptionUpdate.active(RESERVATION));
        verify(f.item().decodeEp(), times(notificationFails ? 3 : 1))
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION"})
    void prefillBackedTerminalWaitsForTheExactDecodeClaimTransaction(RoleType role)
            throws Exception {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var sender =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        assertTrue(sender.owner().data().getLifecycle().tryStartSend(sender));
        RequestProtocolTestSupport.transport(
                sender.owner().data().getLifecycle(),
                sender,
                Delivery.Result.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 4L).orElse(null);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        when(f.item()
                        .decodeEp()
                        .updatePreemption(
                                4L, DecodeResources.PreemptionUpdate.finished(RESERVATION)))
                .thenReturn(
                        DecodeResources.ResourceChange.of(
                                DecodeResources.MutationStatus.STALE, false),
                        DecodeResources.ResourceChange.of(
                                DecodeResources.MutationStatus.APPLIED, true));
        doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(f.requestContext()));
                            assertNull(
                                    RequestProtocolTestSupport.activePreemption(
                                            f.requestContext()));
                            assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
                            assertFalse(claim.completion().toCompletableFuture().isDone());
                            return null;
                        })
                .when(assignedDecode)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        long observedAtMs =
                Math.max(
                        System.currentTimeMillis(),
                        f.requestContext().snapshot().lastWorkerAtMs() + 1L);
        var failed = prefillFailure(f, role, observedAtMs);
        f.scheduler().onEngineEvidence(f.requestContext(), failed);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertFalse((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(RequestProtocolTestSupport.preemptionFinished(claim));
        assertFalse(f.requestContext().data().getFuture().isDone());
        verify(f.item().decodeEp(), never())
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        var retained = f.requestContext().snapshot();
        f.scheduler().onEngineEvidence(f.requestContext(), failed);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertSame(retained, f.requestContext().snapshot());
        verify(f.item().decodeEp(), times(1))
                .updatePreemption(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION));
        f.scheduler()
                .onEngineEvidence(f.requestContext(), prefillFailure(f, role, observedAtMs + 1L));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        verify(f.item().decodeEp(), times(2))
                .updatePreemption(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION));
        assertEquals(
                RequestState.Phase.FAILED,
                RequestState.from(f.requestContext().snapshot()).state());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        RequestProtocolTestSupport.awaitCondition(
                () -> stage(f.requestContext()) == Stage.FINISHED);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertEquals(
                org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                RequestProtocolTestSupport.awaitResult(claim.completion().toCompletableFuture())
                        .outcome());
        verify(f.item().decodeEp())
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        verify(f.item().prefillEp(), never()).releaseRequest(any());
        verify(f.item().decodeEp(), never()).release(any(), any());
    }

    private static EngineEvidence prefillFailure(
            Fixture fixture, RoleType role, long observedAtMs) {
        return new EngineEvidence(
                fixture.item(), RESERVATION, role, EngineKind.FAILED, 9L, null, observedAtMs);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillFactsPublishReconciledPreemptionCapacityOutsideContext(boolean priorityCanceled) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var claim =
                f.requestContext().data().getLifecycle().tryClaim(RESERVATION, 31L).orElse(null);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(),
                        claim,
                        priorityCanceled
                                ? PreemptionCancelPhase.CANCEL_REQUESTED
                                : PreemptionCancelPhase.NOT_FOUND_STALE));
        var update =
                priorityCanceled
                        ? DecodeResources.PreemptionUpdate.canceled(RESERVATION)
                        : DecodeResources.PreemptionUpdate.active(RESERVATION);
        when(f.item().decodeEp().updatePreemption(31L, update))
                .thenAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(f.requestContext()),
                                    "Decode ledger updates must not run under the Context lock");
                            return org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                    org.flexlb.balance.endpoint.DecodeResources.MutationStatus
                                            .APPLIED,
                                    false);
                        });
        doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(f.requestContext()));
                            assertNull(
                                    RequestProtocolTestSupport.activePreemption(
                                            f.requestContext()));
                            assertEquals(
                                    priorityCanceled,
                                    RequestProtocolTestSupport.preemptionFinished(claim));
                            assertFalse(claim.completion().toCompletableFuture().isDone());
                            return null;
                        })
                .when(assignedDecode)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);

        var requestStatus =
                priorityCanceled
                        ? new PrefillEndpoint.PrefillRequestStatus(
                                f.item(),
                                PrefillEndpoint.PrefillRequestStatus.Kind.PRIORITY_CANCELED,
                                0L)
                        : new PrefillEndpoint.PrefillRequestStatus(
                                f.item(), PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE, 0L);
        RequestProtocolTestSupport.applyPrefillStatus(
                f.scheduler(),
                f.requestContext(),
                f.item().prefillEp(),
                RoleType.PREFILL,
                requestStatus);

        verify(f.item().decodeEp()).updatePreemption(31L, update);
        verify(f.item().decodeEp())
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        assertEquals(priorityCanceled, claim.completion().toCompletableFuture().isDone());
        assertEquals(priorityCanceled, f.requestContext().data().getFuture().isDone());
        if (priorityCanceled) {
            assertEquals(Stage.FINISHED, stage(f.requestContext()));
            assertFalse(
                    RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                            .isSuccess());
        } else {
            assertEquals(Stage.READY_TO_DELIVER, stage(f.requestContext()));
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = PreemptionCancelPhase.class,
            names = {"NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void requestExpiryClosesPreemptionAndIgnoresLateCallbacks(PreemptionCancelPhase outcome) {
        Fixture f = fixture(true);
        RequestContext requestContext = f.requestContext();
        Delivery.Member delivery =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        PreemptionRegistration claim =
                requestContext.data().getLifecycle().tryClaim(RESERVATION, 9L).orElse(null);
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, outcome));
        f.scheduler().cancel(requestContext, 7L, CancelReason.CLIENT_CANCELLED);
        var inactivity = requestContext.snapshot().deadlines().inactivity();
        assertNotNull(inactivity);
        RequestProtocolTestSupport.expireInactivity(
                f.scheduler(),
                requestContext,
                inactivity,
                RequestProtocolTestSupport.<Long>inspect(
                        f.scheduler(), requestContext, "inactivityExpiresAtMsLocked"));
        assertFalse(delivery.completion().toCompletableFuture().isDone(), "sender has not exited");
        RequestProtocolTestSupport.transport(
                delivery.owner().data().getLifecycle(),
                delivery,
                Delivery.Result.notSent(new IllegalStateException("expired before send")));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        RequestState ended = RequestState.from(requestContext.snapshot());
        assertEquals(RequestState.Phase.CANCELLED, ended.state());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertTrue((stage(requestContext) == Stage.FINISHED));
        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
        assertNull(requestContext.snapshot().route());
        assertEquals(
                CancelReason.CLIENT_CANCELLED,
                requestContext.snapshot().cancellation().reason(),
                "finished context preserves the original cancellation fact");
        RequestProtocolTestSupport.transport(
                delivery.owner().data().getLifecycle(), delivery, Delivery.Result.delivered());
        assertFalse(
                RequestProtocolTestSupport.finishPreemption(
                        f.scheduler(), claim, "late Cancel ACK"));
        RequestProtocolTestSupport.applyDecodeStatus(
                f.scheduler(),
                requestContext,
                f.item().decodeEp(),
                DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        RequestProtocolTestSupport.expireInactivity(
                f.scheduler(), requestContext, inactivity, Long.MAX_VALUE);
        assertEquals(ended, RequestState.from(requestContext.snapshot()));
        verify(f.item().decodeEp()).release(RESERVATION, DecodeResources.ReleaseReason.NOT_SENT);
        verify(f.item().prefillEp()).releaseRequest(f.item());
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() throws Exception {
        Fixture f = fixture(true);
        Delivery.Member claim =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        assertTrue(claim.owner().data().getLifecycle().tryStartSend(claim));
        RequestProtocolTestSupport.transport(
                claim.owner().data().getLifecycle(), claim, Delivery.Result.delivered());
        Response delivered =
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture());
        assertTrue(delivered.isSuccess());
        RequestProtocolTestSupport.expireInactiveRequest(
                f.scheduler(),
                f.requestContext(),
                RequestProtocolTestSupport.<Long>inspect(
                        f.scheduler(), f.requestContext(), "inactivityExpiresAtMsLocked"));
        assertEquals(
                RequestState.Phase.TIMED_OUT,
                RequestState.from(f.requestContext().snapshot()).state());
        assertSame(
                delivered,
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        RequestProtocolTestSupport.awaitCondition(
                () -> stage(f.requestContext()) == Stage.FINISHED);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeEndDuringAdmissionPreservesTheEarlierCancellationCause(boolean retirement) {
        Fixture f = fixture(false);
        f.scheduler().cancel(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED);
        if (retirement) {
            RequestProtocolTestSupport.decodeRetired(
                    f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        } else {
            RequestProtocolTestSupport.applyDecodeStatus(
                    f.scheduler(),
                    f.requestContext(),
                    f.item().decodeEp(),
                    DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        }
        assertFalse((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(f.requestContext().data().getFuture().isDone());
        f.admission().finish();
        assertEquals(
                RequestState.Phase.CANCELLED,
                RequestState.from(f.requestContext().snapshot()).state());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(
                RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                        .isSuccess());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void workerProofExcludesLateAckBeforeUnlockedCleanupCommits(boolean preempting)
            throws Exception {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        RequestContext requestContext = f.requestContext();
        Delivery.Member delivery =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        if (preempting) {
            PreemptionRegistration claim =
                    requestContext.data().getLifecycle().tryClaim(RESERVATION, 9L).orElse(null);
            assertTrue(
                    RequestProtocolTestSupport.preemptionPhase(
                            f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(requestContext));
                            assertFalse(
                                    (stage(requestContext) == Stage.FINISHED),
                                    "cleanup must precede final record commitment");
                            assertFalse(
                                    RequestProtocolTestSupport.awaitResult(
                                                    requestContext.data().getFuture())
                                            .isSuccess(),
                                    "late ACK cannot replace the terminal response");
                            return DecodeResources.ResourceChange.of(
                                    DecodeResources.ReservationReleaseResult.RELEASED, true);
                        })
                .when(assignedDecode)
                .release(RESERVATION, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        assertTrue(delivery.owner().data().getLifecycle().tryStartSend(delivery));
        RequestProtocolTestSupport.applyDecodeStatus(
                f.scheduler(),
                requestContext,
                f.item().decodeEp(),
                DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 42L));
        assertEquals(
                RequestState.Phase.FAILED, RequestState.from(requestContext.snapshot()).state());
        RequestProtocolTestSupport.transport(
                delivery.owner().data().getLifecycle(), delivery, Delivery.Result.delivered());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        RequestProtocolTestSupport.awaitCondition(() -> stage(requestContext) == Stage.FINISHED);
        assertTrue((stage(requestContext) == Stage.FINISHED));
        assertFalse(
                RequestProtocolTestSupport.awaitResult(requestContext.data().getFuture())
                        .isSuccess());
        verify(f.item().prefillEp()).releaseRequest(f.item());
    }

    @Test
    void completedReleaseEvidenceSurvivesLateFrontendFailureAndWaitsForTerminalEffects() {
        Fixture f = fixture(true);
        RequestContext context = f.requestContext();
        var delivery =
                DeliveryDispatcher.commitDelivery(
                        f.scheduler(),
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                f.item(), RequestProtocolTestSupport.handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        7L);
        assertTrue(f.scheduler().tryStartSend(delivery));
        RequestProtocolTestSupport.transport(f.scheduler(), delivery, Delivery.Result.delivered());
        End action;
        synchronized (context) {
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.recordEngineEvidence(
                                    new EngineEvidence(
                                            f.item(),
                                            RESERVATION,
                                            RoleType.DECODE,
                                            EngineKind.COMPLETED,
                                            0L,
                                            null,
                                            System.currentTimeMillis())));
            var end =
                    context.beginClose(context.snapshot().pendingEnd(), System.currentTimeMillis());
            action = RequestCleanupPassTest.ended(end);
        }
        assertNotNull(action);
        assertNotNull(
                context.snapshot().deadlines().inactivity(),
                "unsettled resource retains inactivity watch");
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                f.scheduler(), "scheduleClose", context);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertEquals(
                Stage.FINISHED,
                stage(context),
                "the single finish owner completes notifications and physical settlement together");
        assertTrue(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
        assertNull(
                context.snapshot().deadlines().inactivity(),
                "completed notification and settlement remove the watch");
        assertFalse(
                f.scheduler()
                        .completeExternal(
                                context,
                                RequestFuture.Completion.FAILURE,
                                null,
                                new IllegalStateException("late frontend failure"),
                                true));
        assertTrue(
                context.snapshot().deliveryReleasable(),
                "late frontend failure cannot revoke completed execution evidence");
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                f.scheduler(), "scheduleClose", context);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        assertEquals(Stage.FINISHED, stage(context));
        verify(f.item().prefillEp(), times(1)).releaseRequest(f.item());
        assertNull(context.snapshot().deadlines().inactivity());
    }

    @Test
    void anotherSchedulerCannotAdvanceOrReleaseAnExactPreemptionClaim() {
        Fixture owner = fixture(true, true);
        Fixture other = fixture(true, true);
        PreemptionRegistration claim =
                owner.requestContext().data().getLifecycle().tryClaim(RESERVATION, 4L).orElse(null);
        assertNotNull(claim);
        var before = owner.requestContext().snapshot();
        assertFalse(
                RequestProtocolTestSupport.preemptionPhase(
                        other.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertFalse(RequestProtocolTestSupport.releasePreemption(other.scheduler(), claim));
        assertFalse(
                RequestProtocolTestSupport.finishPreemption(
                        other.scheduler(), claim, "foreign callback"));
        assertEquals(before, owner.requestContext().snapshot());
        assertFalse(owner.requestContext().data().getFuture().isDone());
        assertFalse(other.requestContext().data().getFuture().isDone());
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        owner.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.releasePreemption(owner.scheduler(), claim));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cleanupProofSettlesDecodeBeforeAdvancingRequest(boolean resourceSettled) {
        Fixture f = fixture(true, true);
        var claim = f.scheduler().tryClaim(RESERVATION, 4L).orElseThrow();
        var proof = DecodeResources.PreemptionUpdate.fenced(RESERVATION);
        when(f.item().decodeEp().updatePreemption(4L, proof))
                .thenAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(f.requestContext()),
                                    "Decode ledger updates must not run under the Context lock");
                            assertFalse(
                                    contextFinished(f.requestContext()),
                                    "resource transaction precedes Context finality");
                            assertFalse(f.requestContext().snapshot().preemption().finished());
                            return DecodeResources.ResourceChange.of(
                                    resourceSettled
                                            ? DecodeResources.MutationStatus.APPLIED
                                            : DecodeResources.MutationStatus.STALE,
                                    resourceSettled);
                        });
        var before = f.requestContext().snapshot();
        boolean applied =
                f.scheduler()
                        .onPreemption(
                                claim,
                                new RequestLifecycle.PreemptionEvent(
                                        null, false, f.item().decodeEp(), RESERVATION, "cleaned"));
        assertEquals(resourceSettled, applied);
        verify(f.item().decodeEp()).updatePreemption(4L, proof);
        if (resourceSettled) {
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
            assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
            assertFalse(
                    RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture())
                            .isSuccess());
        } else {
            assertSame(before, f.requestContext().snapshot());
            assertFalse(f.requestContext().data().getFuture().isDone());
            verify(f.item().decodeEp(), never())
                    .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        }
    }

    @Test
    void cleanupProofCannotMixVictimsFromTheSamePreemptionAttempt() {
        Fixture f = fixture(true, true);
        var claim = f.scheduler().tryClaim(RESERVATION, 4L).orElseThrow();
        var wrongVictim =
                new DecodeResources.ReservationHandle(
                        RESERVATION.endpointGenerationId(),
                        String.valueOf(RESERVATION.requestId() + 1L),
                        RESERVATION.reservationToken());
        var before = f.requestContext().snapshot();
        assertFalse(
                f.scheduler()
                        .onPreemption(
                                claim,
                                new RequestLifecycle.PreemptionEvent(
                                        null,
                                        false,
                                        f.item().decodeEp(),
                                        wrongVictim,
                                        "wrong victim")));
        assertSame(before, f.requestContext().snapshot());
        verify(f.item().decodeEp(), never()).updatePreemption(anyLong(), any());
        verify(f.item().decodeEp(), never())
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5})
    void transientTerminalTimerFailureRetriesAfterActualResourceSettlement(int failures) {
        Fixture f = fixture(true, true);
        var context = f.requestContext();
        var timer =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler())
                        .deadlines();
        org.mockito.Mockito.clearInvocations(timer);
        var expected = new IllegalStateException("timer update failed");
        var attempts = new java.util.concurrent.atomic.AtomicInteger();
        var installedRevision =
                new java.util.concurrent.atomic.AtomicLong(
                        context.snapshot().deadlines().revision());
        when(timer.needsUpdate(any(), any()))
                .thenAnswer(
                        call -> {
                            RequestContext owner = call.getArgument(0);
                            ExpirationTimer.Plan plan = call.getArgument(1);
                            return f.scheduler().requests.isCurrent(owner)
                                    && plan.revision() > installedRevision.get();
                        });
        doAnswer(
                        call -> {
                            if (attempts.incrementAndGet() <= failures) {
                                throw expected;
                            }
                            ExpirationTimer.Plan plan = call.getArgument(1);
                            installedRevision.accumulateAndGet(plan.revision(), Math::max);
                            return null;
                        })
                .when(timer)
                .update(any(), any());
        assertSame(
                expected,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                f.scheduler()
                                        .closeRequest(
                                                context,
                                                new EndEvidence(
                                                        f.item(),
                                                        EndKind.FAILURE,
                                                        null,
                                                        false,
                                                        0L,
                                                        org.flexlb.dao.loadbalance.StrategyErrorType
                                                                .DISPATCH_FAILED,
                                                        "original terminal failure",
                                                        false,
                                                        false))));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        var original = context.snapshot().outcome();
        assertEquals("original terminal failure", original.detail());
        assertFalse(RequestProtocolTestSupport.awaitResult(context.data().getFuture()).isSuccess());
        if (failures >= 3) {
            assertEquals(3, attempts.get(), "one finish wake-up performs at most three attempts");
            assertEquals(Phase.ENDING, context.snapshot().phase());
            assertTrue(context.snapshot().close().notificationsAttempted());
            assertFalse(context.snapshot().close().notificationsDone());
            assertTrue(timer.needsUpdate(context, context.snapshot().deadlines()));
            assertNotNull(
                    context.snapshot().deadlines().inactivity(),
                    "real pending notification keeps its existing watch");
            assertSame(context, f.scheduler().requests.findActive(RESERVATION.requestId()));
            assertNull(f.scheduler().requests.findTerminal(RESERVATION.requestId()));
            verify(f.item().prefillEp(), times(1)).releaseRequest(f.item());
            verify(f.item().decodeEp(), times(1))
                    .release(RESERVATION, DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
            if (failures > 3) {
                assertSame(
                        expected,
                        assertThrows(
                                IllegalStateException.class,
                                () -> f.scheduler().scheduleClose(context)));
            } else {
                f.scheduler().scheduleClose(context);
            }
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.scheduler()));
        }
        assertEquals(
                Phase.CLOSED,
                context.snapshot().phase(),
                "recovering notifications finish their existing owner");
        assertSame(
                original,
                context.snapshot().outcome(),
                "retries preserve the frozen outcome and first cause");
        assertTrue(context.snapshot().close().notificationsDone());
        assertNull(context.snapshot().deadlines().inactivity());
        verify(f.item().prefillEp(), times(1)).releaseRequest(f.item());
        verify(f.item().decodeEp(), times(1))
                .release(RESERVATION, DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        verify(timer, times(failures + 1)).update(any(), any());
    }

    private static boolean contextFinished(RequestContext context) {
        return context.snapshot().phase() == Phase.CLOSED;
    }

    private static Fixture fixture(boolean finishAdmission) {
        return fixture(finishAdmission, false);
    }

    private static Fixture fixture(boolean finishAdmission, boolean queueScheduling) {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        RequestContext context =
                RequestProtocolTestSupport.context(config, Long.parseLong(RESERVATION.requestId()));
        var publisher = mock(ResponseCompletionExecutor.class);
        var timer = mock(ExpirationTimer.class);
        RequestLifecycle requestOwner =
                RequestProtocolTestSupport.initialize(publisher, context, timer);
        when(SchedulerTestSupport.engineClient(requestOwner)
                        .cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        when(publisher.register()).thenReturn(true);
        doAnswer(
                        call -> {
                            call.<Runnable>getArgument(1).run();
                            return null;
                        })
                .when(publisher)
                .submit(any(), any());
        when(publisher.invoke(any()))
                .thenAnswer(
                        call ->
                                call.<java.util.function.BooleanSupplier>getArgument(0)
                                        .getAsBoolean());
        var prefillServer = new org.flexlb.dao.loadbalance.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        RequestRoute item =
                org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        prefillServer,
                        null,
                        mock(PrefillEndpoint.class),
                        RequestProtocolTestSupport.decodeEndpoint(),
                        RESERVATION,
                        System.currentTimeMillis());
        AdmissionHandle admission =
                RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        assertEquals(
                org.flexlb.balance.PlacementResult.Status.SUCCESS,
                RequestProtocolTestSupport.commitRoute(
                        requestOwner, item, RequestProtocolTestSupport.publication(() -> true)));
        if (finishAdmission) {
            admission.finish();
        }
        return new Fixture(requestOwner, context, item, admission);
    }

    private record Fixture(
            RequestLifecycle scheduler,
            RequestContext requestContext,
            RequestRoute item,
            AdmissionHandle admission) {}
}
