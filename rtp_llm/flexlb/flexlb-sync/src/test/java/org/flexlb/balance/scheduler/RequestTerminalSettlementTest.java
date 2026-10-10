package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.flexlb.balance.scheduler.SchedulerTestSupport.stage;
import org.flexlb.balance.scheduler.SchedulerTestSupport.Stage;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
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
            new DecodeResources.ReservationHandle(1L, 2L, 3L);

    @Test
    void latePlacementDiagnosticsCannotOverwriteQueueTimeoutEvidence() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
        var queue = mock(QueuedRequestScheduler.class);
        var timeoutEvidence = java.util.Map.<String, Object>of("cause", "DECODE placement unavailable");
        when(queue.getLatestQueueWaitSnapshot()).thenReturn(timeoutEvidence);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), context, mock(ExpirationTimer.class));
        org.springframework.test.util.ReflectionTestUtils.setField(requestOwner, "latestQueueWaitSnapshot", timeoutEvidence);
        var admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        admission.recordDiagnostics(java.util.Map.of("cause", "earlier placement"));
        assertEquals("earlier placement", context.data().getSchedulingDiagnostics().get("cause"));
        requestOwner.cancel(context, 0L, CancelReason.DEADLINE_EXCEEDED);
        admission.recordDiagnostics(java.util.Map.of("cause", "late placement"));
        assertEquals(timeoutEvidence, context.data().getSchedulingDiagnostics());
        assertFalse(context.data().getFuture().isDone(), "cancellation must still wait for the active admission");
    }

    @Test
    void oldDecodeRetirementCannotRemoveANewSchedulingGeneration() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4202L);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), context, mock(ExpirationTimer.class));
        RequestProtocolTestSupport.decodeRetired(requestOwner, context, RequestProtocolTestSupport.decodeEndpoint(), new DecodeResources.ReservationHandle(1L, 4202L, 7L));
        SchedulerTestSupport.awaitRequestWork(requestOwner.runtime);
        assertTrue((context.snapshot().phase() == Phase.OPEN));
        assertFalse(context.data().getFuture().isDone());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementRejectsAnotherSourceOrReservation(boolean wrongSource) {
        Fixture f = fixture(true, true);
        var source = wrongSource ? RequestProtocolTestSupport.decodeEndpoint() : f.item().decodeEp();
        var reservation = wrongSource ? RESERVATION : new DecodeResources.ReservationHandle(
                RESERVATION.endpointGenerationId(), RESERVATION.requestId(), RESERVATION.reservationToken() + 1);

        RequestProtocolTestSupport.decodeRetired(f.scheduler(), f.requestContext(), source, reservation);
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);

        assertTrue((f.requestContext().snapshot().phase() == Phase.OPEN));
        assertFalse(f.requestContext().data().getFuture().isDone());
        assertSame(f.item(), f.requestContext().snapshot().route());
        verify(f.item().prefillEp(), never()).release(org.mockito.ArgumentMatchers.isNull(), any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementPublishesTheOriginalPreemptionResolution(boolean cancelAlreadyUnknown) {
        Fixture f = fixture(true);
        PreemptionRegistration claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 40L, "victim").orElse(null);
        assertNotNull(claim);
        if (cancelAlreadyUnknown) {
            assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        }
        assertNull(RequestProtocolTestSupport.resolved(claim));

        RequestProtocolTestSupport.decodeRetired(f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);

        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
        assertNotNull(RequestProtocolTestSupport.resolved(claim), "retirement must notify the detached original claim");
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                RequestProtocolTestSupport.resolved(claim).outcome());
        assertEquals(Stage.FINISHED, stage(f.requestContext()));
        assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
        verify(f.item().decodeEp(), never()).updatePreemption(anyLong(), any());
    }

    @Test
    void unknownCancelOutcomeKeepsTheAcknowledgedCancellationVisible() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 4L, "victim").orElse(null);
        assertNotNull(claim);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, RequestState.from(f.requestContext().snapshot()).state());
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, RequestState.from(f.requestContext().snapshot()).state());
        assertEquals(Stage.READY_TO_DELIVER, stage(f.requestContext()));
        assertFalse(f.requestContext().data().getFuture().isDone());
    }

    @Test
    void queuedCancellationResumesAfterExactPreemptionRelease() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 4L, "victim").orElse(null);
        assertNotNull(claim);
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, f.scheduler().cancel(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED).state());
        assertFalse(f.requestContext().data().getFuture().isDone());
        assertTrue(RequestProtocolTestSupport.releasePreemption(f.scheduler(), claim));
        assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
        assertEquals(RequestState.Phase.CANCELLED, RequestState.from(f.requestContext().snapshot()).state());
        assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
    }

    @Test
    void queuedDecodeRetirementFreezesTheResultBeforeAsyncCleanup() throws Exception {
        Fixture f = fixture(true, true);
        AtomicReference<String> cleanupThread = new AtomicReference<>();
        when(f.item().prefillEp().release(org.mockito.ArgumentMatchers.isNull(), any())).thenAnswer(call -> {
            cleanupThread.set(Thread.currentThread().getName());
            return true;
        });
        {

            RequestProtocolTestSupport.decodeRetired(f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
            assertFalse((f.requestContext().snapshot().phase() == Phase.OPEN));
            assertEquals(RequestState.Phase.FAILED, RequestState.from(f.requestContext().snapshot()).state());
            SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
            assertNotNull(cleanupThread.get());
            assertNotEquals(Thread.currentThread().getName(), cleanupThread.get());
        }
    }

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        Fixture f = fixture(true);
        var sender = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(sender.owner().data().getScheduler().tryStartSend(sender));
        RequestProtocolTestSupport.transport(sender.owner().data().getScheduler(), sender, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 4L, "victim").orElse(null);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), f.requestContext(), f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        verify(f.item().decodeEp(), never()).updatePreemption(anyLong(), argThat(update -> update.kind() == DecodeResources.PreemptionUpdate.Kind.FINISHED));
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryAckDuringPreemptionResumesWithTheOriginalBatchIdentity(boolean notificationFails) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        DeliveryClaim delivery = f.scheduler().claimDelivery(
                f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertNotNull(delivery);
        PreemptionRegistration preemption = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 9L, "victim").orElse(null);
        assertNotNull(preemption);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), preemption, PreemptionCancelPhase.CANCEL_IN_FLIGHT));

        RequestProtocolTestSupport.transport(delivery.owner().data().getScheduler(), delivery, DeliveryResult.delivered());
        assertSame(preemption, RequestProtocolTestSupport.activePreemption(f.requestContext()));
        assertFalse(f.requestContext().data().getFuture().isDone());
        assertNull(SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 8L));
        assertEquals(7L, SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 7L).batchId());

        when(f.item().decodeEp().updatePreemption(9L,
                DecodeResources.PreemptionUpdate.active(RESERVATION))).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
        var notificationFailure = new IllegalStateException("capacity listener failed");
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(RequestProtocolTestSupport.activePreemption(f.requestContext()));
            assertFalse(preemption.owner().data().getScheduler().runtime.signals().preemption(preemption).toCompletableFuture().isDone());
            if (notificationFails) { throw notificationFailure; }
            return null;
        }).when(assignedDecode).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        if (notificationFails) {
            assertSame(notificationFailure, assertThrows(IllegalStateException.class,
                    () -> RequestProtocolTestSupport.preemptionPhase(f.scheduler(), preemption, PreemptionCancelPhase.NOT_FOUND_STALE)));
        } else {
            assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        }
        assertTrue(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.DELIVERY_RESUMED,
                RequestProtocolTestSupport.awaitResult(preemption.owner().data().getScheduler().runtime.signals().preemption(preemption).toCompletableFuture()).outcome());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 7L).state());
        assertNull(SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 8L));
        // The new protocol commits the exact reconciliation once; no redundant proof replay.
        verify(f.item().decodeEp(), times(1)).updatePreemption(9L,
                DecodeResources.PreemptionUpdate.active(RESERVATION));
        verify(f.item().decodeEp()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void prefillBackedTerminalWaitsForTheExactDecodeClaimTransaction(RoleType role) throws Exception {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var sender = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(sender.owner().data().getScheduler().tryStartSend(sender));
        RequestProtocolTestSupport.transport(sender.owner().data().getScheduler(), sender, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 4L, "victim").orElse(null);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        when(f.item().decodeEp().updatePreemption(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION))).thenReturn(DecodeResources.ResourceChange.of(DecodeResources.MutationStatus.STALE, false),
                DecodeResources.ResourceChange.of(DecodeResources.MutationStatus.APPLIED, true));
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(RequestProtocolTestSupport.activePreemption(f.requestContext()));
            assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
            assertFalse(claim.owner().data().getScheduler().runtime.signals().preemption(claim).toCompletableFuture().isDone());
            return null;
        }).when(assignedDecode).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        var failed = new PrefillState.PrefillRequestStatus(f.item(), PrefillState.PrefillRequestStatus.Kind.FAILED, 9L);
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), role, failed);
        assertFalse((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(RequestProtocolTestSupport.preemptionFinished(claim));
        assertFalse(f.requestContext().data().getFuture().isDone());
        verify(f.item().decodeEp(), never()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), role, failed);
        verify(f.item().decodeEp(), times(2)).updatePreemption(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION));
        assertEquals(RequestState.Phase.FAILED, RequestState.from(f.requestContext().snapshot()).state());
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        RequestProtocolTestSupport.awaitCondition(() -> stage(f.requestContext()) == Stage.FINISHED);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                RequestProtocolTestSupport.awaitResult(claim.owner().data().getScheduler().runtime.signals().preemption(claim).toCompletableFuture()).outcome());
        verify(f.item().decodeEp()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        verify(f.item().prefillEp(), never()).release(org.mockito.ArgumentMatchers.isNull(), any());
        verify(f.item().decodeEp(), never()).release(any(), any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillFactsPublishReconciledPreemptionCapacityOutsideContext(boolean priorityCanceled) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var claim = f.requestContext().data().getScheduler().tryClaim(RESERVATION, 31L, "victim").orElse(null);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, priorityCanceled
                ? PreemptionCancelPhase.CANCEL_REQUESTED : PreemptionCancelPhase.NOT_FOUND_STALE));
        var update = priorityCanceled ? DecodeResources.PreemptionUpdate.canceled(RESERVATION)
                : DecodeResources.PreemptionUpdate.active(RESERVATION);
        when(f.item().decodeEp().updatePreemption(31L, update)).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(RequestProtocolTestSupport.activePreemption(f.requestContext()));
            assertEquals(priorityCanceled, RequestProtocolTestSupport.preemptionFinished(claim));
            assertFalse(claim.owner().data().getScheduler().runtime.signals().preemption(claim).toCompletableFuture().isDone());
            return null;
        }).when(assignedDecode).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);

        var requestStatus = priorityCanceled ? new PrefillState.PrefillRequestStatus(f.item(), PrefillState.PrefillRequestStatus.Kind.PRIORITY_CANCELED, 0L)
                : new PrefillState.PrefillRequestStatus(f.item(), PrefillState.PrefillRequestStatus.Kind.ACTIVE, 0L);
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), RoleType.PREFILL, requestStatus);

        verify(f.item().decodeEp()).updatePreemption(31L, update);
        verify(f.item().decodeEp()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        assertEquals(priorityCanceled, claim.owner().data().getScheduler().runtime.signals().preemption(claim).toCompletableFuture().isDone());
        assertEquals(priorityCanceled, f.requestContext().data().getFuture().isDone());
        if (priorityCanceled) {
            assertEquals(Stage.FINISHED, stage(f.requestContext()));
            assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
        } else {
            assertEquals(Stage.READY_TO_DELIVER, stage(f.requestContext()));
        }
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class, names = { "NOT_FOUND_STALE", "CANCEL_UNKNOWN" })
    void requestExpiryClosesPreemptionAndIgnoresLateCallbacks(PreemptionCancelPhase outcome) {
        Fixture f = fixture(true);
        RequestContext requestContext = f.requestContext();
        DeliveryClaim delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        PreemptionRegistration claim = requestContext.data().getScheduler().tryClaim(RESERVATION, 9L, "victim").orElse(null);
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, outcome));
        f.scheduler().cancel(requestContext, 7L, CancelReason.CLIENT_CANCELLED);
        var inactivity = requestContext.snapshot().deadlines().inactivity();
        assertNotNull(inactivity);
        RequestProtocolTestSupport.expireInactivity(f.scheduler(), requestContext, inactivity, RequestProtocolTestSupport.<Long>inspect(f.scheduler(), requestContext, "inactivityExpiresAtMsLocked"));
        assertFalse(delivery.owner().data().getScheduler().runtime.signals().delivery(delivery).toCompletableFuture().isDone(), "sender has not exited");
        RequestProtocolTestSupport.transport(delivery.owner().data().getScheduler(), delivery, DeliveryResult.notSent(new IllegalStateException("expired before send")));
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        RequestState ended = RequestState.from(requestContext.snapshot());
        assertEquals(RequestState.Phase.CANCELLED, ended.state());
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        assertTrue((stage(requestContext) == Stage.FINISHED));
        assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
        assertNull(requestContext.snapshot().route());
        assertEquals(CancelReason.CLIENT_CANCELLED, requestContext.snapshot().cancellation().reason(),
                "finished context preserves the original cancellation fact");
        RequestProtocolTestSupport.transport(delivery.owner().data().getScheduler(), delivery, DeliveryResult.delivered());
        assertFalse(RequestProtocolTestSupport.finishPreemption(f.scheduler(), claim, "late Cancel ACK"));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), requestContext, f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        RequestProtocolTestSupport.expireInactivity(f.scheduler(), requestContext, inactivity, Long.MAX_VALUE);
        assertEquals(ended, RequestState.from(requestContext.snapshot()));
        verify(f.item().decodeEp()).release(RESERVATION, DecodeResources.ReleaseReason.NOT_SENT);
        verify(f.item().prefillEp()).release(null, f.item());
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() throws Exception {
        Fixture f = fixture(true);
        DeliveryClaim claim = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(claim.owner().data().getScheduler().tryStartSend(claim));
        RequestProtocolTestSupport.transport(claim.owner().data().getScheduler(), claim, DeliveryResult.delivered());
        Response delivered = RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture());
        assertTrue(delivered.isSuccess());
        RequestProtocolTestSupport.expireInactiveRequest(f.scheduler(), f.requestContext(), RequestProtocolTestSupport.<Long>inspect(f.scheduler(), f.requestContext(), "inactivityExpiresAtMsLocked"));
        assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(f.requestContext().snapshot()).state());
        assertSame(delivered, RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()));
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        RequestProtocolTestSupport.awaitCondition(() -> stage(f.requestContext()) == Stage.FINISHED);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeEndDuringAdmissionPreservesTheEarlierCancellationCause(boolean retirement) {
        Fixture f = fixture(false);
        f.scheduler().cancel(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED);
        if (retirement) {
            RequestProtocolTestSupport.decodeRetired(f.scheduler(), f.requestContext(), f.item().decodeEp(), RESERVATION);
            SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        } else {
            RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), f.requestContext(), f.item().decodeEp(),
                    DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        }
        assertFalse((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(f.requestContext().data().getFuture().isDone());
        f.admission().finish();
        assertEquals(RequestState.Phase.CANCELLED, RequestState.from(f.requestContext().snapshot()).state());
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void workerProofExcludesLateAckBeforeUnlockedCleanupCommits(boolean preempting) throws Exception {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        RequestContext requestContext = f.requestContext();
        DeliveryClaim delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        if (preempting) {
            PreemptionRegistration claim = requestContext.data().getScheduler().tryClaim(RESERVATION, 9L, "victim").orElse(null);
            assertTrue(RequestProtocolTestSupport.preemptionPhase(f.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(requestContext));
            assertFalse((stage(requestContext) == Stage.FINISHED), "cleanup must precede final record commitment");
            assertFalse(RequestProtocolTestSupport.awaitResult(requestContext.data().getFuture()).isSuccess(), "late ACK cannot replace the terminal response");
            return DecodeResources.ResourceChange.of(DecodeResources.ReservationReleaseResult.RELEASED, true);
        }).when(assignedDecode).release(RESERVATION, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        assertTrue(delivery.owner().data().getScheduler().tryStartSend(delivery));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), requestContext, f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 42L));
        assertEquals(RequestState.Phase.FAILED, RequestState.from(requestContext.snapshot()).state());
        RequestProtocolTestSupport.transport(delivery.owner().data().getScheduler(), delivery, DeliveryResult.delivered());
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        RequestProtocolTestSupport.awaitCondition(() -> stage(requestContext) == Stage.FINISHED);
        assertTrue((stage(requestContext) == Stage.FINISHED));
        assertFalse(RequestProtocolTestSupport.awaitResult(requestContext.data().getFuture()).isSuccess());
        verify(f.item().prefillEp()).release(null, f.item());
    }

    @Test
    void completedReleaseEvidenceSurvivesLateFrontendFailureAndWaitsForTerminalEffects() {
        Fixture f = fixture(true);
        RequestContext context = f.requestContext();
        var delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L,
                RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(f.scheduler().tryStartSend(delivery));
        RequestProtocolTestSupport.transport(f.scheduler(), delivery, DeliveryResult.delivered());
        EndToken action;
        synchronized (context) {
            context.recordEngineEvidence(new EngineEvidence(f.item(), RESERVATION, RoleType.DECODE,
                    EngineKind.COMPLETED, 0L, null, System.currentTimeMillis()));
            var end = context.beginClose(context.snapshot().pendingEnd(), System.currentTimeMillis());
            action = ((Granted<EndToken>) end).value();
        }
        assertNotNull(action);
        assertNotNull(context.snapshot().deadlines().inactivity(), "unsettled resource retains inactivity watch");
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(f.scheduler(), "requestCleanup", context);
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        assertEquals(Stage.FINALIZING, stage(context),
                "resource evidence cannot archive before the terminal execution owner finishes");
        assertTrue(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
        assertNotNull(context.snapshot().deadlines().inactivity(), "pending notification obligation retains its watch");
        assertFalse(f.scheduler().completeExternal(context, RequestFuture.Completion.FAILURE, null,
                new IllegalStateException("late frontend failure")));
        assertTrue(context.snapshot().deliveryReleasable(), "late frontend failure cannot revoke completed execution evidence");
        var timer = f.scheduler().runtime.deadlines();
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(context));
            assertNotNull(context.snapshot().deadlines().inactivity());
            return null;
        }).when(timer).update(any(), any());
        var pendingPlan = context.snapshot().deadlines();
        org.mockito.Mockito.clearInvocations(timer);
        timer.update(context, pendingPlan);
        context.completeClose(action, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, true, true),
                System.currentTimeMillis());
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(f.scheduler(), "requestCleanup", context);
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        assertEquals(Stage.FINISHED, stage(context));
        verify(f.item().prefillEp(), times(1)).release(null, f.item());
        assertNull(context.snapshot().deadlines().inactivity());
        verify(timer).update(context, pendingPlan);
    }

    @Test
    void anotherSchedulerCannotAdvanceOrReleaseAnExactPreemptionClaim() {
        Fixture owner = fixture(true, true);
        Fixture other = fixture(true, true);
        PreemptionRegistration claim = owner.requestContext().data().getScheduler().tryClaim(RESERVATION, 4L, "victim").orElse(null);
        assertNotNull(claim);
        var before = owner.requestContext().snapshot();
        assertFalse(RequestProtocolTestSupport.preemptionPhase(other.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertFalse(RequestProtocolTestSupport.releasePreemption(other.scheduler(), claim));
        assertFalse(RequestProtocolTestSupport.finishPreemption(other.scheduler(), claim, "foreign callback"));
        assertEquals(before, owner.requestContext().snapshot());
        assertFalse(owner.requestContext().data().getFuture().isDone());
        assertFalse(other.requestContext().data().getFuture().isDone());
        assertTrue(RequestProtocolTestSupport.preemptionPhase(owner.scheduler(), claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(RequestProtocolTestSupport.releasePreemption(owner.scheduler(), claim));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cleanupProofSettlesDecodeBeforeAdvancingRequest(boolean resourceSettled) {
        Fixture f = fixture(true, true);
        var claim = f.scheduler().tryClaim(RESERVATION, 4L, "victim").orElseThrow();
        var proof = DecodeResources.PreemptionUpdate.fenced(RESERVATION);
        when(f.item().decodeEp().updatePreemption(4L, proof)).thenAnswer(call -> {
            assertFalse(contextFinished(f.requestContext()), "resource transaction precedes Context finality");
            assertFalse(f.requestContext().snapshot().preemption().finished());
            return DecodeResources.ResourceChange.of(resourceSettled
                    ? DecodeResources.MutationStatus.APPLIED : DecodeResources.MutationStatus.STALE, resourceSettled);
        });
        var before = f.requestContext().snapshot();
        boolean applied = f.scheduler().onPreemption(claim, new AbstractRequestScheduler.PreemptionEvent(
                null, false, f.item().decodeEp(), RESERVATION, "cleaned"));
        assertEquals(resourceSettled, applied);
        verify(f.item().decodeEp()).updatePreemption(4L, proof);
        if (resourceSettled) {
            SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
            assertTrue(RequestProtocolTestSupport.preemptionFinished(claim));
            assertFalse(RequestProtocolTestSupport.awaitResult(f.requestContext().data().getFuture()).isSuccess());
        } else {
            assertSame(before, f.requestContext().snapshot());
            assertFalse(f.requestContext().data().getFuture().isDone());
            verify(f.item().decodeEp(), never()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        }
    }

    @Test
    void cleanupProofCannotMixVictimsFromTheSamePreemptionAttempt() {
        Fixture f = fixture(true, true);
        var claim = f.scheduler().tryClaim(RESERVATION, 4L, "victim").orElseThrow();
        var wrongVictim = new DecodeResources.ReservationHandle(RESERVATION.endpointGenerationId(),
                RESERVATION.requestId() + 1L, RESERVATION.reservationToken());
        var before = f.requestContext().snapshot();
        assertFalse(f.scheduler().onPreemption(claim, new AbstractRequestScheduler.PreemptionEvent(
                null, false, f.item().decodeEp(), wrongVictim, "wrong victim")));
        assertSame(before, f.requestContext().snapshot());
        verify(f.item().decodeEp(), never()).updatePreemption(anyLong(), any());
        verify(f.item().decodeEp(), never()).publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3})
    void transientTerminalTimerFailureRetriesAfterActualResourceSettlement(int failures) {
        Fixture f = fixture(true, true);
        var context = f.requestContext();
        var timer = f.scheduler().runtime.deadlines();
        org.mockito.Mockito.clearInvocations(timer);
        var expected = new IllegalStateException("timer update failed");
        var attempts = new java.util.concurrent.atomic.AtomicInteger();
        doAnswer(call -> {
            if (attempts.incrementAndGet() <= failures) { throw expected; }
            return null;
        }).when(timer).update(any(), any());
        assertSame(expected, assertThrows(IllegalStateException.class, () -> f.scheduler().closeRequest(context,
                new EndEvidence(f.item(), EndKind.FAILURE, null, false, 0L,
                        org.flexlb.dao.loadbalance.StrategyErrorType.DISPATCH_FAILED, "original terminal failure", false, false), null)));
        SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        var original = context.snapshot().outcome();
        assertEquals("original terminal failure", original.detail());
        assertFalse(RequestProtocolTestSupport.awaitResult(context.data().getFuture()).isSuccess());
        if (failures == 3) {
            assertEquals(3, attempts.get(), "the first close flow has a three-attempt bound");
            assertEquals(Phase.ENDING, context.snapshot().phase());
            assertNotNull(context.snapshot().close().endOperation());
            assertTrue(context.snapshot().close().notificationsAttempted());
            assertNotNull(context.snapshot().deadlines().inactivity(), "real pending notification keeps its existing watch");
            assertSame(context, f.scheduler().requests.findActive(RESERVATION.requestId()));
            f.scheduler().requestCleanup(context);
            SchedulerTestSupport.awaitRequestWork(f.scheduler().runtime);
        }
        assertEquals(Phase.CLOSED, context.snapshot().phase(), "recovering notifications finish their existing owner");
        assertSame(original, context.snapshot().outcome(), "retries preserve the frozen outcome and first cause");
        assertTrue(context.snapshot().close().notificationsDone());
        assertTrue(context.snapshot().close().timersDone());
        assertNull(context.snapshot().deadlines().inactivity());
        verify(f.item().prefillEp(), times(1)).release(null, f.item());
        verify(f.item().decodeEp(), times(1)).release(RESERVATION, DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        verify(timer, times(failures + 1)).update(any(), any());
    }

    private static boolean contextFinished(RequestContext context) { return context.snapshot().phase() == Phase.CLOSED; }

    private static Fixture fixture(boolean finishAdmission) {
        return fixture(finishAdmission, false);
    }

    private static Fixture fixture(boolean finishAdmission, boolean queueScheduling) {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        RequestContext context = RequestProtocolTestSupport.context(config, RESERVATION.requestId());
        var publisher = mock(ResponseCompletionExecutor.class);
        var timer = mock(ExpirationTimer.class);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(publisher, context, timer);
        when(SchedulerTestSupport.cancelChannel(requestOwner).cancel(any(), anyLong(), any(), anyLong()))
                .thenReturn(java.util.concurrent.CompletableFuture.completedFuture(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED));
        when(publisher.tryRegister()).thenAnswer(call -> {
            var registration = mock(ResponseCompletionExecutor.CompletionRegistration.class);
            doAnswer(completion -> {
                ((java.util.function.BooleanSupplier) completion.getArgument(0)).getAsBoolean();
                return null;
            }).when(registration).submit(any());
            return registration;
        });
        var prefillServer = new org.flexlb.dao.loadbalance.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, new Response(), prefillServer, null, mock(PrefillEndpoint.class), RequestProtocolTestSupport.decodeEndpoint(), RESERVATION, System.currentTimeMillis());
        AdmissionHandle admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, requestOwner.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        if (finishAdmission) {
            admission.finish();
        }
        return new Fixture(requestOwner, context, item, admission);
    }

    private record Fixture(AbstractRequestScheduler scheduler, RequestContext requestContext, RequestRoute item, AdmissionHandle admission) {
    }
}
