package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
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
        org.mockito.Mockito.doReturn(timeoutEvidence).when((QueuedRequestScheduler) requestOwner).getLatestQueueWaitSnapshot();
        var admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        admission.recordDiagnostics(java.util.Map.of("cause", "earlier placement"));
        assertEquals("earlier placement", context.getSchedulingDiagnostics().get("cause"));
        requestOwner.cancelRequest(context, 0L, CancelReason.DEADLINE_EXCEEDED);
        admission.recordDiagnostics(java.util.Map.of("cause", "late placement"));
        assertEquals(timeoutEvidence, context.getSchedulingDiagnostics());
        assertFalse(context.future().isDone(), "cancellation must still wait for the active admission");
    }

    @Test
    void oldDecodeRetirementCannotRemoveANewSchedulingGeneration() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4202L);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), context, mock(ExpirationTimer.class));
        requestOwner.onDecodeGenerationRetired(context, RequestProtocolTestSupport.decodeEndpoint(), new DecodeResources.ReservationHandle(1L, 4202L, 7L));
        requestOwner.runtime.continuations().awaitIdle();
        assertTrue(context.isOpen());
        assertFalse(context.getFuture().isDone());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementRejectsAnotherSourceOrReservation(boolean wrongSource) {
        Fixture f = fixture(true, true);
        var source = wrongSource ? RequestProtocolTestSupport.decodeEndpoint() : f.item().decodeEp();
        var reservation = wrongSource ? RESERVATION : new DecodeResources.ReservationHandle(
                RESERVATION.endpointGenerationId(), RESERVATION.requestId(), RESERVATION.reservationToken() + 1);

        f.scheduler().onDecodeGenerationRetired(f.requestContext(), source, reservation);
        f.scheduler().runtime.continuations().awaitIdle();

        assertTrue(f.requestContext().isOpen());
        assertFalse(f.requestContext().future().isDone());
        assertSame(f.item(), f.requestContext().route());
        verify(f.item().prefillEp(), never()).releaseRequest(any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementPublishesTheOriginalPreemptionResolution(boolean cancelAlreadyUnknown) {
        Fixture f = fixture(true);
        PreemptionRegistration claim = f.requestContext().tryInstallPreemption(RESERVATION, 40L, "victim");
        assertNotNull(claim);
        if (cancelAlreadyUnknown) {
            assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        }
        assertNull(claim.resolvedRequestResult());

        f.scheduler().onDecodeGenerationRetired(f.requestContext(), f.item().decodeEp(), RESERVATION);
        f.scheduler().runtime.continuations().awaitIdle();

        assertTrue(claim.isFinished());
        assertNotNull(claim.resolvedRequestResult(), "retirement must notify the detached original claim");
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                claim.resolvedRequestResult().outcome());
        assertEquals(RequestContext.RequestStage.FINISHED, f.requestContext().stage());
        assertFalse(f.requestContext().future().join().isSuccess());
        verify(f.item().decodeEp(), never()).reconcilePreemptionResources(anyLong(), any());
    }

    @Test
    void unknownCancelOutcomeKeepsTheAcknowledgedCancellationVisible() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim = f.requestContext().tryInstallPreemption(RESERVATION, 4L, "victim");
        assertNotNull(claim);
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, f.requestContext().snapshot().state());
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, f.requestContext().snapshot().state());
        assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, f.requestContext().stage());
        assertFalse(f.requestContext().future().isDone());
    }

    @Test
    void queuedCancellationResumesAfterExactPreemptionRelease() {
        Fixture f = fixture(true, true);
        PreemptionRegistration claim = f.requestContext().tryInstallPreemption(RESERVATION, 4L, "victim");
        assertNotNull(claim);
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, f.scheduler().cancelRequest(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED).state());
        assertFalse(f.requestContext().future().isDone());
        assertTrue(f.scheduler().releasePreemption(claim));
        assertFalse(f.requestContext().future().join().isSuccess());
        assertEquals(RequestState.Phase.CANCELLED, f.requestContext().snapshot().state());
        assertFalse(f.requestContext().future().join().isSuccess());
    }

    @Test
    void queuedDecodeRetirementFreezesTheResultBeforeAsyncCleanup() throws Exception {
        Fixture f = fixture(true, true);
        AtomicReference<String> cleanupThread = new AtomicReference<>();
        when(f.item().prefillEp().releaseRequest(any())).thenAnswer(call -> {
            cleanupThread.set(Thread.currentThread().getName());
            return true;
        });
        try (RequestContinuationExecutor continuations = new RequestContinuationExecutor()) {
            org.springframework.test.util.ReflectionTestUtils.setField(f.scheduler(), "continuations", continuations);
            org.mockito.Mockito.doCallRealMethod().when(f.scheduler()).onDecodeGenerationRetired(any(), any(), any());
            f.scheduler().onDecodeGenerationRetired(f.requestContext(), f.item().decodeEp(), RESERVATION);
            assertFalse(f.requestContext().isOpen());
            assertEquals(RequestState.Phase.FAILED, f.requestContext().snapshot().state());
            continuations.awaitIdle();
            assertNotNull(cleanupThread.get());
            assertNotEquals(Thread.currentThread().getName(), cleanupThread.get());
        }
    }

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        Fixture f = fixture(true);
        var sender = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(sender.item.ctx().scheduler().tryStartSend(sender));
        sender.item.ctx().scheduler().completeDelivery(sender, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim = f.requestContext().tryInstallPreemption(RESERVATION, 4L, "victim");
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), f.requestContext(), f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        verify(f.item().decodeEp(), never()).reconcilePreemptionResources(anyLong(), argThat(update -> update.kind() == DecodeResources.PreemptionUpdate.Kind.FINISHED));
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
        assertTrue(claim.isFinished());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryAckDuringPreemptionResumesWithTheOriginalBatchIdentity(boolean notificationFails) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        RequestContext.DeliveryClaim delivery = f.scheduler().claimDelivery(
                f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertNotNull(delivery);
        PreemptionRegistration preemption = f.requestContext().tryInstallPreemption(RESERVATION, 9L, "victim");
        assertNotNull(preemption);
        assertTrue(f.scheduler().updatePreemption(preemption, PreemptionCancelPhase.CANCEL_IN_FLIGHT));

        delivery.item.ctx().scheduler().completeDelivery(delivery, DeliveryResult.delivered());
        assertSame(preemption, f.requestContext().preemption());
        assertFalse(f.requestContext().future().isDone());
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 8L));
        assertEquals(7L, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 7L).batchId());

        when(f.item().decodeEp().reconcilePreemptionResources(9L,
                DecodeResources.PreemptionUpdate.active(RESERVATION))).thenReturn(true);
        var notificationFailure = new IllegalStateException("capacity listener failed");
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(f.requestContext().preemption());
            assertFalse(preemption.requestResolution().toCompletableFuture().isDone());
            if (notificationFails) { throw notificationFailure; }
            return null;
        }).when(assignedDecode).publishCapacityRelease();
        if (notificationFails) {
            assertSame(notificationFailure, assertThrows(IllegalStateException.class,
                    () -> f.scheduler().updatePreemption(preemption, PreemptionCancelPhase.NOT_FOUND_STALE)));
        } else {
            assertTrue(f.scheduler().updatePreemption(preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        }
        assertTrue(f.requestContext().future().join().isSuccess());
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.DELIVERY_RESUMED,
                preemption.requestResolution().toCompletableFuture().join().outcome());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 7L).state());
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.scheduler()).getRequestState(RESERVATION.requestId(), 8L));
        verify(f.item().decodeEp(), times(2)).reconcilePreemptionResources(9L,
                DecodeResources.PreemptionUpdate.active(RESERVATION));
        verify(f.item().decodeEp()).publishCapacityRelease();
    }

    @Test
    void prefillBackedTerminalWaitsForTheExactDecodeClaimTransaction() {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var sender = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(sender.item.ctx().scheduler().tryStartSend(sender));
        sender.item.ctx().scheduler().completeDelivery(sender, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        PreemptionRegistration claim = f.requestContext().tryInstallPreemption(RESERVATION, 4L, "victim");
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        when(f.item().decodeEp().reconcilePreemptionResources(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION))).thenReturn(false, true);
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(f.requestContext().preemption());
            assertTrue(claim.isFinished());
            assertFalse(claim.requestResolution().toCompletableFuture().isDone());
            return null;
        }).when(assignedDecode).publishCapacityRelease();
        var failed = PrefillState.PrefillRequestStatus.terminal(f.item(), PrefillState.PrefillRequestStatus.Kind.FAILED, 9L);
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), RoleType.PREFILL, failed);
        assertFalse((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
        assertFalse(claim.isFinished());
        assertFalse(f.requestContext().future().isDone());
        verify(f.item().decodeEp(), never()).publishCapacityRelease();
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), RoleType.PREFILL, failed);
        verify(f.item().decodeEp(), times(2)).reconcilePreemptionResources(4L, DecodeResources.PreemptionUpdate.finished(RESERVATION));
        assertEquals(RequestState.Phase.FAILED, f.requestContext().snapshot().state());
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
        assertEquals(org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END,
                claim.requestResolution().toCompletableFuture().join().outcome());
        verify(f.item().decodeEp()).publishCapacityRelease();
        verify(f.item().decodeEp(), never()).release(any(), any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillFactsPublishReconciledPreemptionCapacityOutsideContext(boolean priorityCanceled) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        var claim = f.requestContext().tryInstallPreemption(RESERVATION, 31L, "victim");
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(f.scheduler().updatePreemption(claim, priorityCanceled
                ? PreemptionCancelPhase.CANCEL_REQUESTED : PreemptionCancelPhase.NOT_FOUND_STALE));
        var update = priorityCanceled ? DecodeResources.PreemptionUpdate.canceled(RESERVATION)
                : DecodeResources.PreemptionUpdate.active(RESERVATION);
        when(f.item().decodeEp().reconcilePreemptionResources(31L, update)).thenReturn(true);
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(f.requestContext()));
            assertNull(f.requestContext().preemption());
            assertEquals(priorityCanceled, claim.isFinished());
            assertFalse(claim.requestResolution().toCompletableFuture().isDone());
            return null;
        }).when(assignedDecode).publishCapacityRelease();

        var requestStatus = priorityCanceled ? PrefillState.PrefillRequestStatus.terminal(f.item(),
                PrefillState.PrefillRequestStatus.Kind.PRIORITY_CANCELED, 0L)
                : PrefillState.PrefillRequestStatus.active(f.item());
        RequestProtocolTestSupport.applyPrefillStatus(f.scheduler(), f.requestContext(), f.item().prefillEp(), RoleType.PREFILL, requestStatus);

        verify(f.item().decodeEp()).reconcilePreemptionResources(31L, update);
        verify(f.item().decodeEp()).publishCapacityRelease();
        assertEquals(priorityCanceled, claim.requestResolution().toCompletableFuture().isDone());
        assertEquals(priorityCanceled, f.requestContext().future().isDone());
        if (priorityCanceled) {
            assertEquals(RequestContext.RequestStage.FINISHED, f.requestContext().stage());
            assertFalse(f.requestContext().future().join().isSuccess());
        } else {
            assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, f.requestContext().stage());
        }
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class, names = { "NOT_FOUND_STALE", "CANCEL_UNKNOWN" })
    void requestExpiryClosesPreemptionAndIgnoresLateCallbacks(PreemptionCancelPhase outcome) {
        Fixture f = fixture(true);
        RequestContext requestContext = f.requestContext();
        RequestContext.DeliveryClaim delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        PreemptionRegistration claim = requestContext.tryInstallPreemption(RESERVATION, 9L, "victim");
        assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(f.scheduler().updatePreemption(claim, outcome));
        f.scheduler().cancelRequest(requestContext, 7L, CancelReason.CLIENT_CANCELLED);
        var inactivity = mock(ExpirationTimer.InactivityDeadline.class);
        assertTrue(requestContext.installInactivityDeadline(inactivity));
        RequestProtocolTestSupport.expireInactivity(f.scheduler(), requestContext, inactivity, RequestProtocolTestSupport.<Long>inspect(f.scheduler(), requestContext, "inactivityExpiresAtMsLocked"));
        assertFalse(delivery.settlement().toCompletableFuture().isDone(), "sender has not exited");
        delivery.item.ctx().scheduler().completeDelivery(delivery, DeliveryResult.notSent(new IllegalStateException("expired before send")));
        f.scheduler().runtime.continuations().awaitIdle();
        RequestState ended = requestContext.snapshot();
        assertEquals(RequestState.Phase.CANCELLED, ended.state());
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((requestContext.stage() == RequestContext.RequestStage.FINISHED));
        assertTrue(claim.isFinished());
        assertNull(requestContext.activeRoute());
        assertEquals(CancelReason.CLIENT_CANCELLED, requestContext.cancellationReason(),
                "finished context preserves the original cancellation fact");
        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, () -> delivery.item.ctx().scheduler().completeDelivery(delivery, DeliveryResult.delivered()));
        assertFalse(f.scheduler().completePreemption(claim, "late Cancel ACK"));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), requestContext, f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        RequestProtocolTestSupport.expireInactivity(f.scheduler(), requestContext, inactivity, Long.MAX_VALUE);
        assertEquals(ended, requestContext.snapshot());
        verify(f.item().decodeEp()).release(RESERVATION, DecodeResources.ReleaseReason.NOT_SENT);
        verify(f.item().prefillEp()).releaseRequest(f.item());
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() {
        Fixture f = fixture(true);
        RequestContext.DeliveryClaim claim = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(claim.item.ctx().scheduler().tryStartSend(claim));
        claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
        Response delivered = f.requestContext().future().join();
        assertTrue(delivered.isSuccess());
        RequestProtocolTestSupport.expireInactiveRequest(f.scheduler(), f.requestContext(), RequestProtocolTestSupport.<Long>inspect(f.scheduler(), f.requestContext(), "inactivityExpiresAtMsLocked"));
        assertEquals(RequestState.Phase.TIMED_OUT, f.requestContext().snapshot().state());
        assertSame(delivered, f.requestContext().future().join());
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeEndDuringAdmissionPreservesTheEarlierCancellationCause(boolean retirement) {
        Fixture f = fixture(false);
        f.scheduler().cancelRequest(f.requestContext(), 0L, CancelReason.CLIENT_CANCELLED);
        if (retirement) {
            f.scheduler().onDecodeGenerationRetired(f.requestContext(), f.item().decodeEp(), RESERVATION);
            f.scheduler().runtime.continuations().awaitIdle();
        } else {
            RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), f.requestContext(), f.item().decodeEp(),
                    DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        }
        assertFalse((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
        assertFalse(f.requestContext().future().isDone());
        f.admission().finish();
        assertEquals(RequestState.Phase.CANCELLED, f.requestContext().snapshot().state());
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((f.requestContext().stage() == RequestContext.RequestStage.FINISHED));
        assertFalse(f.requestContext().future().join().isSuccess());
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void workerProofExcludesLateAckBeforeUnlockedCleanupCommits(boolean preempting) {
        Fixture f = fixture(true);
        var assignedDecode = f.item().decodeEp();
        RequestContext requestContext = f.requestContext();
        RequestContext.DeliveryClaim delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, RequestProtocolTestSupport.handoff(() -> true));
        if (preempting) {
            PreemptionRegistration claim = requestContext.tryInstallPreemption(RESERVATION, 9L, "victim");
            assertTrue(f.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(requestContext));
            assertFalse((requestContext.stage() == RequestContext.RequestStage.FINISHED), "cleanup must precede final record commitment");
            assertFalse(requestContext.future().join().isSuccess(), "late ACK cannot replace the terminal response");
            return DecodeResources.ReservationReleaseResult.RELEASED;
        }).when(assignedDecode).release(RESERVATION, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        assertTrue(delivery.item.ctx().scheduler().tryStartSend(delivery));
        RequestProtocolTestSupport.applyDecodeStatus(f.scheduler(), requestContext, f.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 42L));
        assertEquals(RequestState.Phase.FAILED, requestContext.snapshot().state());
        delivery.item.ctx().scheduler().completeDelivery(delivery, DeliveryResult.delivered());
        f.scheduler().runtime.continuations().awaitIdle();
        assertTrue((requestContext.stage() == RequestContext.RequestStage.FINISHED));
        assertFalse(requestContext.future().join().isSuccess());
        verify(f.item().prefillEp()).releaseRequest(f.item());
    }

    @Test
    void completedReleaseEvidenceSurvivesLateFrontendFailureAndWaitsForTerminalEffects() {
        Fixture f = fixture(true);
        RequestContext context = f.requestContext();
        var delivery = f.scheduler().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L,
                RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(delivery.item.ctx().scheduler().tryStartSend(delivery));
        delivery.item.ctx().scheduler().completeDelivery(delivery, DeliveryResult.delivered());
        TerminalAction action;
        synchronized (context) {
            action = f.scheduler().claimFinalizationLocked(context,
                    context.decideRequestEndLocked(DeferredTerminal.worker(WorkerTerminalSource.DECODE_ENDPOINT, true, 0L)));
        }
        assertNotNull(action);
        var inactivity = mock(ExpirationTimer.InactivityDeadline.class);
        assertTrue(context.installInactivityDeadline(inactivity));
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(context));
            assertTrue(context.inactivityDeadlineAtMs().isEmpty(), "archive ownership closes timer registration");
            assertFalse(context.installInactivityDeadline(mock(ExpirationTimer.InactivityDeadline.class)));
            return null;
        }).when(inactivity).cancel();
        delivery.recordWorkerCompletion(f.item());
        delivery.item.ctx().scheduler().settleDelivery(delivery);
        f.scheduler().runtime.continuations().awaitIdle();
        assertEquals(RequestContext.RequestStage.FINALIZING, context.stage(),
                "resource evidence cannot archive before the terminal execution owner finishes");
        f.scheduler().onResponseUndeliverable(context);
        assertFalse(delivery.cleanupRequired(), "late frontend failure cannot revoke completed execution evidence");
        f.scheduler().finalizationEffects(action, null).run();
        f.scheduler().runtime.continuations().awaitIdle();
        assertEquals(RequestContext.RequestStage.FINISHED, context.stage());
        verify(f.item().prefillEp(), times(1)).releaseRequest(f.item());
    }

    @Test
    void anotherSchedulerCannotAdvanceOrReleaseAnExactPreemptionClaim() {
        Fixture owner = fixture(true, true);
        Fixture other = fixture(true, true);
        PreemptionRegistration claim = owner.requestContext().tryInstallPreemption(RESERVATION, 4L, "victim");
        assertNotNull(claim);
        var before = owner.requestContext().snapshot();
        assertFalse(other.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertFalse(other.scheduler().releasePreemption(claim));
        assertFalse(other.scheduler().completePreemption(claim, "foreign callback"));
        assertEquals(before, owner.requestContext().snapshot());
        assertFalse(owner.requestContext().future().isDone());
        assertFalse(other.requestContext().future().isDone());
        assertTrue(owner.scheduler().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(owner.scheduler().releasePreemption(claim));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cleanupProofSettlesDecodeBeforeAdvancingRequest(boolean resourceSettled) {
        var scheduler = mock(AbstractRequestScheduler.class);
        var endpoint = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
        var claim = mock(PreemptionRegistration.class);
        when(claim.attemptToken()).thenReturn(4L);
        when(claim.scheduler()).thenReturn(scheduler);
        when(claim.requestId()).thenReturn(RESERVATION.requestId());
        var proof = org.flexlb.balance.endpoint.DecodeResources.PreemptionUpdate.fenced(RESERVATION);
        when(endpoint.updatePreemption(4L, proof)).thenReturn(resourceSettled);
        when(scheduler.completePreemption(claim, "cleaned")).thenReturn(true);
        org.mockito.Mockito.doCallRealMethod().when(scheduler)
                .onPreemptionCleanupProven(claim, endpoint, RESERVATION, "cleaned");
        assertEquals(resourceSettled, scheduler.onPreemptionCleanupProven(claim, endpoint, RESERVATION, "cleaned"));
        var order = org.mockito.Mockito.inOrder(endpoint, scheduler);
        order.verify(endpoint).updatePreemption(4L, proof);
        order.verify(scheduler, resourceSettled ? times(1) : org.mockito.Mockito.never())
                .completePreemption(claim, "cleaned");
    }

    @Test
    void cleanupProofCannotMixVictimsFromTheSamePreemptionAttempt() {
        var scheduler = mock(AbstractRequestScheduler.class, org.mockito.Mockito.CALLS_REAL_METHODS);
        var endpoint = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
        var claim = mock(PreemptionRegistration.class);
        when(claim.scheduler()).thenReturn(scheduler);
        when(claim.requestId()).thenReturn(RESERVATION.requestId() + 1);
        when(claim.attemptToken()).thenReturn(4L);
        assertThrows(IllegalArgumentException.class,
                () -> scheduler.onPreemptionCleanupProven(claim, endpoint, RESERVATION, "wrong victim"));
        org.mockito.Mockito.verifyNoInteractions(endpoint);
    }

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
        when(publisher.tryRegister()).thenAnswer(call -> new ResponseCompletionExecutor.CompletionRegistration(publisher));
        doAnswer(call -> {
            ((java.util.function.BooleanSupplier) call.getArgument(1)).getAsBoolean();
            return null;
        }).when(publisher).submit(any(), any());
        var prefillServer = new org.flexlb.dao.loadbalance.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillServer, null, mock(PrefillEndpoint.class), RequestProtocolTestSupport.decodeEndpoint(), RESERVATION, System.currentTimeMillis());
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
