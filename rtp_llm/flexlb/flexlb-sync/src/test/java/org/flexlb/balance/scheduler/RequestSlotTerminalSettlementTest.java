package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class RequestSlotTerminalSettlementTest {
    private static final DecodeEndpoint.ReservationHandle RESERVATION =
            new DecodeEndpoint.ReservationHandle(1L, 2L, 3L);

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        Fixture f = fixture(true);
        f.slot().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        PreemptionRegistration claim = f.slot().tryInstallPreemption(3L, 4L, "victim");
        assertTrue(f.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));

        f.slot().processDecodeStatus(f.item().decodeEp(), DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 0L));

        verify(f.item().decodeEp(), never()).updatePreemption(
                anyLong(),
                argThat(update -> update.kind() == DecodeEndpoint.PreemptionUpdate.Kind.FINISHED));
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
        assertTrue(claim.isFinished());
    }

    @Test
    void prefillBackedTerminalWaitsForTheExactDecodeClaimTransaction() {
        Fixture f = fixture(true);
        f.slot().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        PreemptionRegistration claim = f.slot().tryInstallPreemption(3L, 4L, "victim");
        assertTrue(f.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        when(f.item().decodeEp().updatePreemption(4L, DecodeEndpoint.PreemptionUpdate.finished(RESERVATION))).thenReturn(false, true);
        var failed = PrefillState.WorkerStatusFact.terminal(f.item(), PrefillState.WorkerStatusFact.Kind.FAILED, 9L);

        f.slot().processPrefillStatus(f.item().prefillEp(), RoleType.PREFILL, failed);
        assertFalse(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
        assertFalse(claim.isFinished());
        assertFalse(f.slot().future().isDone());

        f.slot().processPrefillStatus(f.item().prefillEp(), RoleType.PREFILL, failed);
        verify(f.item().decodeEp(), times(2)).updatePreemption(4L, DecodeEndpoint.PreemptionUpdate.finished(RESERVATION));
        assertEquals(RequestState.Phase.FAILED, f.slot().snapshot().state());
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class, names = {"NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void requestExpiryClosesPreemptionAndIgnoresLateCallbacks(PreemptionCancelPhase outcome) {
        Fixture f = fixture(true);
        RequestSlot slot = f.slot();
        RequestSlot.DeliveryClaim delivery = slot.claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        PreemptionRegistration claim = slot.tryInstallPreemption(3L, 9L, "victim");
        assertTrue(slot.updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(slot.updatePreemption(claim, outcome));
        slot.cancelRequest(7L, CancelReason.CLIENT_CANCELLED);
        var inactivity = mock(ExpirationTimer.InactivityDeadline.class);
        assertTrue(slot.installInactivityDeadline(inactivity));

        slot.onInactivityDeadline(inactivity,
                RequestLifecycleTestSupport.<Long>inspect(slot, "inactivityExpiresAtMsLocked"));
        RequestState ended = slot.snapshot();
        assertEquals(RequestState.Phase.CANCELLED, ended.state());
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(slot, "isTerminalRecordLocked"));
        assertTrue(claim.isFinished());
        assertNull(slot.activeItem());
        assertFalse(RequestLifecycleTestSupport.<Boolean>inspect(slot, "hasCancellationFirstCauseLocked"));

        delivery.complete(DeliveryResult.delivered());
        assertFalse(slot.completePreemption(claim, "late Cancel ACK"));
        slot.processDecodeStatus(f.item().decodeEp(), DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 0L));
        slot.onInactivityDeadline(inactivity, Long.MAX_VALUE);
        assertEquals(ended, slot.snapshot());
        verify(f.item().decodeEp()).release(RESERVATION, DecodeEndpoint.ReleaseReason.EXPIRED);
        verify(f.item().prefillEp()).expireCommittedItem(f.item());
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() {
        Fixture f = fixture(true);
        RequestSlot.DeliveryClaim claim = f.slot().claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        claim.complete(DeliveryResult.delivered());
        Response delivered = f.slot().future().join();
        assertTrue(delivered.isSuccess());

        f.slot().expireInactiveRequest(
                RequestLifecycleTestSupport.<Long>inspect(f.slot(), "inactivityExpiresAtMsLocked"));

        assertEquals(RequestState.Phase.TIMED_OUT, f.slot().snapshot().state());
        assertSame(delivered, f.slot().future().join());
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
    }

    @Test
    void workerTerminalDuringAdmissionPreservesTheEarlierCancellationCause() {
        Fixture f = fixture(false);
        f.slot().cancelRequest(0L, CancelReason.CLIENT_CANCELLED);
        f.slot().processDecodeStatus(f.item().decodeEp(), DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 0L));
        assertFalse(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
        assertFalse(f.slot().future().isDone());

        f.admission().close();

        assertEquals(RequestState.Phase.CANCELLED, f.slot().snapshot().state());
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(f.slot(), "isTerminalRecordLocked"));
        assertFalse(f.slot().future().join().isSuccess());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void workerProofExcludesLateAckBeforeUnlockedCleanupCommits(boolean preempting) {
        Fixture f = fixture(true);
        RequestSlot slot = f.slot();
        RequestSlot.DeliveryClaim delivery = slot.claimDelivery(f.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        if (preempting) {
            PreemptionRegistration claim = slot.tryInstallPreemption(3L, 9L, "victim");
            assertTrue(slot.updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(slot));
            assertFalse(RequestLifecycleTestSupport.<Boolean>inspect(slot, "isTerminalRecordLocked"), "cleanup must precede final record commitment");
            delivery.complete(DeliveryResult.delivered());
            assertFalse(slot.future().isDone(), "late ACK cannot publish after cleanup was claimed");
            return DecodeEndpoint.ReservationReleaseResult.RELEASED;
        }).when(f.item().decodeEp()).release(RESERVATION, DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED);

        slot.processDecodeStatus(f.item().decodeEp(), DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 42L));

        assertEquals(RequestState.Phase.FAILED, slot.snapshot().state());
        assertTrue(RequestLifecycleTestSupport.<Boolean>inspect(slot, "isTerminalRecordLocked"));
        assertFalse(slot.future().join().isSuccess());
        verify(f.item().prefillEp(), never()).releaseCommittedItem(any());
        verify(f.item().prefillEp(), never()).expireCommittedItem(any());
    }

    private static Fixture fixture(boolean finishAdmission) {
        var config = SchedulingTestConfig.newConfig();
        BalanceContext context = RequestLifecycleTestSupport.context(config, RESERVATION.requestId());
        var publisher = mock(RequestCompletionPublisher.class);
        var timer = mock(ExpirationTimer.class);
        RequestSlot slot = new RequestSlot(publisher, RESERVATION.requestId(), timer,
                new RequestTerminalCleanup(timer), () -> { });
        when(publisher.tryReservePublication(any(), any())).thenAnswer(call ->
                new RequestCompletionPublisher.PublicationPermit(publisher, slot, call.getArgument(1)));
        doAnswer(call -> { ((RequestCompletionPublisher.SelectedPublication) call.getArgument(0)).complete(); return null; })
                .when(publisher).submit(any());
        doCallRealMethod().when(publisher).submitDelivery(any(), any());
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), null, null,
                mock(PrefillEndpoint.class), mock(DecodeEndpoint.class), RESERVATION, System.currentTimeMillis());
        slot.configureInactivityTimeout(60_000L);
        AdmissionHandle admission = slot.tryBeginAdmissionHandle();
        assertNotNull(admission);
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, slot.commitRoute(item, () -> true));
        if (finishAdmission) { admission.close(); }
        return new Fixture(slot, item, admission);
    }

    private record Fixture(RequestSlot slot, ScheduledRequest item, AdmissionHandle admission) { }
}
