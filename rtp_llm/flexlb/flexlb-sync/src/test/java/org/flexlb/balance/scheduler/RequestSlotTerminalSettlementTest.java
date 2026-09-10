package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestSlotTerminalSettlementTest {

    private static final DecodeEndpoint.ReservationHandle RESERVATION =
            new DecodeEndpoint.ReservationHandle(1L, 2L, 3L);

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DeferredTerminal terminal = DeferredTerminal.worker(
                WorkerTerminalSource.DECODE_ENDPOINT_SETTLED, true, 0L);

        assertTrue(RequestSlot.terminalOwnsDecodeSettlement(
                terminal, decode, 4L, RESERVATION));
        verifyNoInteractions(decode);
    }

    @Test
    void prefillBackedTerminalDelegatesToTheExactDecodeClaimTransaction() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DeferredTerminal terminal = DeferredTerminal.worker(
                WorkerTerminalSource.PREFILL_BACKED, false, 9L);
        when(decode.reconcilePriorityVictimFinished(4L, RESERVATION))
                .thenReturn(false);

        assertFalse(RequestSlot.terminalOwnsDecodeSettlement(
                terminal, decode, 4L, RESERVATION));
        verify(decode).reconcilePriorityVictimFinished(4L, RESERVATION);
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class,
            names = {"NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void requestExpiryClaimsUnknownPreemptionAndRejectsEveryLateCallback(
            PreemptionCancelPhase outcome) {
        Fixture fixture = fixture();
        RequestSlot slot = fixture.slot();
        var inactivity = mock(ExpirationTimer.InactivityDeadline.class);
        var timer = mock(ExpirationTimer.class);
        synchronized (slot) {
            slot.startBatchEnqueue(7L);
            PreemptionRegistration claim = slot.tryInstallPreemption(
                    RESERVATION.reservationToken(), 9L, "priority victim");
            assertNotNull(claim);
            assertEquals(RequestSlot.PreemptionReduction.Status.NONE,
                    slot.applyPreemptionPhase(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT).status());
            assertEquals(RequestSlot.PreemptionReduction.Status.NONE,
                    slot.applyPreemptionPhase(claim, outcome).status());
            slot.markCancellationRequested(CancelReason.CLIENT_CANCELLED, "original client cancellation");
            assertTrue(slot.installInactivityDeadline(inactivity));

            TerminalAction action = slot.beginTerminalizing(true, false, false, null,
                    owner -> owner.cancel("original client cancellation; request inactive"), null);
            assertNotNull(action);
            assertSame(claim, action.preemption());
            assertTrue(claim.isSettled());
            assertEquals(CancelReason.CLIENT_CANCELLED, slot.requireCancellationFirstCause());
            assertEquals(RequestSlot.PreemptionReduction.Status.STALE,
                    slot.reduceDeliveryConfirmed(7L).status());
            assertEquals(RequestSlot.PreemptionReduction.Status.STALE,
                    slot.applyPreemptionTombstone(claim, "late Cancel ACK").status());
            assertEquals(RequestSlot.PreemptionReduction.Status.STALE,
                    slot.reduceWorkerTerminal(fixture.item(), DeferredTerminal.worker(
                            WorkerTerminalSource.DECODE_ENDPOINT_SETTLED, true, 0L)).status());
            assertFalse(slot.expireInactivityDeadline(inactivity));
            action.terminalResources().release(timer);
            action.terminalResources().release(timer);
            verify(timer).cancel(inactivity);

            TombstoneResult settled = slot.finishTombstone(action);
            assertEquals(RequestState.Phase.CANCELLED, settled.terminal().state());
            assertNull(settled.transitionFailure());
            assertTrue(slot.isTombstone());
            assertNull(slot.activeItem());
            assertFalse(slot.hasCancellationFirstCause());
            assertNull(slot.beginTerminalizing(true, false, false, null,
                    owner -> owner.timeout("duplicate expiry"), null));
        }
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() {
        Fixture fixture = fixture();
        RequestSlot slot = fixture.slot();
        synchronized (slot) {
            slot.startBatchEnqueue(7L);
            slot.markDeliveryConfirmed();
            Response delivered = new Response();
            delivered.setSuccess(true);
            assertTrue(slot.future().completeOwned(delivered));

            TerminalAction action = slot.beginTerminalizing(true, false, false, null,
                    owner -> owner.timeout("request inactive"), new Response());
            assertNotNull(action);
            assertNull(action.publication());
            assertNull(action.response());
            assertEquals(RequestState.Phase.TIMED_OUT,
                    slot.finishTombstone(action).terminal().state());
            assertSame(delivered, slot.future().join());
            assertTrue(slot.isTombstone());
        }
    }

    @Test
    void workerTerminalDuringAdmissionPreservesTheEarlierCancellationCause() {
        Fixture fixture = fixture(false);
        RequestSlot slot = fixture.slot();
        DeferredTerminal workerTerminal = DeferredTerminal.worker(
                WorkerTerminalSource.DECODE_ENDPOINT_SETTLED, true, 0L);
        synchronized (slot) {
            assertTrue(slot.deferCancellationDuringAdmission(
                    CancelReason.CLIENT_CANCELLED, "first client cancellation"));
            assertEquals(RequestSlot.PreemptionReduction.Status.NONE,
                    slot.reduceWorkerTerminal(fixture.item(), workerTerminal).status());

            RequestSlot.AdmissionMutationCompletion completion =
                    slot.completeAdmissionMutation(fixture.admission());
            assertTrue(completion.owned());
            assertNull(completion.cancellationToResume(), "terminal proof needs no extra Cancel send");
            assertSame(workerTerminal, completion.pendingTerminal());
            assertEquals(CancelReason.CLIENT_CANCELLED, slot.requireCancellationFirstCause());
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, slot.snapshot().state());

            TerminalAction action = slot.beginTerminalizing(false, false, false, null,
                    owner -> owner.cancel("first client cancellation; worker completed"), null);
            assertNotNull(action);
            assertEquals(RequestState.Phase.CANCELLED,
                    slot.finishTombstone(action).terminal().state());
        }
    }

    private static Fixture fixture() {
        return fixture(true);
    }

    private static Fixture fixture(boolean completeAdmission) {
        var config = SchedulingTestConfig.newConfig();
        BalanceContext context = RequestLifecycleTestSupport.context(config, RESERVATION.requestId());
        RequestSlot slot = new RequestSlot(mock(RequestCompletionPublisher.class), RESERVATION.requestId());
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), null, null,
                mock(PrefillEndpoint.class), mock(DecodeEndpoint.class), RESERVATION,
                System.currentTimeMillis());
        AdmissionMutation admission;
        synchronized (slot) {
            slot.configureInactivityTimeout(60_000L);
            admission = slot.tryBeginAdmissionMutation((owner, response) -> { }, owner -> { });
            assertNotNull(admission);
            assertTrue(slot.tryBindItemForPublication(item));
            if (completeAdmission) {
                slot.completeAdmissionMutation(admission);
            }
        }
        return new Fixture(slot, item, admission);
    }

    private record Fixture(RequestSlot slot, ScheduledRequest item, AdmissionMutation admission) { }
}
