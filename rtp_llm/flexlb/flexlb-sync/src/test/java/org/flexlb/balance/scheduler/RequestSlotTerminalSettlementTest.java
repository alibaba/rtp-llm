package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
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
                WorkerTerminalSource.DECODE_ENDPOINT, true, 0L);

        assertTrue(RequestSlot.tryReconcileDecodeTerminal(
                terminal, decode, 4L, RESERVATION));
        verifyNoInteractions(decode);
    }

    @Test
    void prefillBackedTerminalDelegatesToTheExactDecodeClaimTransaction() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DeferredTerminal terminal = DeferredTerminal.worker(
                WorkerTerminalSource.PREFILL_ENDPOINT, false, 9L);
        when(decode.reconcilePriorityVictimFinished(4L, RESERVATION))
                .thenReturn(false);

        assertFalse(RequestSlot.tryReconcileDecodeTerminal(
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
            RequestLifecycleTestSupport.startBatchDelivery(slot, 7L);
            PreemptionRegistration claim = slot.tryInstallPreemption(
                    RESERVATION.reservationToken(), 9L, "priority victim");
            assertNotNull(claim);
            assertEquals(RequestSlot.RequestEffect.Status.NONE,
                    slot.applyPreemptionPhase(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT).status());
            assertEquals(RequestSlot.RequestEffect.Status.NONE,
                    slot.applyPreemptionPhase(claim, outcome).status());
            RequestLifecycleTestSupport.recordCancellation(slot, CancelReason.CLIENT_CANCELLED, "original client cancellation");
            assertTrue(slot.installInactivityDeadline(inactivity));

            TerminalAction action = slot.beginTerminalizing(
                    TerminalOutcome.cancel("original client cancellation; request inactive"), null);
            assertNotNull(action);
            assertSame(claim, action.preemption());
            assertTrue(claim.isFinished());
            assertEquals(CancelReason.CLIENT_CANCELLED, slot.requireCancellationFirstCause());
            assertEquals(RequestSlot.RequestEffect.Status.STALE,
                    RequestLifecycleTestSupport.acknowledge(slot, 7L).status());
            assertEquals(RequestSlot.RequestEffect.Status.STALE,
                    slot.applyPreemptionCompleted(claim, "late Cancel ACK").status());
            assertEquals(RequestSlot.RequestEffect.Status.STALE,
                    slot.reduceWorkerTerminal(fixture.item(), DeferredTerminal.worker(
                            WorkerTerminalSource.DECODE_ENDPOINT, true, 0L)).status());
            assertFalse(slot.consumeInactivityDeadline(inactivity));
            action.terminalResources().release(timer);
            action.terminalResources().release(timer);
            verify(timer).cancel(inactivity);

            TerminationResult settled = slot.finishTermination(action);
            assertEquals(RequestState.Phase.CANCELLED, settled.terminal().state());
            assertNull(settled.transitionFailure());
            assertTrue(slot.isTerminalRecord());
            assertNull(slot.activeItem());
            assertFalse(slot.hasCancellationFirstCause());
            assertNull(slot.beginTerminalizing(
                    TerminalOutcome.timeout("duplicate expiry"), null));
        }
    }

    @Test
    void completedDeliveryFutureCannotPreventRequestExpiry() {
        Fixture fixture = fixture();
        RequestSlot slot = fixture.slot();
        synchronized (slot) {
            RequestLifecycleTestSupport.startBatchDelivery(slot, 7L);
            RequestLifecycleTestSupport.markAcknowledged(slot);
            Response delivered = new Response();
            delivered.setSuccess(true);
            assertTrue(slot.future().completeOwned(delivered));

            TerminalAction action = slot.beginTerminalizing(
                    TerminalOutcome.timeout("request inactive"), new Response());
            assertNotNull(action);
            assertNull(action.publication());
            assertNull(action.response());
            assertEquals(RequestState.Phase.TIMED_OUT,
                    slot.finishTermination(action).terminal().state());
            assertSame(delivered, slot.future().join());
            assertTrue(slot.isTerminalRecord());
        }
    }

    @Test
    void workerTerminalDuringAdmissionPreservesTheEarlierCancellationCause() {
        Fixture fixture = fixture(false);
        RequestSlot slot = fixture.slot();
        DeferredTerminal workerTerminal = DeferredTerminal.worker(
                WorkerTerminalSource.DECODE_ENDPOINT, true, 0L);
        synchronized (slot) {
            assertTrue(RequestLifecycleTestSupport.recordCancellation(slot,
                    CancelReason.CLIENT_CANCELLED, "first client cancellation"));
            assertEquals(RequestSlot.RequestEffect.Status.NONE,
                    slot.reduceWorkerTerminal(fixture.item(), workerTerminal).status());

            RequestSlot.AdmissionHandleCompletion completion =
                    slot.completeAdmissionHandle(fixture.admission());
            assertTrue(completion.owned());
            assertNull(completion.cancellationReason(), "terminal proof needs no extra Cancel send");
            assertSame(workerTerminal, completion.pendingTerminal());
            assertEquals(CancelReason.CLIENT_CANCELLED, slot.requireCancellationFirstCause());
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, slot.snapshot().state());

            TerminalAction action = slot.beginTerminalizing(
                    TerminalOutcome.cancel("first client cancellation; worker completed"), null);
            assertNotNull(action);
            assertEquals(RequestState.Phase.CANCELLED,
                    slot.finishTermination(action).terminal().state());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void workerProofClaimsTerminalBeforeCleanupWithOrWithoutPreemption(boolean preempting) {
        Fixture fixture = fixture();
        RequestSlot slot = fixture.slot();
        TerminalAction action;
        Response acknowledged = new Response();
        synchronized (slot) {
            RequestLifecycleTestSupport.startBatchDelivery(slot, 7L);
            RequestLifecycleTestSupport.markAcknowledged(slot);
            acknowledged.setSuccess(true);
            slot.future().completeOwned(acknowledged);
            PreemptionRegistration claim = preempting
                    ? slot.tryInstallPreemption(RESERVATION.reservationToken(), 9L, "victim") : null;
            if (preempting) {
                assertNotNull(claim);
                slot.applyPreemptionPhase(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT);
            }

            RequestSlot.RequestEffect effect = slot.reduceWorkerTerminal(fixture.item(),
                    DeferredTerminal.worker(WorkerTerminalSource.DECODE_ENDPOINT, false, 42L));
            assertEquals(RequestSlot.RequestEffect.Status.READY, effect.status());
            assertNotNull(effect.terminal());
            assertSame(fixture.item(), effect.terminal().item());
            assertSame(claim, effect.signal());
            assertFalse(slot.isTerminalRecord(), "cleanup must precede terminal record commitment");
            assertEquals(RequestSlot.RequestEffect.Status.STALE, RequestLifecycleTestSupport.acknowledge(slot, 7L).status(),
                    "terminal ownership must exclude late ACK before cleanup runs");
            verifyNoInteractions(fixture.item().decodeEp());
            action = effect.terminal();
        }
        slot.releaseTerminalEndpoints(action);
        // Decode completion settles this request, not the Prefill-owned batch group.
        verify(fixture.item().prefillEp(), never()).releaseCommittedItem(any());
        verify(fixture.item().prefillEp(), never()).expireCommittedItem(any());
        synchronized (slot) {
            assertEquals(RequestState.Phase.FAILED, slot.finishTermination(action).terminal().state());
            assertSame(acknowledged, slot.future().join(), "worker termination cannot replace a published response");
        }
    }

    private static Fixture fixture() {
        return fixture(true);
    }

    private static Fixture fixture(boolean finishAdmission) {
        var config = SchedulingTestConfig.newConfig();
        BalanceContext context = RequestLifecycleTestSupport.context(config, RESERVATION.requestId());
        RequestSlot slot = new RequestSlot(mock(RequestCompletionPublisher.class), RESERVATION.requestId(), null, null, null, null);
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), null, null,
                mock(PrefillEndpoint.class), mock(DecodeEndpoint.class), RESERVATION,
                System.currentTimeMillis());
        AdmissionHandle admission;
        synchronized (slot) {
            slot.configureInactivityTimeout(60_000L);
            admission = slot.tryBeginAdmissionHandle();
            assertNotNull(admission);
            assertTrue(slot.tryBindItemForPublication(item));
            if (finishAdmission) {
                slot.completeAdmissionHandle(admission);
            }
        }
        return new Fixture(slot, item, admission);
    }

    private record Fixture(RequestSlot slot, ScheduledRequest item, AdmissionHandle admission) { }
}
