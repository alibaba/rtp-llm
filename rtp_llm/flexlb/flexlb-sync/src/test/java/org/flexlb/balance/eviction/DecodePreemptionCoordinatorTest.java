package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionClaim;
import org.flexlb.balance.preemption.PreemptionLifecyclePort;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodePreemptionCoordinatorTest {

    private static final long VICTIM_ID = 81L;
    private static final long VICTIM_TOKEN = 17L;
    private static final long INCOMING_ID = 91L;

    private EngineCancelChannel cancelChannel;
    private PreemptionLifecyclePort lifecycle;
    private DecodeEndpoint endpoint;
    private AdmissionMutation mutation;
    private BooleanSupplier placementSeal;
    private PreemptionClaim claim;
    private DecodeEndpoint.ReservationHandle incoming;
    private CompletableFuture<VictimTerminal> terminal;
    private DecodePreemptionCoordinator coordinator;

    @BeforeEach
    void setUp() {
        cancelChannel = mock(EngineCancelChannel.class);
        lifecycle = mock(PreemptionLifecyclePort.class);
        endpoint = mock(DecodeEndpoint.class);
        mutation = mock(AdmissionMutation.class);
        placementSeal = mock(BooleanSupplier.class);
        claim = mock(PreemptionClaim.class);
        coordinator = new DecodePreemptionCoordinator(
                cancelChannel, lifecycle);

        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getGenerationId()).thenReturn(7L);
        when(endpoint.getStatus()).thenReturn(status);
        incoming = new DecodeEndpoint.ReservationHandle(
                7L, INCOMING_ID, 23L);
        when(lifecycle.resolveCancelTarget(VICTIM_ID, VICTIM_TOKEN))
                .thenReturn(new CancelTarget("prefill-a", 8081));
        when(lifecycle.claimForPreemption(
                eq(VICTIM_ID), eq(VICTIM_TOKEN), eq(1L), any()))
                .thenReturn(claim);
        when(claim.requestId()).thenReturn(VICTIM_ID);
        when(claim.attemptToken()).thenReturn(1L);
        terminal = new CompletableFuture<>();
        when(claim.terminal()).thenReturn(terminal);
        when(endpoint.beginPriorityPreemption(
                eq(1L), any(), eq(INCOMING_ID), anyLong(), anyLong(),
                eq(80), any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(new DecodeEndpoint.PreemptionBegin(
                        DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                        incoming));
        when(placementSeal.getAsBoolean()).thenReturn(true);
    }

    @Test
    void sealFailureAbortsEveryReversibleOwnerBeforeCancelPnr() {
        when(mutation.seal()).thenReturn(false);

        DecodePreemptionCoordinator.ExecutionResult result =
                coordinator.execute(request()).join();

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONFLICT,
                result.code());
        assertNull(result.incoming());
        InOrder order = inOrder(lifecycle, endpoint, mutation);
        order.verify(lifecycle).claimForPreemption(
                eq(VICTIM_ID), eq(VICTIM_TOKEN), eq(1L), any());
        order.verify(endpoint).beginPriorityPreemption(
                eq(1L), any(), eq(INCOMING_ID), anyLong(), anyLong(),
                eq(80), any(DecodeEndpoint.AdmissionCapacity.class));
        order.verify(mutation).seal();
        order.verify(endpoint).abortPriorityPreemption(1L);
        order.verify(lifecycle).releasePreemptionClaim(claim);
        verify(endpoint, never()).markPriorityCancelInFlight(anyLong());
        verify(lifecycle, never()).markPreemptionCancelInFlight(any());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void committedResultCarriesBeginHandleAndSealsImmediatelyBeforeCancel() {
        when(mutation.seal()).thenReturn(true);
        when(endpoint.markPriorityCancelInFlight(1L)).thenReturn(true);
        when(lifecycle.markPreemptionCancelInFlight(claim)).thenReturn(true);
        when(endpoint.markPriorityCancelAccepted(1L, VICTIM_ID))
                .thenReturn(true);
        when(lifecycle.markPreemptionCancelAccepted(claim)).thenReturn(true);
        when(cancelChannel.cancel(any(), eq(VICTIM_ID), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.accepted()));
        when(endpoint.commitPriorityPreemption(1L)).thenReturn(true);

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> execution =
                coordinator.execute(request());
        InOrder pnrOrder = inOrder(
                mutation, placementSeal, endpoint, lifecycle, cancelChannel);
        pnrOrder.verify(mutation).seal();
        pnrOrder.verify(placementSeal).getAsBoolean();
        pnrOrder.verify(endpoint).markPriorityCancelInFlight(1L);
        pnrOrder.verify(lifecycle).markPreemptionCancelInFlight(claim);
        pnrOrder.verify(cancelChannel)
                .cancel(any(), eq(VICTIM_ID), anyLong());

        terminal.complete(new VictimTerminal(VICTIM_ID));
        DecodePreemptionCoordinator.ExecutionResult result = execution.join();

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED,
                result.code());
        assertSame(incoming, result.incoming());
        verify(endpoint).commitPriorityPreemption(1L);
    }

    @Test
    void revokedPlacementAbortsBeforeFirstCancelSideEffect() {
        when(mutation.seal()).thenReturn(true);
        when(placementSeal.getAsBoolean()).thenReturn(false);

        DecodePreemptionCoordinator.ExecutionResult result =
                coordinator.execute(request()).join();

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONFLICT,
                result.code());
        InOrder order = inOrder(mutation, placementSeal, endpoint, lifecycle);
        order.verify(mutation).seal();
        order.verify(placementSeal).getAsBoolean();
        order.verify(endpoint).abortPriorityPreemption(1L);
        order.verify(lifecycle).releasePreemptionClaim(claim);
        verify(endpoint, never()).markPriorityCancelInFlight(anyLong());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void coordinatorBoundsAChannelThatNeverCompletesItsAck()
            throws Exception {
        when(mutation.seal()).thenReturn(true);
        when(endpoint.markPriorityCancelInFlight(1L)).thenReturn(true);
        when(lifecycle.markPreemptionCancelInFlight(claim)).thenReturn(true);
        when(cancelChannel.cancel(any(), eq(VICTIM_ID), anyLong()))
                .thenReturn(new CompletableFuture<>());

        DecodePreemptionCoordinator.ExecutionResult result =
                coordinator.execute(request(5L, 5L))
                        .get(5, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                result.code());
        assertNull(result.incoming());
        verify(cancelChannel).cancel(any(), eq(VICTIM_ID), eq(5L));
    }

    private DecodePreemptionCoordinator.Request request() {
        return request(1_000L, 1_000L);
    }

    private DecodePreemptionCoordinator.Request request(
            long ackTimeoutMs,
            long completionTimeoutMs) {
        DecodeRequestSnapshot victim = new DecodeRequestSnapshot(
                VICTIM_ID,
                30,
                DecodeTaskPhase.RUNNING,
                256L,
                256L,
                true,
                false,
                VICTIM_TOKEN);
        return new DecodePreemptionCoordinator.Request(
                endpoint,
                INCOMING_ID,
                128L,
                129L,
                80,
                new DecodeEndpoint.AdmissionCapacity(1L, 100L),
                List.of(victim),
                ackTimeoutMs,
                completionTimeoutMs,
                mutation,
                placementSeal,
                "preempted for test");
    }
}
