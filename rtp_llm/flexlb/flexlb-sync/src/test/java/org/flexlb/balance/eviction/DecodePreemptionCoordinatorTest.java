package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodePreemptionCoordinatorTest {

    @Test
    void commitsOnlyAfterEveryExactVictimIsTerminal() throws Exception {
        RequestRegistry requests = mock(RequestRegistry.class);
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getGenerationId()).thenReturn(9L);
        when(endpoint.beginPriorityPreemption(
                anyLong(), anyList(), anyLong(), anyLong(), anyLong(),
                anyInt(), any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(DecodeEndpoint.PreemptionBeginResult.SUCCESS);
        when(endpoint.markPriorityCancelInFlight(anyLong())).thenReturn(true);
        when(endpoint.recordPriorityCancelPhase(anyLong(), anyLong(), any()))
                .thenReturn(true);
        when(endpoint.commitPriorityPreemption(anyLong())).thenReturn(true);
        when(requests.findCancelTarget(anyLong(), anyLong())).thenReturn(
                Optional.of(new CancelTarget("10.0.0.1", 9090)));
        when(requests.tryApplyPreemptionPhase(any(), any())).thenReturn(true);

        CompletableFuture<VictimTerminal> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimTerminal> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(11L, firstTerminal);
        PreemptionRegistration second = claim(12L, secondTerminal);
        when(requests.tryClaim(anyLong(), anyLong(), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<Long>getArgument(0) == 11L ? first : second));

        EngineCancelChannel cancelChannel = mock(EngineCancelChannel.class);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelAck.ACCEPTED));
        DecodePreemptionCoordinator coordinator =
                new DecodePreemptionCoordinator(cancelChannel, requests);
        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult> result =
                coordinator.preempt(new DecodePreemptionCoordinator.PreemptionCommand(
                        endpoint, 20L, 64L, 64L, 70,
                        new DecodeEndpoint.AdmissionCapacity(2L, 100L),
                        List.of(victim(11L, 101L), victim(12L, 102L)),
                        1_000L, 1_000L, () -> true, "test"));

        assertFalse(result.isDone());
        firstTerminal.complete(new VictimTerminal(11L));
        assertFalse(result.isDone(), "one terminal cannot release two victims");
        secondTerminal.complete(new VictimTerminal(12L));

        assertTrue(result.get(1, TimeUnit.SECONDS).committed());
        verify(endpoint).commitPriorityPreemption(1L);
        verify(endpoint, never()).abortPriorityPreemption(anyLong());
    }

    private static PreemptionRegistration claim(
            long requestId,
            CompletableFuture<VictimTerminal> terminal) {
        PreemptionRegistration claim = mock(PreemptionRegistration.class);
        when(claim.requestId()).thenReturn(requestId);
        when(claim.attemptToken()).thenReturn(1L);
        when(claim.terminalObservation()).thenReturn(terminal);
        return claim;
    }

    private static DecodeRequestView victim(long requestId, long reservationToken) {
        return new DecodeRequestView(
                requestId, 30, 64L, 64L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true, reservationToken, false, false);
    }
}
