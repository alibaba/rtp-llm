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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodePreemptionCoordinatorTest {

    @Test
    void commitsOnlyAfterEveryExactVictimIsTerminal() throws Exception {
        Fixture fixture = fixture();
        RequestRegistry requests = fixture.requests();
        DecodeEndpoint endpoint = fixture.endpoint();

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
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq(11L), anyLong());
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq(12L), anyLong());
        verify(endpoint).commitPriorityPreemption(1L);
        verify(endpoint, never()).abortPriorityPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class, names = {"ACCEPTED", "FAILED"})
    void timeoutBeginsAfterAckAndLateTerminalCannotReopenIncoming(
            EngineCancelChannel.CancelAck acknowledgement) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimTerminal> terminal = new CompletableFuture<>();
        PreemptionRegistration victimClaim = claim(11L, terminal);
        when(fixture.requests().tryClaim(anyLong(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        CompletableFuture<EngineCancelChannel.CancelAck> ack = new CompletableFuture<>();
        when(channel.cancel(any(), anyLong(), anyLong())).thenReturn(ack);
        DecodePreemptionCoordinator coordinator =
                new DecodePreemptionCoordinator(channel, fixture.requests());
        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult> outcome =
                coordinator.preempt(new DecodePreemptionCoordinator.PreemptionCommand(
                        fixture.endpoint(), 20L, 64L, 64L, 70,
                        new DecodeEndpoint.AdmissionCapacity(1L, 100L),
                        List.of(victim(11L, 101L)),
                        50L, 20L, () -> true, "test"));

        // Transport owns the ACK deadline. The terminal-wait budget starts
        // only when that phase resolves, even if the transport outcome is unknown.
        assertThrows(TimeoutException.class, () -> outcome.get(60L, TimeUnit.MILLISECONDS));
        ack.complete(acknowledgement);
        var timedOut = outcome.get(1L, TimeUnit.SECONDS);
        assertFalse(timedOut.committed());
        assertTrue(timedOut.controlFailure());
        assertEquals("cancel_terminal_unknown", timedOut.detail());
        verify(fixture.endpoint()).abortPriorityPreemption(1L);
        verify(fixture.requests(), never()).tryReleasePreemption(victimClaim);
        assertFalse(terminal.isDone(), "timing out admission must retain the victim terminal observation");

        terminal.complete(new VictimTerminal(11L));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPriorityPreemption(anyLong());
    }

    private static Fixture fixture() {
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
        return new Fixture(requests, endpoint);
    }

    private record Fixture(RequestRegistry requests, DecodeEndpoint endpoint) { }

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
