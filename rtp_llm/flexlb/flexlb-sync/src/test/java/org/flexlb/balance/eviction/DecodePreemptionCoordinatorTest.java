package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
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
        when(requests.tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<String>getArgument(0).equals("11") ? first : second));

        EngineCancelChannel cancelChannel = mock(EngineCancelChannel.class);
        when(cancelChannel.cancel(any(), anyString(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelAck.ACCEPTED));
        DecodePreemptionCoordinator coordinator =
                new DecodePreemptionCoordinator(cancelChannel, requests, fixture.reporter());
        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult> result =
                coordinator.preempt(new DecodePreemptionCoordinator.PreemptionCommand(
                        endpoint, "20", 64L, 64L, 70,
                        new DecodeEndpoint.AdmissionCapacity(2L, 100L),
                        List.of(victim(11L, 101L), victim(12L, 102L)),
                        1_000L, 1_000L, () -> true, "test"));

        assertFalse(result.isDone());
        firstTerminal.complete(new VictimTerminal("11"));
        assertFalse(result.isDone(), "one terminal cannot release two victims");
        secondTerminal.complete(new VictimTerminal("12"));

        assertTrue(result.get(1, TimeUnit.SECONDS).committed());
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq("11"), anyLong());
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq("12"), anyLong());
        verify(endpoint).finishPreemption(1L, DecodeEndpoint.PreemptionDecision.COMMIT);
        verify(endpoint, never()).finishPreemption(anyLong(), eq(DecodeEndpoint.PreemptionDecision.ABORT));
        verifyNoInteractions(fixture.reporter());
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class, names = {"ACCEPTED", "FAILED"})
    void timeoutBeginsAfterAckAndLateTerminalCannotReopenIncoming(
            EngineCancelChannel.CancelAck acknowledgement) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimTerminal> terminal = new CompletableFuture<>();
        PreemptionRegistration victimClaim = claim(11L, terminal);
        when(fixture.requests().tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        CompletableFuture<EngineCancelChannel.CancelAck> ack = new CompletableFuture<>();
        when(channel.cancel(any(), anyString(), anyLong())).thenReturn(ack);
        DecodePreemptionCoordinator coordinator =
                new DecodePreemptionCoordinator(channel, fixture.requests(), fixture.reporter());
        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult> outcome =
                coordinator.preempt(new DecodePreemptionCoordinator.PreemptionCommand(
                        fixture.endpoint(), "20", 64L, 64L, 70,
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
        verify(fixture.endpoint()).finishPreemption(1L, DecodeEndpoint.PreemptionDecision.ABORT);
        verify(victimClaim, never()).release();
        assertFalse(terminal.isDone(), "timing out admission must retain the victim terminal observation");

        terminal.complete(new VictimTerminal("11"));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).finishPreemption(anyLong(), eq(DecodeEndpoint.PreemptionDecision.COMMIT));
        verifyNoInteractions(fixture.reporter());
    }

    @ParameterizedTest
    @MethodSource("beginResults")
    void reportsOnlyTargetValidationFailuresOncePerPlan(
            boolean returnInstructions, DecodeEndpoint.PreemptionBeginResult beginResult) {
        Fixture fixture = fixture();
        setBeginResult(fixture.endpoint(), beginResult);
        CompletableFuture<VictimTerminal> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimTerminal> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(11L, firstTerminal);
        PreemptionRegistration second = claim(12L, secondTerminal);
        when(fixture.requests().tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(first), Optional.of(second));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyString(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
        DecodePreemptionCoordinator coordinator = new DecodePreemptionCoordinator(
                channel, fixture.requests(), fixture.reporter());
        var command = command(fixture.endpoint(), List.of(victim(11L, 101L), victim(12L, 102L)));

        var resultFuture = returnInstructions
                ? coordinator.prepareReturnedPreemption(command) : coordinator.preempt(command);
        firstTerminal.complete(new VictimTerminal("11"));
        secondTerminal.complete(new VictimTerminal("12"));
        var result = resultFuture.join();

        assertEquals(beginResult == DecodeEndpoint.PreemptionBeginResult.SUCCESS, result.committed());
        assertEquals(beginResult == DecodeEndpoint.PreemptionBeginResult.ENDPOINT_RETIRED, result.controlFailure());
        if (!result.committed()) {
            assertEquals((returnInstructions ? "return_" : "begin_") + beginResult.name().toLowerCase(),
                    result.detail());
            verifyNoInteractions(channel);
            if (!returnInstructions) {
                verify(first).release();
                verify(second).release();
            }
        }
        String reason = switch (beginResult) {
            case VICTIM_GONE -> "victim_state_changed";
            case VICTIM_ALREADY_CLAIMED -> "victim_already_claimed";
            case INVALID_PRIORITY -> "priority_not_preemptible";
            default -> null;
        };
        if (reason == null) {
            verifyNoInteractions(fixture.reporter());
        } else {
            verify(fixture.reporter()).reportPreemptionTargetInvalid(returnInstructions ? "return" : "rpc", reason);
            verifyNoMoreInteractions(fixture.reporter());
        }
    }

    private static Stream<Arguments> beginResults() {
        return Stream.of(DecodeEndpoint.PreemptionBeginResult.values())
                .flatMap(result -> Stream.of(Arguments.of(true, result), Arguments.of(false, result)));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void reportsUnavailableCancelTargetWithoutStartingPreemption(boolean reporterFails) {
        Fixture fixture = fixture();
        when(fixture.requests().findCancelTarget("12", 102L)).thenReturn(Optional.empty());
        if (reporterFails) {
            doThrow(new IllegalStateException("metrics unavailable")).when(fixture.reporter())
                    .reportPreemptionTargetInvalid("rpc", "cancel_target_unavailable");
        }
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        DecodePreemptionCoordinator coordinator = new DecodePreemptionCoordinator(
                channel, fixture.requests(), fixture.reporter());

        var result = coordinator.preempt(command(
                fixture.endpoint(), List.of(victim(11L, 101L), victim(12L, 102L)))).join();

        assertFalse(result.committed());
        assertTrue(result.controlFailure());
        assertEquals("cancel_owner_missing:12", result.detail());
        verify(fixture.reporter()).reportPreemptionTargetInvalid("rpc", "cancel_target_unavailable");
        verifyNoMoreInteractions(fixture.reporter());
        verify(fixture.requests(), never()).tryClaim(anyString(), anyLong(), anyLong(), any());
        verifyNoInteractions(channel);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void reportsRejectedRequestClaimAndReleasesEarlierClaim(boolean reporterFails) {
        Fixture fixture = fixture();
        PreemptionRegistration first = claim(11L, new CompletableFuture<>());
        when(fixture.requests().tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(first), Optional.empty());
        if (reporterFails) {
            doThrow(new IllegalStateException("metrics unavailable")).when(fixture.reporter())
                    .reportPreemptionTargetInvalid("rpc", "request_claim_rejected");
        }
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        DecodePreemptionCoordinator coordinator = new DecodePreemptionCoordinator(
                channel, fixture.requests(), fixture.reporter());

        var result = coordinator.preempt(command(
                fixture.endpoint(), List.of(victim(11L, 101L), victim(12L, 102L)))).join();

        assertFalse(result.committed());
        assertFalse(result.controlFailure());
        assertEquals("victim_inflight_gone", result.detail());
        verify(fixture.reporter()).reportPreemptionTargetInvalid("rpc", "request_claim_rejected");
        verifyNoMoreInteractions(fixture.reporter());
        verify(first).release();
        verifyNoInteractions(channel);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void targetMetricFailurePreservesBeginResultAndReleasesRpcClaim(boolean returnInstructions) {
        Fixture fixture = fixture();
        setBeginResult(fixture.endpoint(), DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE);
        PreemptionRegistration victimClaim = claim(11L, new CompletableFuture<>());
        when(fixture.requests().tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        doThrow(new IllegalStateException("metrics unavailable")).when(fixture.reporter())
                .reportPreemptionTargetInvalid(returnInstructions ? "return" : "rpc", "victim_state_changed");
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        DecodePreemptionCoordinator coordinator = new DecodePreemptionCoordinator(
                channel, fixture.requests(), fixture.reporter());
        var command = command(fixture.endpoint(), List.of(victim(11L, 101L)));

        var result = (returnInstructions ? coordinator.prepareReturnedPreemption(command)
                : coordinator.preempt(command)).join();

        assertFalse(result.committed());
        assertFalse(result.controlFailure());
        assertEquals(returnInstructions ? "return_victim_gone" : "begin_victim_gone", result.detail());
        if (!returnInstructions) {
            verify(victimClaim).release();
        }
        verifyNoInteractions(channel);
    }

    @Test
    void engineNotFoundReplyIsNotAMasterTargetValidationFailure() {
        Fixture fixture = fixture();
        PreemptionRegistration victimClaim = claim(11L, new CompletableFuture<>());
        when(fixture.requests().tryClaim(anyString(), anyLong(), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyString(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.NOT_FOUND));
        DecodePreemptionCoordinator coordinator = new DecodePreemptionCoordinator(
                channel, fixture.requests(), fixture.reporter());

        var result = coordinator.preempt(command(fixture.endpoint(), List.of(victim(11L, 101L)))).join();

        assertFalse(result.committed());
        assertFalse(result.controlFailure());
        assertEquals("cancel_not_found", result.detail());
        verify(fixture.endpoint()).finishPreemption(1L, DecodeEndpoint.PreemptionDecision.ABORT);
        verifyNoInteractions(fixture.reporter());
    }

    private static DecodePreemptionCoordinator.PreemptionCommand command(
            DecodeEndpoint endpoint, List<DecodeRequestView> victims) {
        return new DecodePreemptionCoordinator.PreemptionCommand(
                endpoint, "20", 64L, 64L, 70,
                new DecodeEndpoint.AdmissionCapacity(2L, 100L), victims,
                1_000L, 1_000L, () -> true, "test");
    }

    private static void setBeginResult(DecodeEndpoint endpoint, DecodeEndpoint.PreemptionBeginResult result) {
        when(endpoint.beginPreemption(anyLong(), anyList(), anyString(), anyLong(), anyLong(), anyInt(),
                any(DecodeEndpoint.AdmissionCapacity.class))).thenReturn(result);
        when(endpoint.beginReturnedPreemption(anyLong(), anyList(), anyString(), anyLong(), anyLong(), anyInt(),
                any(DecodeEndpoint.AdmissionCapacity.class))).thenReturn(result);
    }

    private static Fixture fixture() {
        RequestRegistry requests = mock(RequestRegistry.class);
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getGenerationId()).thenReturn(9L);
        when(endpoint.beginPreemption(
                anyLong(),
                anyList(),
                anyString(),
                anyLong(),
                anyLong(),
                anyInt(),
                any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(DecodeEndpoint.PreemptionBeginResult.SUCCESS);
        when(endpoint.updatePreemption(anyLong(), any(DecodeEndpoint.PreemptionUpdate.class))).thenReturn(true);
        when(endpoint.finishPreemption(anyLong(), eq(DecodeEndpoint.PreemptionDecision.COMMIT))).thenReturn(true);
        when(requests.findCancelTarget(anyString(), anyLong())).thenReturn(
                Optional.of(new CancelTarget("10.0.0.1", 9090)));
        return new Fixture(requests, endpoint, mock(RequestSchedulerReporter.class));
    }

    private record Fixture(RequestRegistry requests, DecodeEndpoint endpoint, RequestSchedulerReporter reporter) { }

    private static PreemptionRegistration claim(
            long requestId,
            CompletableFuture<VictimTerminal> terminal) {
        PreemptionRegistration claim = mock(PreemptionRegistration.class);
        when(claim.requestId()).thenReturn(Long.toString(requestId));
        when(claim.applyPhase(any())).thenReturn(true);
        when(claim.attemptToken()).thenReturn(1L);
        when(claim.terminalObservation()).thenReturn(terminal);
        return claim;
    }

    private static DecodeRequestView victim(long requestId, long reservationToken) {
        return new DecodeRequestView(
                Long.toString(requestId), 30, 64L, 64L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true, reservationToken, false, false);
    }
}
