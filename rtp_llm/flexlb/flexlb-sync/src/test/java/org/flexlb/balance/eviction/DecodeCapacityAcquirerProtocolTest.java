package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.SchedulerRuntime;
import org.flexlb.balance.scheduler.SchedulerTestSupport;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.flexlb.engine.grpc.EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodeCapacityAcquirerProtocolTest {

    private SchedulerRuntime runtime;

    @BeforeEach
    void createRuntime() {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        runtime = SchedulerTestSupport.runtime(SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class)));
    }

    @AfterEach
    void closeRuntime() {
        try { org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "shutdown"); }
        finally { assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime).isTerminated(), "the runtime owns every preemption deadline thread"); }
    }

    private static org.flexlb.dao.loadbalance.Request incoming() {
        return org.flexlb.dao.loadbalance.Request.builder().requestId(20L).seqLen(64L).maxNewTokens(0).build();
    }

    @Test
    void commitsOnlyAfterEveryExactVictimIsTerminal() throws Exception {
        Fixture fixture = fixture();
        AbstractRequestScheduler requests = fixture.requests();
        DecodeEndpoint endpoint = fixture.endpoint();

        CompletableFuture<VictimResolution> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimResolution> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(requests.tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));

        EngineGrpcClient engineClient = mock(EngineGrpcClient.class);
        when(engineClient.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineGrpcClient.CancelReply.ACCEPTED));
        DecodeCapacityAcquirer acquirer =
                new DecodeCapacityAcquirer(dispatcher(engineClient), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(2L, 100L));
        CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> result =
                preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        endpoint, incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                        List.of(victim(11L, 101L), victim(12L, 102L)),
                        1_000L, 1_000L, () -> true, "test"));

        assertFalse(result.isDone());
        firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        assertFalse(result.isDone(), "one terminal cannot release two victims");
        secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));

        assertTrue(result.get(1, TimeUnit.SECONDS).committed());
        assertEquals(new DecodeResources.ReservationHandle(9L, 20L, 100L), result.join().reservation());
        verify(engineClient).cancelAsync(eq("10.0.0.1"), eq(9090), eq(CancelReason.PRIORITY_PREEMPTED.toRequest(11L)), anyLong());
        verify(engineClient).cancelAsync(eq("10.0.0.1"), eq(9090), eq(CancelReason.PRIORITY_PREEMPTED.toRequest(12L)), anyLong());
        verify(endpoint).commitPreemption(1L);
        verify(endpoint, never()).abortPreemption(anyLong());
    }

    @ParameterizedTest
    @ValueSource(strings = {"missing", "foreign", "failed"})
    void completedFutureRequiresAValidExactRequestResolution(String completionKind) throws Exception {
        Fixture fixture = fixture();
        var resolution = new CompletableFuture<VictimResolution>();
        switch (completionKind) {
            case "missing" -> resolution.complete(null);
            case "foreign" -> resolution.complete(new VictimResolution(99L, VictimResolution.Outcome.REQUEST_END));
            case "failed" -> resolution.completeExceptionally(new IllegalStateException("resolution failed"));
            default -> throw new AssertionError(completionKind);
        }
        PreemptionRegistration victimClaim = claim(fixture.requests(), 11L, resolution);
        var signalInput = mock(org.flexlb.balance.scheduler.RequestSignals.class);
        when(signalInput.preemption(victimClaim)).thenReturn(resolution);
        when(signalInput.resolved(victimClaim)).thenAnswer(call -> resolution.handle((result, failure) -> result).getNow(null));
        org.mockito.Mockito.doReturn(signalInput).when(runtime).signals();
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                50L, 1_000L, () -> true, "exact resolution")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(fixture.requests(), never()).onPreemption(victimClaim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
    }

    @Test
    void resumedDeliveryRequiresIndependentDecodeReleaseProof() throws Exception {
        Fixture fixture = fixture();
        var resolution = CompletableFuture.completedFuture(
                new VictimResolution(11L, VictimResolution.Outcome.DELIVERY_RESUMED));
        var victimClaim = claim(fixture.requests(), 11L, resolution);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        when(fixture.endpoint().commitPreemption(anyLong())).thenReturn(null);
        var channel = mock(EngineGrpcClient.class);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                50L, 1_000L, () -> true, "delivery resumed")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed(), "request resolution cannot manufacture Decode release proof");
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(fixture.requests(), never()).onPreemption(victimClaim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class, names = {"ACCEPTED", "FAILED", "REQUEST_FENCED"})
    void timeoutBeginsAfterAckAndLateTerminalCannotReopenIncoming(
            EngineGrpcClient.CancelReply acknowledgement) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> terminal = new CompletableFuture<>();
        PreemptionRegistration victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        CompletableFuture<EngineGrpcClient.CancelReply> ack = new CompletableFuture<>();
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong())).thenReturn(ack);
        DecodeCapacityAcquirer acquirer =
                new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));
        CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> outcome =
                preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
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
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(fixture.requests(), never()).onPreemption(victimClaim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        assertFalse(terminal.isDone(), "timing out admission must retain the victim terminal observation");

        terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void partialTerminalTimeoutRetainsUnknownVictimAndNeverReopensIncoming(
            boolean terminalBeforeSend) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimResolution> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong())).thenAnswer(invocation ->
                CompletableFuture.completedFuture(invocation.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                        ? EngineGrpcClient.CancelReply.ACCEPTED : EngineGrpcClient.CancelReply.FAILED));
        DecodeCapacityAcquirer acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(2L, 100L));
        if (terminalBeforeSend) {
            firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        }
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                List.of(victim(11L, 101L), victim(12L, 102L)),
                50L, 1_000L, () -> true, "test"));
        if (!terminalBeforeSend) {
            verify(channel, org.mockito.Mockito.timeout(2000)).cancelAsync(any(), anyInt(), eq(CancelReason.PRIORITY_PREEMPTED.toRequest(11L)), anyLong());
            firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            assertFalse(outcome.isDone(), "one of two terminal observations cannot finish the aggregate");
        } else {
            verify(channel, never()).cancelAsync(any(), anyInt(), eq(CancelReason.PRIORITY_PREEMPTED.toRequest(11L)), anyLong());
        }
        var timedOut = outcome.get(3L, TimeUnit.SECONDS);
        assertFalse(timedOut.committed());
        assertTrue(timedOut.controlFailure());
        verify(channel).cancelAsync(any(), anyInt(), eq(CancelReason.PRIORITY_PREEMPTED.toRequest(12L)), anyLong());
        verify(fixture.requests(), never()).onPreemption(first, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        verify(fixture.requests(), never()).onPreemption(second, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        verify(fixture.requests()).onPreemption(eq(second), eq(phaseEvent(org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_UNKNOWN)));
        assertFalse(secondTerminal.isDone());
        verify(fixture.endpoint()).abortPreemption(1L);
        secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class,
            names = {"FAILED", "NOT_FOUND", "REQUEST_FENCED", "REQUEST_CLEANED"})
    void reversedAcknowledgementsStayBoundToTheirVictims(
            EngineGrpcClient.CancelReply secondReply) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimResolution> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));
        when(fixture.requests().onPreemption(eq(second), argThat(event -> event.endpoint() == fixture.endpoint() && event.reservation() != null)))
                .thenAnswer(invocation -> {
                    secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
                    return true;
                });
        CompletableFuture<EngineGrpcClient.CancelReply> firstAck = new CompletableFuture<>();
        CompletableFuture<EngineGrpcClient.CancelReply> secondAck = new CompletableFuture<>();
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong())).thenAnswer(invocation ->
                invocation.<CancelRequestPB>getArgument(2).getRequestId() == 11L ? firstAck : secondAck);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(2L, 100L));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                List.of(victim(11L, 101L), victim(12L, 102L)),
                1_000L, 1_000L, () -> true, "test"));

        secondAck.complete(secondReply);
        assertFalse(result.isDone());
        verify(fixture.requests(), never()).onPreemption(first, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        verify(fixture.requests(), never()).onPreemption(eq(second), argThat(event -> event.endpoint() == fixture.endpoint() && event.reservation() != null));
        firstAck.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        verify(fixture.requests(), org.mockito.Mockito.timeout(2000)).onPreemption(first, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        switch (secondReply) {
            case FAILED -> verify(fixture.requests()).onPreemption(second, phaseEvent(PreemptionCancelPhase.CANCEL_UNKNOWN));
            case NOT_FOUND -> verify(fixture.requests()).onPreemption(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
            case REQUEST_FENCED -> verify(fixture.requests()).onPreemption(second, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
            case REQUEST_CLEANED -> {
                verify(fixture.requests()).onPreemption(second, new AbstractRequestScheduler.PreemptionEvent(null, false, fixture.endpoint(),
                        new DecodeResources.ReservationHandle(9L, 12L, 102L), "test"));
            }
            default -> throw new AssertionError("unexpected test reply");
        }
        assertFalse(result.isDone(), "ACKs cannot replace the first victim's terminal proof");
        if (secondReply != EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
            secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
        }
        firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        assertTrue(result.get(1L, TimeUnit.SECONDS).committed());
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint(), never()).abortPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class,
            names = {"REQUEST_FENCED", "REQUEST_CLEANED"})
    void onlyDownstreamCleanupProofCanReplaceDecodeTerminal(EngineGrpcClient.CancelReply reply) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> terminal = new CompletableFuture<>();
        PreemptionRegistration claim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(claim));
        when(fixture.requests().onPreemption(eq(claim), argThat(event -> event.endpoint() == fixture.endpoint() && event.reservation() != null)))
                .thenAnswer(invocation -> {
                    terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
                    return true;
                });
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(reply));
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                1_000L, 1_000L, () -> true, "test"));
        if (reply == EngineGrpcClient.CancelReply.REQUEST_FENCED) {
            assertFalse(outcome.isDone(), "Prefill fence alone does not prove Decode resource release");
            verify(fixture.requests(), never()).onPreemption(eq(claim), argThat(event -> event.endpoint() == fixture.endpoint() && event.reservation() != null));
            verify(fixture.endpoint(), never()).updatePreemption(1L,
                    DecodeResources.PreemptionUpdate.fenced(new DecodeResources.ReservationHandle(9L, 11L, 101L)));
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
            terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        }
        assertTrue(outcome.get(1L, TimeUnit.SECONDS).committed());
        verify(fixture.endpoint()).commitPreemption(1L);
    }

    @Test
    void laterVictimClaimFailureReleasesEarlierClaimBeforeAnyCancel() throws Exception {
        Fixture fixture = fixture();
        PreemptionRegistration first = claim(fixture.requests(), 11L, new CompletableFuture<>());
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(first), Optional.empty());
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(2L, 100L));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L), victim(12L, 102L)),
                1000L, 1000L, () -> true, "test")).get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        verify(fixture.requests()).onPreemption(first, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        verify(fixture.endpoint(), never()).beginPreemption(anyLong(), anyList(), anyLong(),
                anyLong(), anyLong(), anyInt(), any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    void invalidCancelTargetReleasesEarlierClaimAndReportsControlFailureWithoutRpc() throws Exception {
        Fixture fixture = fixture();
        PreemptionRegistration first = claim(fixture.requests(), 11L, new CompletableFuture<>());
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(first))
                .thenThrow(new IllegalStateException("Priority victim has no routable Cancel target"));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(2L, 100L));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L), victim(12L, 102L)),
                1000L, 1000L, () -> true, "test")).get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertTrue(result.controlFailure());
        verify(fixture.requests()).onPreemption(first, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
        verify(fixture.endpoint(), never()).beginPreemption(anyLong(), anyList(), anyLong(),
                anyLong(), anyLong(), anyInt(), any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    void staleEndpointGenerationCannotCancelANewerRouteWithTheSameRequestAndToken() throws Exception {
        Fixture fixture = fixture();
        var stale = new DecodeResources.ReservationHandle(9L, 11L, 101L);
        when(fixture.requests().tryClaim(eq(stale), anyLong(), any())).thenReturn(Optional.empty());
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                1000L, 1000L, () -> true, "old endpoint")).get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertFalse(result.controlFailure());
        verify(fixture.requests()).tryClaim(eq(stale), anyLong(), eq("old endpoint"));
        verify(fixture.endpoint(), never()).beginPreemption(anyLong(), anyList(), anyLong(),
                anyLong(), anyLong(), anyInt(), any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void blockedPreemptionSettlementDoesNotDelayOtherCleanupDeadlines() throws Exception {
        var entered = new java.util.concurrent.CountDownLatch(2);
        var release = new java.util.concurrent.CountDownLatch(1);
        var deadlineRan = new java.util.concurrent.CountDownLatch(1);
        var settlementThreads = new java.util.concurrent.CopyOnWriteArrayList<String>();
        var outcomes = new java.util.ArrayList<CompletableFuture<DecodeCapacityAcquirer.PreemptionResult>>();
        try {
            for (int i = 0; i < 2; i++) {
                Fixture fixture = fixture();
                var terminal = new CompletableFuture<VictimResolution>();
                var victimClaim = claim(fixture.requests(), 11L, terminal);
                when(fixture.requests().tryClaim(any(), anyLong(), any()))
                        .thenReturn(Optional.of(victimClaim));
                org.mockito.Mockito.doAnswer(call -> {
                    settlementThreads.add(Thread.currentThread().getName());
                    entered.countDown();
                    if (!release.await(10, TimeUnit.SECONDS)) { throw new AssertionError("settlement not released"); }
                    return null;
                }).when(fixture.endpoint()).abortPreemption(anyLong());
                var channel = mock(EngineGrpcClient.class);
                when(channel.cancelAsync(any(), anyInt(),
                        argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                        .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
                var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));
                outcomes.add(preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                        50L, 20L, () -> true, "timer isolation")));
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS), "both settlements must be blocked before probing the timer");
            org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime).schedule(deadlineRan::countDown, 1, TimeUnit.MILLISECONDS);
            assertTrue(deadlineRan.await(1, TimeUnit.SECONDS), "blocked settlement must not occupy the shared deadline threads");
            assertTrue(settlementThreads.stream().allMatch(name -> name.startsWith("request-work-")),
                    "settlement threads: " + settlementThreads);
            assertTrue(outcomes.stream().noneMatch(CompletableFuture::isDone));
        } finally {
            release.countDown();
            for (var outcome : outcomes) {
                assertEquals("cancel_terminal_unknown", outcome.get(5, TimeUnit.SECONDS).detail());
            }
        }
    }

    @Test
    void timeoutBeforeScheduleReturnsCannotBeReopenedByLateResolution() throws Exception {
        Fixture fixture = fixture();
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var deadline = mock(org.flexlb.balance.scheduler.RequestTaskScheduler.Cancellation.class);
        var protocolRuntime = runtime;
        org.mockito.Mockito.doAnswer(call -> {
                    call.<Runnable>getArgument(0).run();
                    terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
                    return deadline;
                }).when(protocolRuntime).scheduleTimeout(any(Runnable.class), anyLong());
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()), protocolRuntime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                50L, 1L, () -> true, "schedule return race")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed(), "the deadline must freeze rejection before late request resolution");
        assertTrue(terminal.isDone(), "late request resolution remains available to its resource owner");
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(deadline).cancel();
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,true", "true,true", "true,false"})
    @org.junit.jupiter.api.Timeout(15)
    void settlementHasAnExplicitDependencyAndShutdownDrainsIt(
            boolean terminalWins, boolean admissionOpenAtSettlement) throws Exception {
        Fixture fixture = fixture();
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getScheduler()).thenReturn(fixture.requests());
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(),
                argThat(request -> request.getReason() == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var release = new java.util.concurrent.CountDownLatch(1);
        var settlementQueued = new java.util.concurrent.CountDownLatch(1);
        var afterSettlement = new CompletableFuture<Boolean>();
        var shutdown = new CompletableFuture<Void>();
        var admissionOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
        var submissions = new java.util.concurrent.atomic.AtomicInteger();
        Thread closer = null;
        try {
            var tasks = new org.flexlb.balance.scheduler.RequestTaskScheduler() {
                @Override public org.flexlb.balance.scheduler.RequestSignals signals() { return runtime.signals(); }
                @Override public Cancellation scheduleTimeout(Runnable work, long delayMs) {
                    return runtime.scheduleTimeout(work, delayMs);
                }
                @Override public void executeRequestWork(org.flexlb.balance.scheduler.RequestContext exact, Runnable work) {
                    boolean settlement = submissions.incrementAndGet() > 1;
                    runtime.executeRequestWork(exact, () -> {
                        // Let ACK coordination run; pause the subsequent resource-settlement continuation.
                        if (!settlement) { work.run(); return; }
                        settlementQueued.countDown();
                        try { assertTrue(release.await(10, TimeUnit.SECONDS)); }
                        catch (InterruptedException failure) { throw new AssertionError(failure); }
                        work.run();
                    });
                }
            };
            var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()), tasks, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()), new DecodeResources.AdmissionCapacity(1L, 100L));
            var outcome = acquirer.preempt(context, new DecodeCapacityAcquirer.PreemptionCommand(
                    fixture.endpoint(), incoming(), org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                    50L, terminalWins ? 5_000L : 20L, admissionOpen::get, "explicit settlement dependency"));
            verify(fixture.requests(), org.mockito.Mockito.timeout(2000)).onPreemption(victimClaim,
                    phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
            if (terminalWins) {
                terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            }
            assertTrue(settlementQueued.await(5, TimeUnit.SECONDS));
            assertFalse(outcome.isDone(), "settlement is paused before endpoint commit");
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
            verify(fixture.endpoint(), never()).abortPreemption(anyLong());
            if (!terminalWins) {
                assertFalse(terminal.isDone(), "timeout retains the exact late resolution observer");
                terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            }
            outcome.thenRun(() -> afterSettlement.complete(true));
            var independent = new CompletableFuture<Void>();
            runtime.executeRequestWork(context, () -> independent.complete(null));
            independent.get(5, TimeUnit.SECONDS);
            closer = new Thread(() -> {
                try {
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "shutdown");
                    shutdown.complete(null);
                } catch (Throwable failure) { shutdown.completeExceptionally(failure); }
            });
            closer.start();
            assertThrows(TimeoutException.class, () -> shutdown.get(100, TimeUnit.MILLISECONDS));
            admissionOpen.set(admissionOpenAtSettlement);
            release.countDown();
            assertEquals(terminalWins && admissionOpenAtSettlement, outcome.get(5, TimeUnit.SECONDS).committed());
            assertTrue(afterSettlement.get(5, TimeUnit.SECONDS), "dependent work follows the settlement future");
            shutdown.get(5, TimeUnit.SECONDS);
            assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime).isTerminated());
            if (!terminalWins || !admissionOpenAtSettlement) {
                verify(fixture.endpoint(), never()).commitPreemption(anyLong());
                verify(fixture.endpoint()).abortPreemption(1L);
            }
        } finally {
            release.countDown();
            if (closer != null) { closer.join(5_000); }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acknowledgementCoordinationRunsOnRequestWorkers(boolean rejected) throws Exception {
        if (rejected) {
            ((java.util.concurrent.ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(runtime, "requestWorkers")).shutdown();
        }
        var fixture = fixture();
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var threadName = new CompletableFuture<String>();
        org.mockito.Mockito.doAnswer(call -> {
            threadName.complete(Thread.currentThread().getName());
            return true;
        }).when(fixture.requests()).onPreemption(victimClaim, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()),
                runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                new DecodeResources.AdmissionCapacity(1L, 100L));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(fixture.endpoint(), incoming(),
                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                500L, 1000L, () -> true, "executor audit"));
        verify(channel, org.mockito.Mockito.timeout(2000)).cancelAsync(any(), anyInt(), any(), anyLong());
        ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        String observed = threadName.get(2, TimeUnit.SECONDS);
        assertTrue(observed.startsWith(rejected ? "request-work-recovery-" : "request-work-"),
                "coordinator must leave the stop worker, including executor rejection");
        terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        outcome.get(2, TimeUnit.SECONDS);
    }

    @Test void slowAckCoordinationDoesNotPreventAnUnrelatedCancel() throws Exception {
        var release = new java.util.concurrent.CountDownLatch(1);
        int size = ((java.util.concurrent.ThreadPoolExecutor) org.springframework.test.util.ReflectionTestUtils.getField(runtime, "stopWorkers")).getMaximumPoolSize();
        var entered = new java.util.concurrent.CountDownLatch(size);
        var outcomes = new java.util.ArrayList<CompletableFuture<?>>();
        var terminals = new java.util.ArrayList<CompletableFuture<VictimResolution>>();
        try {
            for (int i = 0; i < size; i++) {
                var fixture = fixture();
                var terminal = new CompletableFuture<VictimResolution>();
                terminals.add(terminal);
                var victimClaim = claim(fixture.requests(), 11L, terminal);
                when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
                org.mockito.Mockito.doAnswer(call -> {
                    entered.countDown();
                    if (!release.await(5, TimeUnit.SECONDS)) { throw new IllegalStateException("test release deadline"); }
                    return true;
                }).when(fixture.requests()).onPreemption(victimClaim, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
                var channel = mock(EngineGrpcClient.class);
                var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
                when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
                var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(fixture.requests()),
                        runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L));
                outcomes.add(preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(fixture.endpoint(), incoming(),
                        org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                        500L, 1000L, () -> true, "slow coordination")));
                verify(channel, org.mockito.Mockito.timeout(1000)).cancelAsync(any(), anyInt(), any(), anyLong());
                ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
            }
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var nextClient = mock(EngineGrpcClient.class);
            var nextSent = new java.util.concurrent.CountDownLatch(1);
            when(nextClient.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> {
                nextSent.countDown();
                return CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            });
            var nextOwner = fixture().requests();
            var nextClaim = claim(nextOwner, 12L, new CompletableFuture<>());
            var next = dispatcher(nextClient).preempt(nextClaim, 500L);
            assertTrue(nextSent.await(1, TimeUnit.SECONDS), "slow coordinators must not occupy stop workers");
            assertEquals(EngineGrpcClient.CancelReply.REQUEST_CLEANED, next.get(1, TimeUnit.SECONDS));
            release.countDown();
        } finally {
            release.countDown();
            for (var terminal : terminals) { terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END)); }
            CompletableFuture.allOf(outcomes.toArray(CompletableFuture[]::new)).get(3, TimeUnit.SECONDS);
        }
    }

    private CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> preempt(
            DecodeCapacityAcquirer acquirer, DecodeCapacityAcquirer.PreemptionCommand command) {
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getScheduler()).thenReturn(mock(AbstractRequestScheduler.class));
        return acquirer.preempt(context, command);
    }

    private org.flexlb.balance.scheduler.DeliveryDispatcher dispatcher(EngineGrpcClient client) {
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "engineClient", client);
        return (org.flexlb.balance.scheduler.DeliveryDispatcher) org.springframework.test.util.ReflectionTestUtils.getField(runtime, "dispatcher");
    }

    private Fixture fixture() {
        AbstractRequestScheduler requests = mock(AbstractRequestScheduler.class);
        org.springframework.test.util.ReflectionTestUtils.setField(requests, "runtime", runtime);
        when(requests.onPreemption(any(), any())).thenReturn(true);
        DecodeEndpoint endpoint = org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        WorkerStatus status = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getGenerationId()).thenReturn(9L);
        when(endpoint.beginPreemption(
                anyLong(),
                anyList(),
                anyLong(),
                anyLong(),
                anyLong(),
                anyInt(),
                any(DecodeResources.AdmissionCapacity.class)))
                .thenReturn(DecodeResources.PreemptionBeginResult.SUCCESS);
        when(endpoint.updatePreemption(anyLong(), any(DecodeResources.PreemptionUpdate.class))).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
        when(endpoint.commitPreemption(anyLong()))
                .thenReturn(new DecodeResources.ReservationHandle(9L, 20L, 100L));
        return new Fixture(requests, endpoint);
    }

    private record Fixture(AbstractRequestScheduler requests, DecodeEndpoint endpoint) { }

    private PreemptionRegistration claim(
            AbstractRequestScheduler owner, long requestId,
            CompletableFuture<VictimResolution> terminal) {
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getRequestId()).thenReturn(requestId);
        when(context.snapshot().delivery()).thenReturn(null);
        when(context.snapshot().cancellation()).thenReturn(null);
        when(context.data().getScheduler()).thenReturn(owner);
        var reservation = new DecodeResources.ReservationHandle(9L, requestId, requestId == 11L ? 101L : 102L);
        var claim = new PreemptionRegistration(context, 1L, reservation, new CancelTarget("10.0.0.1", 9090));
        terminal.whenComplete((result, failure) -> {
            if (failure == null) { runtime.signals().resolvePreemption(claim, result); }
        });
        return claim;
    }

    private static AbstractRequestScheduler.PreemptionEvent phaseEvent(PreemptionCancelPhase phase) {
        return new AbstractRequestScheduler.PreemptionEvent(phase, false, null, null, null);
    }

    private static DecodeRequestView victim(long requestId, long reservationToken) {
        return new DecodeRequestView(
                requestId, 30, 64L, 64L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true, reservationToken, false);
    }
    @Test void lateResolutionWhileAckCoordinationIsQueuedMustNotReopenExpiredAttempt() throws Exception {
        var f = fixture();
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(f.requests(), 11L, terminal);
        when(f.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var workers = (java.util.concurrent.ThreadPoolExecutor) org.springframework.test.util.ReflectionTestUtils.getField(runtime, "requestWorkers");
        var entered = new java.util.concurrent.CountDownLatch(workers.getMaximumPoolSize());
        var queued = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++) runtime.executeRequestWork(context, () -> {
                entered.countDown();
                try { if (!release.await(5, TimeUnit.SECONDS)) throw new AssertionError("release timeout"); }
                catch (InterruptedException e) { throw new AssertionError(e); }
            });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            org.mockito.Mockito.doAnswer(call -> {
                Object result = call.callRealMethod(); queued.countDown(); return result;
            }).when(runtime).executeRequestWork(any(), any());
            var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(f.requests()),
                    runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                    new DecodeResources.AdmissionCapacity(1L, 100L));
            var outcome = acquirer.preempt(context, new DecodeCapacityAcquirer.PreemptionCommand(f.endpoint(), incoming(),
                    org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                    20L, 20L, () -> true, "late resolution after ack budget"));
            assertTrue(queued.await(2, TimeUnit.SECONDS), "ACK continuation or timed-out settlement must be queued");
            assertThrows(TimeoutException.class, () -> outcome.get(150, TimeUnit.MILLISECONDS));
            terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            release.countDown();
            var result = outcome.get(2, TimeUnit.SECONDS);
            System.out.println("AUDIT late_resolution committed=" + result.committed() + " detail=" + result.detail());
            assertFalse(result.committed(), "worker queue delay must not extend the 20ms cleanup window past 150ms");
            verify(f.endpoint(), never()).commitPreemption(anyLong());
            verify(f.endpoint()).abortPreemption(1L);
        } finally { release.countDown(); }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"true,true", "true,false", "false,true", "false,false"})
    @org.junit.jupiter.api.Timeout(15)
    void queuedCoordinationUsesFrozenDeadlineResult(boolean completedInTime, boolean admissionOpen) throws Exception {
        var f = fixture();
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(f.requests(), 11L, terminal);
        when(f.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var workers = (java.util.concurrent.ThreadPoolExecutor)
                org.springframework.test.util.ReflectionTestUtils.getField(runtime, "requestWorkers");
        var entered = new java.util.concurrent.CountDownLatch(workers.getMaximumPoolSize());
        var release = new java.util.concurrent.CountDownLatch(1);
        var registered = new java.util.concurrent.CountDownLatch(1);
        var timeout = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        var cancellation = mock(org.flexlb.balance.scheduler.RequestTaskScheduler.Cancellation.class);
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        var open = new java.util.concurrent.atomic.AtomicBoolean(true);
        org.mockito.Mockito.doAnswer(call -> {
            assertEquals(20L, call.<Long>getArgument(1));
            timeout.set(call.getArgument(0)); registered.countDown(); return cancellation;
        }).when(runtime).scheduleTimeout(any(), anyLong());
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++) runtime.executeRequestWork(context, () -> {
                entered.countDown();
                try { assertTrue(release.await(5, TimeUnit.SECONDS)); }
                catch (InterruptedException e) { throw new AssertionError(e); }
            });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(f.requests()),
                    runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                    new DecodeResources.AdmissionCapacity(1L, 100L));
            var outcome = acquirer.preempt(context, new DecodeCapacityAcquirer.PreemptionCommand(f.endpoint(), incoming(),
                    org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                    1000L, 20L, open::get, "frozen ACK deadline"));
            verify(channel, org.mockito.Mockito.timeout(1000)).cancelAsync(any(), anyInt(), any(), anyLong());
            verify(runtime, never()).scheduleTimeout(any(), anyLong());
            ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
            assertTrue(registered.await(2, TimeUnit.SECONDS), "deadline must register before request workers are available");
            if (!completedInTime) { timeout.get().run(); }
            terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            if (completedInTime) { timeout.get().run(); }
            assertFalse(outcome.isDone(), "coordination remains queued after eligibility is fixed");
            open.set(admissionOpen);
            release.countDown();
            assertEquals(completedInTime && admissionOpen, outcome.get(2, TimeUnit.SECONDS).committed());
            if (completedInTime && admissionOpen) { verify(f.endpoint()).commitPreemption(1L); }
            else { verify(f.endpoint(), never()).commitPreemption(anyLong()); verify(f.endpoint()).abortPreemption(1L); }
            verify(cancellation).cancel();
        } finally { release.countDown(); }
    }

    @Test
    void unresolvedNotFoundStillRejectsAsSoonAsTheOtherVictimResolves() throws Exception {
        var f = fixture();
        var firstTerminal = new CompletableFuture<VictimResolution>();
        var secondTerminal = new CompletableFuture<VictimResolution>();
        var first = claim(f.requests(), 11L, firstTerminal);
        var second = claim(f.requests(), 12L, secondTerminal);
        when(f.requests().tryClaim(any(), anyLong(), any())).thenAnswer(call -> Optional.of(
                call.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call ->
                CompletableFuture.completedFuture(call.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                        ? EngineGrpcClient.CancelReply.ACCEPTED : EngineGrpcClient.CancelReply.NOT_FOUND));
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(f.requests()),
                runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                new DecodeResources.AdmissionCapacity(2L, 100L));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(f.endpoint(), incoming(),
                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L), victim(12L, 102L)),
                1000L, 5000L, () -> true, "NOT_FOUND early rejection"));
        verify(f.requests(), org.mockito.Mockito.timeout(1000)).onPreemption(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
        firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        var result = outcome.get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertFalse(secondTerminal.isDone());
        secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
        assertSame(result, outcome.join());
        verify(f.endpoint(), never()).commitPreemption(anyLong());
        verify(f.endpoint()).abortPreemption(1L);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void completedCanonicalFactUsesTheFrozenDeadlineWhileNotificationIsPaused(boolean expired) throws Exception {
        var f = fixture();
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(f.requests(), 11L, terminal);
        when(f.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var queued = new java.util.concurrent.LinkedBlockingQueue<Runnable>();
        org.mockito.Mockito.doAnswer(call -> {
            queued.add(call.getArgument(1)); return null;
        }).when(runtime).executeRequestWork(any(), any());
        var timeout = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        org.mockito.Mockito.doAnswer(call -> {
            timeout.set(call.getArgument(0));
            return mock(org.flexlb.balance.scheduler.RequestTaskScheduler.Cancellation.class);
        }).when(runtime).scheduleTimeout(any(), anyLong());
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(f.requests()),
                runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                new DecodeResources.AdmissionCapacity(1L, 100L));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(f.endpoint(), incoming(),
                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), List.of(victim(11L, 101L)),
                1000L, 2000L, () -> true, "canonical propagation gap"));
        verify(channel, org.mockito.Mockito.timeout(1000)).cancelAsync(any(), anyInt(), any(), anyLong());
        ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        Runnable coordinator = queued.poll(2, TimeUnit.SECONDS);
        assertTrue(coordinator != null);
        assertTrue(timeout.get() != null);
        @SuppressWarnings("unchecked")
        var signals = (java.util.concurrent.ConcurrentMap<PreemptionRegistration,CompletableFuture<VictimResolution>>)
                org.springframework.test.util.ReflectionTestUtils.getField(runtime.signals(), "preemptions");
        var canonical = signals.get(victimClaim);
        var factVisible = new java.util.concurrent.CountDownLatch(1);
        var resumeNotification = new java.util.concurrent.CountDownLatch(1);
        // CompletableFuture completes the result before running dependents. Pause propagation
        // at this legal scheduling boundary before the aggregate notification is delivered.
        canonical.whenComplete((value,failure) -> {
            factVisible.countDown();
            try { assertTrue(resumeNotification.await(5, TimeUnit.SECONDS)); }
            catch (InterruptedException e) { throw new AssertionError(e); }
        });
        if (expired) { timeout.get().run(); }
        var producer = Thread.ofPlatform().start(() -> terminal.complete(
                new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END)));
        try {
            assertTrue(factVisible.await(2, TimeUnit.SECONDS));
            assertTrue(runtime.signals().resolved(victimClaim) != null);
            coordinator.run();
            var result = outcome.get(2, TimeUnit.SECONDS);
            assertEquals(!expired, result.committed(), "read the recorded proof without reopening a completed timeout");
        } finally {
            resumeNotification.countDown(); producer.join(2000);
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "true,false", "false,true", "true,true"})
    void finishedAttemptReleasesObserversWithoutCompletingVictimSignals(boolean expire, boolean mixed) throws Exception {
        var predicate = retiredAdmissionPredicate(expire, mixed);
        for (int i = 0; i < 10; i++) {
            System.gc();
            Thread.sleep(25);
        }
        org.junit.jupiter.api.Assertions.assertNull(predicate.get(),
                "a finished wait must release its attempt even when the victim signal never completes");
    }

    private java.lang.ref.WeakReference<java.util.function.BooleanSupplier> retiredAdmissionPredicate(
            boolean expire, boolean mixed) throws Exception {
        var f = fixture();
        var firstTerminal = new CompletableFuture<VictimResolution>();
        var secondTerminal = new CompletableFuture<VictimResolution>();
        var first = claim(f.requests(), 11L, firstTerminal);
        var second = claim(f.requests(), 12L, secondTerminal);
        when(f.requests().tryClaim(any(), anyLong(), any())).thenAnswer(call -> Optional.of(
                call.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call ->
                CompletableFuture.completedFuture(call.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                        && (expire || mixed) ? EngineGrpcClient.CancelReply.ACCEPTED : EngineGrpcClient.CancelReply.NOT_FOUND));
        var deadline = new CompletableFuture<Runnable>();
        org.mockito.Mockito.doAnswer(call -> {
            deadline.complete(call.getArgument(0));
            return mock(org.flexlb.balance.scheduler.RequestTaskScheduler.Cancellation.class);
        }).when(runtime).scheduleTimeout(any(), anyLong());
        var acquirer = new DecodeCapacityAcquirer(dispatcher(channel), SchedulerTestSupport.repository(f.requests()),
                runtime, mock(RequestSchedulerReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                new DecodeResources.AdmissionCapacity(2L, 100L));
        // A distinct instance: a stateless lambda could be cached for the lifetime of the JVM.
        var admissionOpen = new java.util.function.BooleanSupplier() {
            @Override public boolean getAsBoolean() { return true; }
        };
        var victims = mixed ? List.of(victim(11L, 101L), victim(12L, 102L)) : List.of(victim(11L, 101L));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(f.endpoint(), incoming(),
                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), victims,
                1000L, 5000L, admissionOpen, "observer lifetime"));
        if (expire) { deadline.get(2, TimeUnit.SECONDS).run(); }
        else if (mixed) {
            verify(f.requests(), org.mockito.Mockito.timeout(1000)).onPreemption(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
            firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        }
        assertFalse(outcome.get(2, TimeUnit.SECONDS).committed());
        var unresolved = mixed ? second : first;
        assertFalse(runtime.signals().preemption(unresolved).toCompletableFuture().isDone(),
                "ending the local wait must not manufacture remote cleanup evidence or cancel its signal");
        verify(f.endpoint(), never()).commitPreemption(anyLong());
        // Mockito retains submitted Runnable objects after execution; real executors discard them.
        org.mockito.Mockito.clearInvocations(runtime);
        return new java.lang.ref.WeakReference<>(admissionOpen);
    }

}
