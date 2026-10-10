package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestRequirements.DecodeMode;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.balance.scheduler.SchedulerRuntime;
import org.flexlb.balance.scheduler.SchedulerTestSupport;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
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
        finally { assertTrue(runtime.cleanupExecutor().isTerminated(), "the runtime owns every preemption deadline thread"); }
    }

    private static RequestRequirements incoming(long maxRequests) {
        return new RequestRequirements(20L, org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE), 64L,
                new DecodeResources.AdmissionCapacity(maxRequests, 100L),
                DecodeMode.PREEMPT_AT_PLACEMENT, mock(DecodeCostFormula.class), 64L, null, List.of(), 0L, true, 0);
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

        EngineCancelChannel cancelChannel = mock(EngineCancelChannel.class);
        when(cancelChannel.cancel(any(), anyLong(), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelAck.ACCEPTED));
        DecodeCapacityAcquirer acquirer =
                new DecodeCapacityAcquirer(cancelChannel, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests), runtime, mock(RequestSchedulerReporter.class));
        CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> result =
                preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        endpoint, incoming(2L),
                        List.of(victim(11L, 101L), victim(12L, 102L)),
                        1_000L, 1_000L, () -> true, "test"));

        assertFalse(result.isDone());
        firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        assertFalse(result.isDone(), "one terminal cannot release two victims");
        secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));

        assertTrue(result.get(1, TimeUnit.SECONDS).committed());
        assertEquals(new DecodeResources.ReservationHandle(9L, 20L, 100L), result.join().reservation());
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq(11L), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong());
        verify(cancelChannel).cancel(eq(new CancelTarget("10.0.0.1", 9090)), eq(12L), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong());
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
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), eq(CancelReason.PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
        var acquirer = new DecodeCapacityAcquirer(channel, SchedulerTestSupport.repository(fixture.requests()),
                runtime, mock(RequestSchedulerReporter.class));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                50L, 1_000L, () -> true, "exact resolution")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(fixture.requests(), never()).releasePreemption(victimClaim);
    }

    @Test
    void resumedDeliveryRequiresIndependentDecodeReleaseProof() throws Exception {
        Fixture fixture = fixture();
        var resolution = CompletableFuture.completedFuture(
                new VictimResolution(11L, VictimResolution.Outcome.DELIVERY_RESUMED));
        var victimClaim = claim(fixture.requests(), 11L, resolution);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        when(fixture.endpoint().commitPreemption(anyLong())).thenReturn(null);
        var channel = mock(EngineCancelChannel.class);
        var acquirer = new DecodeCapacityAcquirer(channel, SchedulerTestSupport.repository(fixture.requests()),
                runtime, mock(RequestSchedulerReporter.class));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                50L, 1_000L, () -> true, "delivery resumed")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed(), "request resolution cannot manufacture Decode release proof");
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(fixture.requests(), never()).releasePreemption(victimClaim);
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class, names = {"ACCEPTED", "FAILED", "REQUEST_FENCED"})
    void timeoutBeginsAfterAckAndLateTerminalCannotReopenIncoming(
            EngineCancelChannel.CancelAck acknowledgement) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> terminal = new CompletableFuture<>();
        PreemptionRegistration victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(victimClaim));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        CompletableFuture<EngineCancelChannel.CancelAck> ack = new CompletableFuture<>();
        when(channel.cancel(any(), anyLong(), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong())).thenReturn(ack);
        DecodeCapacityAcquirer acquirer =
                new DecodeCapacityAcquirer(channel, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> outcome =
                preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        fixture.endpoint(), incoming(1L),
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
        verify(fixture.requests(), never()).releasePreemption(victimClaim);
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
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong())).thenAnswer(invocation ->
                CompletableFuture.completedFuture(invocation.<Long>getArgument(1) == 11L
                        ? EngineCancelChannel.CancelAck.ACCEPTED : EngineCancelChannel.CancelAck.FAILED));
        DecodeCapacityAcquirer acquirer = new DecodeCapacityAcquirer(channel, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        if (terminalBeforeSend) {
            firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        }
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(2L),
                List.of(victim(11L, 101L), victim(12L, 102L)),
                50L, 1_000L, () -> true, "test"));
        if (!terminalBeforeSend) {
            verify(channel).cancel(any(), eq(11L), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong());
            firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            assertFalse(outcome.isDone(), "one of two terminal observations cannot finish the aggregate");
        } else {
            verify(channel, never()).cancel(any(), eq(11L), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong());
        }
        var timedOut = outcome.get(3L, TimeUnit.SECONDS);
        assertFalse(timedOut.committed());
        assertTrue(timedOut.controlFailure());
        verify(channel).cancel(any(), eq(12L), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong());
        verify(fixture.requests(), never()).releasePreemption(first);
        verify(fixture.requests(), never()).releasePreemption(second);
        verify(fixture.requests()).updatePreemption(eq(second), eq(org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertFalse(secondTerminal.isDone());
        verify(fixture.endpoint()).abortPreemption(1L);
        secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"FAILED", "NOT_FOUND", "REQUEST_FENCED", "REQUEST_CLEANED"})
    void reversedAcknowledgementsStayBoundToTheirVictims(
            EngineCancelChannel.CancelAck secondReply) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> firstTerminal = new CompletableFuture<>();
        CompletableFuture<VictimResolution> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenAnswer(invocation -> Optional.of(
                        invocation.<DecodeResources.ReservationHandle>getArgument(0).requestId() == 11L ? first : second));
        when(fixture.requests().onPreemptionCleanupProven(eq(second), eq(fixture.endpoint()), any(), any()))
                .thenAnswer(invocation -> {
                    secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
                    return true;
                });
        CompletableFuture<EngineCancelChannel.CancelAck> firstAck = new CompletableFuture<>();
        CompletableFuture<EngineCancelChannel.CancelAck> secondAck = new CompletableFuture<>();
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), org.mockito.ArgumentMatchers.eq(CancelReason.PRIORITY_PREEMPTED), anyLong())).thenAnswer(invocation ->
                invocation.<Long>getArgument(1) == 11L ? firstAck : secondAck);
        var acquirer = new DecodeCapacityAcquirer(channel, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(2L),
                List.of(victim(11L, 101L), victim(12L, 102L)),
                1_000L, 1_000L, () -> true, "test"));

        secondAck.complete(secondReply);
        assertFalse(result.isDone());
        verify(fixture.requests(), never()).updatePreemption(first, PreemptionCancelPhase.CANCEL_REQUESTED);
        verify(fixture.requests(), never()).onPreemptionCleanupProven(eq(second), eq(fixture.endpoint()), any(), any());
        firstAck.complete(EngineCancelChannel.CancelAck.ACCEPTED);
        verify(fixture.requests()).updatePreemption(first, PreemptionCancelPhase.CANCEL_REQUESTED);
        switch (secondReply) {
            case FAILED -> verify(fixture.requests()).updatePreemption(second, PreemptionCancelPhase.CANCEL_UNKNOWN);
            case NOT_FOUND -> verify(fixture.requests()).updatePreemption(second, PreemptionCancelPhase.NOT_FOUND_STALE);
            case REQUEST_FENCED -> verify(fixture.requests()).updatePreemption(second, PreemptionCancelPhase.CANCEL_REQUESTED);
            case REQUEST_CLEANED -> {
                verify(fixture.requests()).onPreemptionCleanupProven(second, fixture.endpoint(),
                        new DecodeResources.ReservationHandle(9L, 12L, 102L), "test");
            }
            default -> throw new AssertionError("unexpected test reply");
        }
        assertFalse(result.isDone(), "ACKs cannot replace the first victim's terminal proof");
        if (secondReply != EngineCancelChannel.CancelAck.REQUEST_CLEANED) {
            secondTerminal.complete(new VictimResolution(12L, VictimResolution.Outcome.REQUEST_END));
        }
        firstTerminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
        assertTrue(result.get(1L, TimeUnit.SECONDS).committed());
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint(), never()).abortPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"REQUEST_FENCED", "REQUEST_CLEANED"})
    void onlyDownstreamCleanupProofCanReplaceDecodeTerminal(EngineCancelChannel.CancelAck reply) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<VictimResolution> terminal = new CompletableFuture<>();
        PreemptionRegistration claim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(DecodeResources.ReservationHandle.class), anyLong(), any()))
                .thenReturn(Optional.of(claim));
        when(fixture.requests().onPreemptionCleanupProven(eq(claim), eq(fixture.endpoint()), any(), any()))
                .thenAnswer(invocation -> {
                    terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
                    return true;
                });
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), eq(CancelReason.PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(reply));
        var acquirer = new DecodeCapacityAcquirer(channel,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                1_000L, 1_000L, () -> true, "test"));
        if (reply == EngineCancelChannel.CancelAck.REQUEST_FENCED) {
            assertFalse(outcome.isDone(), "Prefill fence alone does not prove Decode resource release");
            verify(fixture.requests(), never()).onPreemptionCleanupProven(eq(claim), eq(fixture.endpoint()), any(), any());
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
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        var acquirer = new DecodeCapacityAcquirer(channel,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(2L), List.of(victim(11L, 101L), victim(12L, 102L)),
                1000L, 1000L, () -> true, "test")).get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        verify(fixture.requests()).releasePreemption(first);
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
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        var acquirer = new DecodeCapacityAcquirer(channel,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(2L), List.of(victim(11L, 101L), victim(12L, 102L)),
                1000L, 1000L, () -> true, "test")).get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertTrue(result.controlFailure());
        verify(fixture.requests()).releasePreemption(first);
        verify(fixture.endpoint(), never()).beginPreemption(anyLong(), anyList(), anyLong(),
                anyLong(), anyLong(), anyInt(), any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    void staleEndpointGenerationCannotCancelANewerRouteWithTheSameRequestAndToken() throws Exception {
        Fixture fixture = fixture();
        var stale = new DecodeResources.ReservationHandle(9L, 11L, 101L);
        when(fixture.requests().tryClaim(eq(stale), anyLong(), any())).thenReturn(Optional.empty());
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        var acquirer = new DecodeCapacityAcquirer(channel,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests()), runtime, mock(RequestSchedulerReporter.class));
        var result = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
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
                var channel = mock(EngineCancelChannel.class);
                when(channel.cancel(any(), anyLong(), eq(CancelReason.PRIORITY_PREEMPTED), anyLong()))
                        .thenReturn(CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
                var acquirer = new DecodeCapacityAcquirer(channel, SchedulerTestSupport.repository(fixture.requests()),
                        runtime, mock(RequestSchedulerReporter.class));
                outcomes.add(preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                        fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                        50L, 20L, () -> true, "timer isolation")));
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS), "both settlements must be blocked before probing the timer");
            runtime.cleanupExecutor().schedule(deadlineRan::countDown, 1, TimeUnit.MILLISECONDS);
            assertTrue(deadlineRan.await(1, TimeUnit.SECONDS), "blocked settlement must not occupy the shared deadline threads");
            assertTrue(settlementThreads.stream().allMatch(name -> name.startsWith("request-continuation-")),
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
        var timer = mock(java.util.concurrent.ScheduledExecutorService.class);
        var deadline = mock(java.util.concurrent.ScheduledFuture.class);
        when(timer.schedule(any(java.util.concurrent.Callable.class), anyLong(), any(TimeUnit.class)))
                .thenAnswer(call -> {
                    call.<java.util.concurrent.Callable<?>>getArgument(0).call();
                    terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
                    return deadline;
                });
        var protocolRuntime = org.mockito.Mockito.spy(runtime);
        org.mockito.Mockito.doReturn(timer).when(protocolRuntime).cleanupExecutor();
        var channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), eq(CancelReason.PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
        var acquirer = new DecodeCapacityAcquirer(channel, SchedulerTestSupport.repository(fixture.requests()),
                protocolRuntime, mock(RequestSchedulerReporter.class));

        var outcome = preempt(acquirer, new DecodeCapacityAcquirer.PreemptionCommand(
                fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                50L, 1L, () -> true, "schedule return race")).get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed(), "the deadline must freeze rejection before late request resolution");
        assertTrue(terminal.isDone(), "late request resolution remains available to its resource owner");
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(deadline).cancel(false);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,true", "true,true", "true,false"})
    @org.junit.jupiter.api.Timeout(15)
    void settlementPreservesRequestOrderAndShutdownDrainsIt(
            boolean terminalWins, boolean admissionOpenAtSettlement) throws Exception {
        runtime = org.mockito.Mockito.spy(runtime);
        Fixture fixture = fixture();
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class);
        when(context.scheduler()).thenReturn(fixture.requests());
        var terminal = new CompletableFuture<VictimResolution>();
        var victimClaim = claim(fixture.requests(), 11L, terminal);
        when(fixture.requests().tryClaim(any(), anyLong(), any())).thenReturn(Optional.of(victimClaim));
        var channel = mock(EngineCancelChannel.class);
        when(channel.cancel(any(), anyLong(), eq(CancelReason.PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.ACCEPTED));
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        var settlementQueued = new java.util.concurrent.CountDownLatch(1);
        var afterSettlement = new CompletableFuture<Boolean>();
        var shutdown = new CompletableFuture<Void>();
        var admissionOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
        Thread closer = null;
        try {
            runtime.executeContinuation(context, () -> {
                entered.countDown();
                try { assertTrue(release.await(10, TimeUnit.SECONDS)); }
                catch (InterruptedException failure) { throw new AssertionError(failure); }
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            org.mockito.Mockito.doAnswer(call -> {
                call.callRealMethod();
                settlementQueued.countDown();
                return null;
            }).when(runtime).executeContinuation(eq(context), any());
            var acquirer = new DecodeCapacityAcquirer(channel, SchedulerTestSupport.repository(fixture.requests()),
                    runtime, mock(RequestSchedulerReporter.class));
            var outcome = acquirer.preempt(context, new DecodeCapacityAcquirer.PreemptionCommand(
                    fixture.endpoint(), incoming(1L), List.of(victim(11L, 101L)),
                    50L, terminalWins ? 5_000L : 20L, admissionOpen::get, "ordered settlement"));
            if (terminalWins) {
                terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            }
            assertTrue(settlementQueued.await(5, TimeUnit.SECONDS));
            assertFalse(outcome.isDone(), "settlement cannot overtake the request's earlier continuation");
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
            verify(fixture.endpoint(), never()).abortPreemption(anyLong());
            if (!terminalWins) {
                assertFalse(terminal.isDone(), "timeout retains the exact late resolution observer");
                terminal.complete(new VictimResolution(11L, VictimResolution.Outcome.REQUEST_END));
            }
            runtime.executeContinuation(context, () -> afterSettlement.complete(outcome.isDone()));
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
            assertTrue(afterSettlement.get(5, TimeUnit.SECONDS), "later request work must follow settlement");
            shutdown.get(5, TimeUnit.SECONDS);
            assertTrue(runtime.cleanupExecutor().isTerminated());
            if (!terminalWins || !admissionOpenAtSettlement) {
                verify(fixture.endpoint(), never()).commitPreemption(anyLong());
                verify(fixture.endpoint()).abortPreemption(1L);
            }
        } finally {
            release.countDown();
            if (closer != null) { closer.join(5_000); }
        }
    }

    private static CompletableFuture<DecodeCapacityAcquirer.PreemptionResult> preempt(
            DecodeCapacityAcquirer acquirer, DecodeCapacityAcquirer.PreemptionCommand command) {
        var context = mock(org.flexlb.balance.scheduler.RequestContext.class);
        when(context.scheduler()).thenReturn(mock(AbstractRequestScheduler.class));
        return acquirer.preempt(context, command);
    }

    private static Fixture fixture() {
        AbstractRequestScheduler requests = mock(AbstractRequestScheduler.class);
        when(requests.updatePreemption(any(), any())).thenReturn(true);
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
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
        when(endpoint.updatePreemption(anyLong(), any(DecodeResources.PreemptionUpdate.class))).thenReturn(true);
        when(endpoint.commitPreemption(anyLong()))
                .thenReturn(new DecodeResources.ReservationHandle(9L, 20L, 100L));
        return new Fixture(requests, endpoint);
    }

    private record Fixture(AbstractRequestScheduler requests, DecodeEndpoint endpoint) { }

    private static PreemptionRegistration claim(
            AbstractRequestScheduler owner, long requestId,
            CompletableFuture<VictimResolution> terminal) {
        PreemptionRegistration claim = mock(PreemptionRegistration.class);
        when(claim.cancelTarget()).thenReturn(new CancelTarget("10.0.0.1", 9090));
        when(claim.requestId()).thenReturn(requestId);
        when(claim.scheduler()).thenReturn(owner);
        when(claim.attemptToken()).thenReturn(1L);
        when(claim.requestResolution()).thenReturn(terminal);
        when(claim.resolvedRequestResult()).thenAnswer(ignored ->
                terminal.handle((result, failure) -> result).getNow(null));
        return claim;
    }

    private static DecodeRequestView victim(long requestId, long reservationToken) {
        return new DecodeRequestView(
                requestId, 30, 64L, 64L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true, reservationToken, false);
    }
}
