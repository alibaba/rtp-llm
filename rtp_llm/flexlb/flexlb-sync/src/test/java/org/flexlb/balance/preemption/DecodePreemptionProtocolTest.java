package org.flexlb.balance.preemption;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestTaskExecutor;
import org.flexlb.balance.scheduler.SchedulerTestServices;
import org.flexlb.balance.scheduler.SchedulerTestSupport;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.TaskTimer;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
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

/**
 * Exercises request selected for preemption cancellation and resource release across deadlines, stale callbacks and worker replacement.
 * A cancellation receipt alone cannot authorize reuse of a request selected for preemption's Decode resources.
 */
class DecodePreemptionProtocolTest {
    private final java.util.Map<
                    AbstractRequestScheduler, java.util.Set<DecodeResources.RequestReservation>>
            releasedResources = new java.util.HashMap<>();
    private final java.util.Map<AbstractRequestScheduler, java.util.Set<Runnable>>
            resourceListeners = new java.util.HashMap<>();
    private SchedulerTestServices runtime;

    private final DefaultDecodePreemption registrations = mock(DefaultDecodePreemption.class);

    @BeforeEach
    void createRuntime() {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        runtime =
                SchedulerTestSupport.runtime(
                        SchedulerTestSupport.create(
                                service,
                                mock(DeliveryMetricsReporter.class),
                                mock(RequestSchedulerReporter.class),
                                mock(TheoryCacheHitReporter.class)));
    }

    @AfterEach
    void closeRuntime() {
        try {
            runtime.shutdown(
                    org.flexlb.balance.scheduler.ShutdownMode.DRAIN, Duration.ofSeconds(30L));
        } finally {
            assertTrue(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime)
                            .isTerminated(),
                    "the runtime owns every preemption deadline thread");
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void commitsOnlyAfterEveryExactVictimIsTerminal(boolean cancelledResult) throws Exception {
        Fixture fixture = fixture();
        AbstractRequestScheduler requests = fixture.requests();
        DecodeEndpoint endpoint = fixture.endpoint();

        CompletableFuture<PreemptionRequestResult> firstTerminal = new CompletableFuture<>();
        CompletableFuture<PreemptionRequestResult> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenAnswer(
                        invocation ->
                                Optional.of(
                                        invocation
                                                        .<DecodeResources.RequestReservation>
                                                                getArgument(0)
                                                        .requestId()
                                                        .equals("11")
                                                ? first
                                                : second));

        EngineGrpcClient engineClient = mock(EngineGrpcClient.class);
        when(engineClient.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        DefaultDecodePreemption acquirer =
                newPreemption(
                        dispatcher(engineClient),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        CompletableFuture<DecodePreemption.PreemptionResult> result =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                endpoint,
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L), requestToPreempt(12L, 102L)),
                                1_000L,
                                1_000L,
                                () -> true,
                                "test"));

        assertFalse(result.isDone());
        assertFalse(acquirer.awaitIdle(System.nanoTime()));
        if (cancelledResult) assertTrue(result.cancel(false));
        assertFalse(
                acquirer.awaitIdle(System.nanoTime()),
                "cancelling a result is not resource settlement");
        firstTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        assertEquals(cancelledResult, result.isDone(), "one terminal cannot release two victims");
        secondTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(12L), PreemptionRequestResult.Outcome.REQUEST_END));

        assertTrue(acquirer.awaitIdle(System.nanoTime() + TimeUnit.SECONDS.toNanos(1L)));
        if (cancelledResult) assertTrue(result.isCancelled());
        else {
            assertTrue(result.get(1, TimeUnit.SECONDS).committed());
            assertEquals(
                    new DecodeResources.RequestReservation(9L, String.valueOf(20L), 100L),
                    result.join().reservation());
        }
        verify(engineClient)
                .cancelAsync(
                        eq("10.0.0.1"),
                        eq(9090),
                        eq(CancelReason.PRIORITY_PREEMPTED.toRequest(String.valueOf(11L))),
                        anyLong());
        verify(engineClient)
                .cancelAsync(
                        eq("10.0.0.1"),
                        eq(9090),
                        eq(CancelReason.PRIORITY_PREEMPTED.toRequest(String.valueOf(12L))),
                        anyLong());
        verify(endpoint).commitPreemption(1L);
        verify(endpoint, never()).abortPreemption(anyLong());
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "releaseBeforeSubscribe",
                "releaseAfterSubscribe",
                "requestCompletedBeforeRelease",
                "incomingCancelled"
            })
    void preemptionWaitsForResourcesRatherThanRequestCompletion(String scenario) throws Exception {
        var fixture = fixture();
        var requestCompletion = new CompletableFuture<PreemptionRequestResult>();
        if (scenario.equals("releaseBeforeSubscribe")) {
            when(SchedulerTestSupport.preemption(fixture.requests()).applyEvent(any(), any()))
                    .thenReturn(false);
        }
        var registration = claim(fixture.requests(), 11L, requestCompletion);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(registration));
        var resourcesReleased = new java.util.concurrent.atomic.AtomicBoolean();
        var incomingOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
        var listener = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        var unsubscribed = new java.util.concurrent.atomic.AtomicBoolean();
        org.mockito.Mockito.doAnswer(call -> resourcesReleased.get())
                .when(fixture.endpoint())
                .isPreemptionResourceReleased(anyLong(), any());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            listener.set(call.getArgument(0));
                            if (scenario.equals("releaseBeforeSubscribe"))
                                resourcesReleased.set(true);
                            return (Runnable) () -> unsubscribed.set(true);
                        })
                .when(fixture.endpoint())
                .watchEngineDispatchCapacity(any());
        var client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var acquirer =
                newPreemption(
                        dispatcher(client),
                        SchedulerTestSupport.repository(fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L)),
                                1000L,
                                5000L,
                                incomingOpen::get,
                                "resource wait"));
        if (!scenario.equals("releaseBeforeSubscribe")) {
            assertFalse(result.isDone());
            if (scenario.equals("requestCompletedBeforeRelease")) {
                requestCompletion.complete(
                        new PreemptionRequestResult(
                                "11", PreemptionRequestResult.Outcome.REQUEST_END));
                assertFalse(result.isDone(), "B ending must not release resources on A's behalf");
            }
            if (scenario.equals("incomingCancelled")) incomingOpen.set(false);
            resourcesReleased.set(true);
            org.junit.jupiter.api.Assertions.assertNotNull(
                    listener.get(), "must subscribe to resource changes");
            listener.get().run();
        }
        assertEquals(
                !scenario.equals("incomingCancelled"), result.get(2, TimeUnit.SECONDS).committed());
        if (!scenario.equals("requestCompletedBeforeRelease"))
            assertFalse(
                    requestCompletion.isDone(),
                    "resource release must not require B's request completion");
        assertTrue(unsubscribed.get(), "completed waits must remove the capacity listener");
        if (scenario.equals("incomingCancelled"))
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @ValueSource(strings = {"missing", "foreign", "failed"})
    void invalidRequestCompletionDoesNotReplaceResourceRelease(String completionKind)
            throws Exception {
        Fixture fixture = fixture();
        var resolution = new CompletableFuture<PreemptionRequestResult>();
        switch (completionKind) {
            case "missing" -> resolution.complete(null);
            case "foreign" ->
                    resolution.complete(
                            new PreemptionRequestResult(
                                    String.valueOf(99L),
                                    PreemptionRequestResult.Outcome.REQUEST_END));
            case "failed" ->
                    resolution.completeExceptionally(
                            new IllegalStateException("resolution failed"));
            default -> throw new AssertionError(completionKind);
        }
        PreemptionRegistration preemptionRegistration =
                claim(fixture.requests(), 11L, new CompletableFuture<>());
        preemptionRegistration = org.mockito.Mockito.spy(preemptionRegistration);
        org.mockito.Mockito.doReturn(resolution).when(preemptionRegistration).completion();
        when(preemptionRegistration.result())
                .thenAnswer(call -> resolution.handle((result, failure) -> result).getNow(null));
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());

        var outcome =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        50L,
                                        20L,
                                        () -> true,
                                        "exact resolution"))
                        .get(1L, TimeUnit.SECONDS);

        assertFalse(outcome.committed());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(channel).cancelAsync(any(), anyInt(), any(), anyLong());
    }

    @Test
    void resumedDeliveryRequiresIndependentDecodeReleaseProof() throws Exception {
        Fixture fixture = fixture();
        var resolution =
                CompletableFuture.completedFuture(
                        new PreemptionRequestResult(
                                String.valueOf(11L),
                                PreemptionRequestResult.Outcome.DELIVERY_RESUMED));
        var preemptionRegistration = claim(fixture.requests(), 11L, resolution);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        when(fixture.endpoint().commitPreemption(anyLong())).thenReturn(null);
        var channel = mock(EngineGrpcClient.class);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());

        var outcome =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        50L,
                                        1_000L,
                                        () -> true,
                                        "delivery resumed"))
                        .get(1L, TimeUnit.SECONDS);

        assertFalse(
                outcome.committed(), "request resolution cannot manufacture Decode release proof");
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(
                        preemptionRegistration,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @ParameterizedTest
    @EnumSource(
            value = EngineGrpcClient.CancelReply.class,
            names = {"ACCEPTED", "FAILED", "REQUEST_FENCED"})
    void timeoutIncludesAckWaitAndLateTerminalCannotReopenIncoming(
            EngineGrpcClient.CancelReply acknowledgement) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<PreemptionRequestResult> terminal = new CompletableFuture<>();
        PreemptionRegistration preemptionRegistration = claim(fixture.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        CompletableFuture<EngineGrpcClient.CancelReply> ack = new CompletableFuture<>();
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(ack);
        DefaultDecodePreemption acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        CompletableFuture<DecodePreemption.PreemptionResult> outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L)),
                                50L,
                                20L,
                                () -> true,
                                "test"));

        // The admission budget includes the outstanding RPC; a late ACK cannot restart it.
        var timedOut = outcome.get(1L, TimeUnit.SECONDS);
        ack.complete(acknowledgement);
        assertFalse(timedOut.committed());
        assertTrue(timedOut.controlFailure());
        assertEquals("cancel_terminal_unknown", timedOut.detail());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(
                        preemptionRegistration,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        assertFalse(
                terminal.isDone(),
                "timing out admission must retain the victim terminal observation");

        terminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void partialTerminalTimeoutRetainsUnknownVictimAndNeverReopensIncoming(
            boolean terminalBeforeSend) throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<PreemptionRequestResult> firstTerminal = new CompletableFuture<>();
        CompletableFuture<PreemptionRequestResult> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenAnswer(
                        invocation ->
                                Optional.of(
                                        invocation
                                                        .<DecodeResources.RequestReservation>
                                                                getArgument(0)
                                                        .requestId()
                                                        .equals("11")
                                                ? first
                                                : second));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenAnswer(
                        invocation ->
                                CompletableFuture.completedFuture(
                                        invocation.<CancelRequestPB>getArgument(2).getRequestId()
                                                        == 11L
                                                ? EngineGrpcClient.CancelReply.ACCEPTED
                                                : EngineGrpcClient.CancelReply.FAILED));
        DefaultDecodePreemption acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        if (terminalBeforeSend) {
            firstTerminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        }
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L), requestToPreempt(12L, 102L)),
                                50L,
                                1_000L,
                                () -> true,
                                "test"));
        if (!terminalBeforeSend) {
            verify(channel, org.mockito.Mockito.timeout(2000))
                    .cancelAsync(
                            any(),
                            anyInt(),
                            eq(CancelReason.PRIORITY_PREEMPTED.toRequest(String.valueOf(11L))),
                            anyLong());
            firstTerminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            assertFalse(
                    outcome.isDone(),
                    "one of two terminal observations cannot finish the aggregate");
        } else {
            verify(channel, never())
                    .cancelAsync(
                            any(),
                            anyInt(),
                            eq(CancelReason.PRIORITY_PREEMPTED.toRequest(String.valueOf(11L))),
                            anyLong());
        }
        var timedOut = outcome.get(3L, TimeUnit.SECONDS);
        assertFalse(timedOut.committed());
        assertTrue(timedOut.controlFailure());
        verify(channel)
                .cancelAsync(
                        any(),
                        anyInt(),
                        eq(CancelReason.PRIORITY_PREEMPTED.toRequest(String.valueOf(12L))),
                        anyLong());
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(
                        first,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(
                        second,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        // The RPC failure and the attempt timeout both report uncertainty on the same identity.
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        org.mockito.Mockito.times(2))
                .applyEvent(
                        eq(second),
                        eq(
                                phaseEvent(
                                        org.flexlb.balance.preemption.PreemptionCancelPhase
                                                .CANCEL_UNKNOWN)));
        assertFalse(secondTerminal.isDone());
        verify(fixture.endpoint()).abortPreemption(1L);
        secondTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(12L), PreemptionRequestResult.Outcome.REQUEST_END));
        assertSame(timedOut, outcome.join());
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(
            value = EngineGrpcClient.CancelReply.class,
            names = {"FAILED", "NOT_FOUND", "REQUEST_FENCED", "REQUEST_CLEANED"})
    void reversedAcknowledgementsStayBoundToTheirVictims(EngineGrpcClient.CancelReply secondReply)
            throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<PreemptionRequestResult> firstTerminal = new CompletableFuture<>();
        CompletableFuture<PreemptionRequestResult> secondTerminal = new CompletableFuture<>();
        PreemptionRegistration first = claim(fixture.requests(), 11L, firstTerminal);
        PreemptionRegistration second = claim(fixture.requests(), 12L, secondTerminal);
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenAnswer(
                        invocation ->
                                Optional.of(
                                        invocation
                                                        .<DecodeResources.RequestReservation>
                                                                getArgument(0)
                                                        .requestId()
                                                        .equals("11")
                                                ? first
                                                : second));
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(fixture.requests())
                        .applyEvent(
                                eq(second),
                                argThat(
                                        event ->
                                                event.endpoint() == fixture.endpoint()
                                                        && event.reservation() != null)))
                .thenAnswer(
                        invocation -> {
                            secondTerminal.complete(
                                    new PreemptionRequestResult(
                                            String.valueOf(12L),
                                            PreemptionRequestResult.Outcome.REQUEST_END));
                            return true;
                        });
        CompletableFuture<EngineGrpcClient.CancelReply> firstAck = new CompletableFuture<>();
        CompletableFuture<EngineGrpcClient.CancelReply> secondAck = new CompletableFuture<>();
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenAnswer(
                        invocation ->
                                invocation.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                                        ? firstAck
                                        : secondAck);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L), requestToPreempt(12L, 102L)),
                                1_000L,
                                1_000L,
                                () -> true,
                                "test"));

        secondAck.complete(secondReply);
        assertFalse(result.isDone());
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(first, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        never())
                .applyEvent(
                        eq(second),
                        argThat(
                                event ->
                                        event.endpoint() == fixture.endpoint()
                                                && event.reservation() != null));
        firstAck.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        verify(channel, org.mockito.Mockito.timeout(2000).times(2))
                .cancelAsync(any(), anyInt(), any(), anyLong());
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()),
                        org.mockito.Mockito.timeout(2000))
                .applyEvent(first, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        switch (secondReply) {
            case FAILED ->
                    verify(
                                    org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                            fixture.requests()),
                                    org.mockito.Mockito.timeout(2000).times(2))
                            .applyEvent(second, phaseEvent(PreemptionCancelPhase.CANCEL_UNKNOWN));
            case NOT_FOUND ->
                    verify(
                                    org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                            fixture.requests()),
                                    org.mockito.Mockito.timeout(2000))
                            .applyEvent(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
            case REQUEST_FENCED ->
                    verify(
                                    org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                            fixture.requests()),
                                    org.mockito.Mockito.timeout(2000))
                            .applyEvent(second, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
            case REQUEST_CLEANED -> {
                verify(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                        fixture.requests()),
                                org.mockito.Mockito.timeout(2000))
                        .applyEvent(
                                second,
                                new org.flexlb.balance.preemption.PreemptionCompletion
                                        .PreemptionEvent(
                                        null,
                                        false,
                                        fixture.endpoint(),
                                        new DecodeResources.RequestReservation(
                                                9L, String.valueOf(12L), 102L),
                                        "test"));
            }
            default -> throw new AssertionError("unexpected test reply");
        }
        assertFalse(result.isDone(), "ACKs cannot replace the first victim's terminal proof");
        if (secondReply != EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
            secondTerminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(12L), PreemptionRequestResult.Outcome.REQUEST_END));
        }
        firstTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        assertTrue(result.get(1L, TimeUnit.SECONDS).committed());
        verify(fixture.endpoint()).commitPreemption(1L);
        verify(fixture.endpoint(), never()).abortPreemption(anyLong());
    }

    @ParameterizedTest
    @EnumSource(
            value = EngineGrpcClient.CancelReply.class,
            names = {"REQUEST_FENCED", "REQUEST_CLEANED"})
    void onlyDownstreamCleanupProofCanReplaceDecodeTerminal(EngineGrpcClient.CancelReply reply)
            throws Exception {
        Fixture fixture = fixture();
        CompletableFuture<PreemptionRequestResult> terminal = new CompletableFuture<>();
        PreemptionRegistration claim = claim(fixture.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenReturn(Optional.of(claim));
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(fixture.requests())
                        .applyEvent(
                                eq(claim),
                                argThat(
                                        event ->
                                                event.endpoint() == fixture.endpoint()
                                                        && event.reservation() != null)))
                .thenAnswer(
                        invocation -> {
                            terminal.complete(
                                    new PreemptionRequestResult(
                                            String.valueOf(11L),
                                            PreemptionRequestResult.Outcome.REQUEST_END));
                            return true;
                        });
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(CompletableFuture.completedFuture(reply));
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L)),
                                1_000L,
                                1_000L,
                                () -> true,
                                "test"));
        if (reply == EngineGrpcClient.CancelReply.REQUEST_FENCED) {
            assertFalse(
                    outcome.isDone(), "Prefill fence alone does not prove Decode resource release");
            verify(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                    fixture.requests()),
                            never())
                    .applyEvent(
                            eq(claim),
                            argThat(
                                    event ->
                                            event.endpoint() == fixture.endpoint()
                                                    && event.reservation() != null));
            verify(fixture.endpoint(), never())
                    .updatePreemption(
                            1L,
                            DecodeResources.PreemptionUpdate.fenced(
                                    new DecodeResources.RequestReservation(
                                            9L, String.valueOf(11L), 101L)));
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
            terminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        }
        assertTrue(outcome.get(1L, TimeUnit.SECONDS).committed());
        verify(fixture.endpoint()).commitPreemption(1L);
    }

    @Test
    void laterVictimClaimFailureReleasesEarlierClaimBeforeAnyCancel() throws Exception {
        Fixture fixture = fixture();
        PreemptionRegistration first = claim(fixture.requests(), 11L, new CompletableFuture<>());
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenReturn(Optional.of(first), Optional.empty());
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(
                                                requestToPreempt(11L, 101L),
                                                requestToPreempt(12L, 102L)),
                                        1000L,
                                        1000L,
                                        () -> true,
                                        "test"))
                        .get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        verify(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(fixture.requests()))
                .applyEvent(
                        first,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        verify(fixture.endpoint(), never())
                .beginPreemption(
                        anyLong(),
                        anyList(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    void invalidCancelTargetReleasesEarlierClaimAndReportsControlFailureWithoutRpc()
            throws Exception {
        Fixture fixture = fixture();
        PreemptionRegistration first = claim(fixture.requests(), 11L, new CompletableFuture<>());
        when(registrations.tryRegisterPreemption(
                        any(DecodeResources.RequestReservation.class), anyLong()))
                .thenReturn(Optional.of(first))
                .thenThrow(
                        new IllegalStateException(
                                "Priority requestToPreempt has no routable Cancel target"));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(
                                                requestToPreempt(11L, 101L),
                                                requestToPreempt(12L, 102L)),
                                        1000L,
                                        1000L,
                                        () -> true,
                                        "test"))
                        .get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertTrue(result.controlFailure());
        verify(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(fixture.requests()))
                .applyEvent(
                        first,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
        verify(fixture.endpoint(), never())
                .beginPreemption(
                        anyLong(),
                        anyList(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    void staleEndpointGenerationCannotCancelANewerRouteWithTheSameRequestAndToken()
            throws Exception {
        Fixture fixture = fixture();
        var stale = new DecodeResources.RequestReservation(9L, String.valueOf(11L), 101L);
        when(registrations.tryRegisterPreemption(eq(stale), anyLong()))
                .thenReturn(Optional.empty());
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        1000L,
                                        1000L,
                                        () -> true,
                                        "old endpoint"))
                        .get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertFalse(result.controlFailure());
        verify(registrations).tryRegisterPreemption(eq(stale), anyLong());
        verify(fixture.endpoint(), never())
                .beginPreemption(
                        anyLong(),
                        anyList(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any());
        org.mockito.Mockito.verifyNoInteractions(channel);
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void blockedPreemptionSettlementDoesNotDelayOtherCleanupDeadlines() throws Exception {
        var entered = new java.util.concurrent.CountDownLatch(2);
        var release = new java.util.concurrent.CountDownLatch(1);
        var deadlineRan = new java.util.concurrent.CountDownLatch(1);
        var settlementThreads = new java.util.concurrent.CopyOnWriteArrayList<String>();
        var outcomes =
                new java.util.ArrayList<CompletableFuture<DecodePreemption.PreemptionResult>>();
        try {
            for (int i = 0; i < 2; i++) {
                Fixture fixture = fixture();
                var terminal = new CompletableFuture<PreemptionRequestResult>();
                var preemptionRegistration = claim(fixture.requests(), 11L, terminal);
                when(registrations.tryRegisterPreemption(any(), anyLong()))
                        .thenReturn(Optional.of(preemptionRegistration));
                org.mockito.Mockito.doAnswer(
                                call -> {
                                    settlementThreads.add(Thread.currentThread().getName());
                                    entered.countDown();
                                    if (!release.await(10, TimeUnit.SECONDS)) {
                                        throw new AssertionError("settlement not released");
                                    }
                                    return null;
                                })
                        .when(fixture.endpoint())
                        .abortPreemption(anyLong());
                var channel = mock(EngineGrpcClient.class);
                when(channel.cancelAsync(
                                any(),
                                anyInt(),
                                argThat(
                                        request ->
                                                request.getReason()
                                                        == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                                anyLong()))
                        .thenReturn(
                                CompletableFuture.completedFuture(
                                        EngineGrpcClient.CancelReply.ACCEPTED));
                var acquirer =
                        newPreemption(
                                dispatcher(channel),
                                SchedulerTestSupport.repository(fixture.requests()),
                                runtime.tasks(),
                                runtime.taskTimer(),
                                mock(RequestSchedulerReporter.class),
                                SchedulingTestConfig.configService(
                                        SchedulingTestConfig.newConfig()),
                                new DecodeResources.AdmissionCapacity(1L, 100L),
                                new PriorityDecodePreemptionPolicy());
                outcomes.add(
                        preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        50L,
                                        20L,
                                        () -> true,
                                        "timer isolation")));
            }
            assertTrue(
                    entered.await(5, TimeUnit.SECONDS),
                    "both settlements must be blocked before probing the timer");
            org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime)
                    .schedule(deadlineRan::countDown, 1, TimeUnit.MILLISECONDS);
            assertTrue(
                    deadlineRan.await(1, TimeUnit.SECONDS),
                    "blocked settlement must not occupy the shared deadline threads");
            assertTrue(
                    settlementThreads.stream().allMatch(name -> name.startsWith("request-work-")),
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
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(fixture.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var deadline = mock(java.util.concurrent.Future.class);
        var protocolRuntime =
                mock(
                        TaskTimer.class,
                        org.mockito.AdditionalAnswers.delegatesTo(runtime.taskTimer()));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            call.<Runnable>getArgument(0).run();
                            terminal.complete(
                                    new PreemptionRequestResult(
                                            String.valueOf(11L),
                                            PreemptionRequestResult.Outcome.REQUEST_END));
                            return deadline;
                        })
                .when(protocolRuntime)
                .schedule(any(Runnable.class), anyLong());
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(fixture.requests()),
                        runtime.tasks(),
                        protocolRuntime,
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());

        var outcome =
                preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        50L,
                                        1000L,
                                        () -> true,
                                        "schedule return race"))
                        .get(1L, TimeUnit.SECONDS);

        assertFalse(
                outcome.committed(),
                "the deadline must freeze rejection before late request resolution");
        assertTrue(
                terminal.isDone(),
                "late request resolution remains available to its resource owner");
        verify(fixture.endpoint(), never()).commitPreemption(anyLong());
        verify(fixture.endpoint()).abortPreemption(1L);
        verify(deadline).cancel(false);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,true", "true,true", "true,false"})
    @org.junit.jupiter.api.Timeout(15)
    void settlementHasAnExplicitDependencyAndShutdownDrainsIt(
            boolean terminalWins, boolean admissionOpenAtSettlement) throws Exception {
        Fixture fixture = fixture();
        var context =
                mock(
                        org.flexlb.balance.scheduler.RequestContext.class,
                        org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getRequestScheduler()).thenReturn(fixture.requests());
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(fixture.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(
                        any(),
                        anyInt(),
                        argThat(
                                request ->
                                        request.getReason()
                                                == REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                        anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var release = new java.util.concurrent.CountDownLatch(1);
        var settlementQueued = new java.util.concurrent.CountDownLatch(1);
        var afterSettlement = new CompletableFuture<Boolean>();
        var shutdown = new CompletableFuture<Void>();
        var admissionOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
        var submissions = new java.util.concurrent.atomic.AtomicInteger();
        Thread closer = null;
        try {
            var tasks =
                    new RequestTaskExecutor() {
                        @Override
                        public void execute(
                                org.flexlb.balance.scheduler.RequestContext exact, Runnable work) {
                            boolean settlement = submissions.incrementAndGet() > 1;
                            runtime.tasks()
                                    .execute(
                                            exact,
                                            () -> {
                                                // Let ACK coordination run; pause the subsequent
                                                // resource-settlement continuation.
                                                if (!settlement) {
                                                    work.run();
                                                    return;
                                                }
                                                settlementQueued.countDown();
                                                try {
                                                    assertTrue(release.await(10, TimeUnit.SECONDS));
                                                } catch (InterruptedException failure) {
                                                    throw new AssertionError(failure);
                                                }
                                                work.run();
                                            });
                        }
                    };
            var acquirer =
                    newPreemption(
                            dispatcher(channel),
                            SchedulerTestSupport.repository(fixture.requests()),
                            tasks,
                            runtime.taskTimer(),
                            mock(RequestSchedulerReporter.class),
                            SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                            new DecodeResources.AdmissionCapacity(1L, 100L),
                            new PriorityDecodePreemptionPolicy());
            var outcome =
                    acquirer.preempt(
                            context,
                            new DefaultDecodePreemption.PreemptionCommand(
                                    fixture.endpoint(),
                                    incoming(),
                                    org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                    List.of(requestToPreempt(11L, 101L)),
                                    50L,
                                    terminalWins ? 5_000L : 20L,
                                    admissionOpen::get,
                                    "explicit settlement dependency"));
            verify(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                    fixture.requests()),
                            org.mockito.Mockito.timeout(2000))
                    .applyEvent(
                            preemptionRegistration,
                            phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
            if (terminalWins) {
                terminal.complete(
                        new PreemptionRequestResult(
                                String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            }
            assertTrue(settlementQueued.await(5, TimeUnit.SECONDS));
            assertFalse(outcome.isDone(), "settlement is paused before endpoint commit");
            verify(fixture.endpoint(), never()).commitPreemption(anyLong());
            verify(fixture.endpoint(), never()).abortPreemption(anyLong());
            if (!terminalWins) {
                assertFalse(
                        terminal.isDone(), "timeout retains the exact late resolution observer");
                terminal.complete(
                        new PreemptionRequestResult(
                                String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            }
            outcome.thenRun(() -> afterSettlement.complete(true));
            var independent = new CompletableFuture<Void>();
            runtime.tasks().execute(context, () -> independent.complete(null));
            independent.get(5, TimeUnit.SECONDS);
            closer =
                    new Thread(
                            () -> {
                                try {
                                    runtime.shutdown(
                                            org.flexlb.balance.scheduler.ShutdownMode.DRAIN,
                                            Duration.ofSeconds(30L));
                                    shutdown.complete(null);
                                } catch (Throwable failure) {
                                    shutdown.completeExceptionally(failure);
                                }
                            });
            closer.start();
            assertThrows(TimeoutException.class, () -> shutdown.get(100, TimeUnit.MILLISECONDS));
            admissionOpen.set(admissionOpenAtSettlement);
            release.countDown();
            assertEquals(
                    terminalWins && admissionOpenAtSettlement,
                    outcome.get(5, TimeUnit.SECONDS).committed());
            assertTrue(
                    afterSettlement.get(5, TimeUnit.SECONDS),
                    "dependent work follows the settlement future");
            shutdown.get(5, TimeUnit.SECONDS);
            assertTrue(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.cleanupExecutor(runtime)
                            .isTerminated());
            if (!terminalWins || !admissionOpenAtSettlement) {
                verify(fixture.endpoint(), never()).commitPreemption(anyLong());
                verify(fixture.endpoint()).abortPreemption(1L);
            }
        } finally {
            release.countDown();
            if (closer != null) {
                closer.join(5_000);
            }
        }
    }

    @Test
    void cancellationReceiptDoesNotHopThroughRequestWorkers() throws Exception {
        var fixture = fixture();
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(fixture.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var threadName = new CompletableFuture<String>();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            threadName.complete(Thread.currentThread().getName());
                            return true;
                        })
                .when(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                fixture.requests()))
                .applyEvent(
                        preemptionRegistration, phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(fixture.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                fixture.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L)),
                                500L,
                                1000L,
                                () -> true,
                                "executor audit"));
        verify(channel, org.mockito.Mockito.timeout(2000))
                .cancelAsync(any(), anyInt(), any(), anyLong());
        ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        String observed = threadName.get(2, TimeUnit.SECONDS);
        assertTrue(
                observed.equals(Thread.currentThread().getName())
                        || observed.startsWith("delivery-stop-"),
                "apply directly on completion, including a reply already done at registration");
        terminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        outcome.get(2, TimeUnit.SECONDS);
    }

    @Test
    void slowAckCoordinationDoesNotPreventAnUnrelatedCancel() throws Exception {
        var release = new java.util.concurrent.CountDownLatch(1);
        int size = 8;
        var entered = new java.util.concurrent.CountDownLatch(size);
        var outcomes = new java.util.ArrayList<CompletableFuture<?>>();
        var terminals = new java.util.ArrayList<CompletableFuture<PreemptionRequestResult>>();
        try {
            for (int i = 0; i < size; i++) {
                var fixture = fixture();
                var terminal = new CompletableFuture<PreemptionRequestResult>();
                terminals.add(terminal);
                var preemptionRegistration = claim(fixture.requests(), 11L, terminal);
                when(registrations.tryRegisterPreemption(any(), anyLong()))
                        .thenReturn(Optional.of(preemptionRegistration));
                org.mockito.Mockito.doAnswer(
                                call -> {
                                    entered.countDown();
                                    if (!release.await(5, TimeUnit.SECONDS)) {
                                        throw new IllegalStateException("test release deadline");
                                    }
                                    return true;
                                })
                        .when(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                        fixture.requests()))
                        .applyEvent(
                                preemptionRegistration,
                                phaseEvent(PreemptionCancelPhase.CANCEL_REQUESTED));
                var channel = mock(EngineGrpcClient.class);
                var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
                when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
                var acquirer =
                        newPreemption(
                                dispatcher(channel),
                                SchedulerTestSupport.repository(fixture.requests()),
                                runtime.tasks(),
                                runtime.taskTimer(),
                                mock(RequestSchedulerReporter.class),
                                SchedulingTestConfig.configService(
                                        SchedulingTestConfig.newConfig()),
                                new DecodeResources.AdmissionCapacity(1L, 100L),
                                new PriorityDecodePreemptionPolicy());
                outcomes.add(
                        preempt(
                                acquirer,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        fixture.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        500L,
                                        1000L,
                                        () -> true,
                                        "slow coordination")));
                verify(channel, org.mockito.Mockito.timeout(1000))
                        .cancelAsync(any(), anyInt(), any(), anyLong());
                Thread.ofVirtual().start(() -> ack.complete(EngineGrpcClient.CancelReply.ACCEPTED));
            }
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var nextClient = mock(EngineGrpcClient.class);
            var nextSent = new java.util.concurrent.CountDownLatch(1);
            when(nextClient.cancelAsync(any(), anyInt(), any(), anyLong()))
                    .thenAnswer(
                            call -> {
                                nextSent.countDown();
                                return CompletableFuture.completedFuture(
                                        EngineGrpcClient.CancelReply.REQUEST_CLEANED);
                            });
            var nextRequestToPreempt = fixture();
            var nextClaim = claim(nextRequestToPreempt.requests(), 12L, new CompletableFuture<>());
            var next =
                    SchedulerTestSupport.preemption(nextClaim.owner().data().getRequestScheduler())
                            .cancelRequest(
                                    dispatcher(nextClient),
                                    runtime.taskTimer(),
                                    nextClaim,
                                    nextRequestToPreempt.endpoint(),
                                    "priority preemption",
                                    500L);
            assertTrue(
                    nextSent.await(1, TimeUnit.SECONDS),
                    "slow coordinators must not occupy stop workers");
            assertEquals(PreemptionRegistration.Reply.WAITING, next.get(1, TimeUnit.SECONDS));
            release.countDown();
        } finally {
            release.countDown();
            for (var terminal : terminals) {
                terminal.complete(
                        new PreemptionRequestResult(
                                String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            }
            CompletableFuture.allOf(outcomes.toArray(CompletableFuture[]::new))
                    .get(3, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void returnedClaimMustMatchTheExactVictimReservation(boolean differentGeneration)
            throws Exception {
        var f = fixture();
        var original = claim(f.requests(), 11L, new CompletableFuture<>());
        var stale =
                new PreemptionRegistration(
                        original.owner(),
                        1L,
                        new DecodeResources.RequestReservation(
                                differentGeneration ? 10L : 9L,
                                String.valueOf(11L),
                                differentGeneration ? 101L : 999L),
                        original.cancelTarget());
        when(registrations.tryRegisterPreemption(any(), anyLong())).thenReturn(Optional.of(stale));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var preemption =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(f.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var result =
                preempt(
                                preemption,
                                new DefaultDecodePreemption.PreemptionCommand(
                                        f.endpoint(),
                                        incoming(),
                                        org.flexlb.dao.SchedulingMetadata.explicit(
                                                70, Long.MAX_VALUE),
                                        List.of(requestToPreempt(11L, 101L)),
                                        50L,
                                        50L,
                                        () -> true,
                                        "exact requestToPreempt"))
                        .get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertTrue(result.detail().contains("mismatched_claim"));
        verify(f.endpoint(), never())
                .beginPreemption(
                        anyLong(),
                        anyList(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any());
        verify(channel, never()).cancelAsync(any(), anyInt(), any(), anyLong());
        verify(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(f.requests()))
                .applyEvent(
                        stale,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
    }

    @Test
    void lateResolutionWhileAckCoordinationIsQueuedMustNotReopenExpiredAttempt() throws Exception {
        var taskExecutor = org.mockito.Mockito.spy(runtime.tasks());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                runtime, "tasks", taskExecutor);
        var f = fixture();
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(f.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(
                        CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.ACCEPTED));
        var workers =
                (java.util.concurrent.ThreadPoolExecutor)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                runtime.tasks(), "requestWorkers");
        var entered = new java.util.concurrent.CountDownLatch(workers.getMaximumPoolSize());
        var queued = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        var context =
                mock(
                        org.flexlb.balance.scheduler.RequestContext.class,
                        org.mockito.Mockito.RETURNS_DEEP_STUBS);
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++)
                runtime.tasks()
                        .execute(
                                context,
                                () -> {
                                    entered.countDown();
                                    try {
                                        if (!release.await(5, TimeUnit.SECONDS))
                                            throw new AssertionError("release timeout");
                                    } catch (InterruptedException e) {
                                        throw new AssertionError(e);
                                    }
                                });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            org.mockito.Mockito.doAnswer(
                            call -> {
                                Object result = call.callRealMethod();
                                queued.countDown();
                                return result;
                            })
                    .when(taskExecutor)
                    .execute(any(), any());
            var acquirer =
                    newPreemption(
                            dispatcher(channel),
                            SchedulerTestSupport.repository(f.requests()),
                            runtime.tasks(),
                            runtime.taskTimer(),
                            mock(RequestSchedulerReporter.class),
                            SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                            new DecodeResources.AdmissionCapacity(1L, 100L),
                            new PriorityDecodePreemptionPolicy());
            var outcome =
                    acquirer.preempt(
                            context,
                            new DefaultDecodePreemption.PreemptionCommand(
                                    f.endpoint(),
                                    incoming(),
                                    org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                    List.of(requestToPreempt(11L, 101L)),
                                    20L,
                                    20L,
                                    () -> true,
                                    "late resolution after ack budget"));
            assertTrue(
                    queued.await(2, TimeUnit.SECONDS),
                    "ACK continuation or timed-out settlement must be queued");
            assertThrows(TimeoutException.class, () -> outcome.get(150, TimeUnit.MILLISECONDS));
            terminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            release.countDown();
            var result = outcome.get(2, TimeUnit.SECONDS);
            System.out.println(
                    "AUDIT late_resolution committed="
                            + result.committed()
                            + " detail="
                            + result.detail());
            assertFalse(
                    result.committed(),
                    "worker queue delay must not extend the 20ms cleanup window past 150ms");
            verify(f.endpoint(), never()).commitPreemption(anyLong());
            verify(f.endpoint()).abortPreemption(1L);
        } finally {
            release.countDown();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "true,true",
        "true,false",
        "false,true",
        "false,false"
    })
    @org.junit.jupiter.api.Timeout(15)
    void queuedCoordinationUsesFrozenDeadlineResult(boolean completedInTime, boolean admissionOpen)
            throws Exception {
        var timing =
                mock(
                        TaskTimer.class,
                        org.mockito.AdditionalAnswers.delegatesTo(runtime.taskTimer()));
        var f = fixture();
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(f.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var workers =
                (java.util.concurrent.ThreadPoolExecutor)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                runtime.tasks(), "requestWorkers");
        var entered = new java.util.concurrent.CountDownLatch(workers.getMaximumPoolSize());
        var release = new java.util.concurrent.CountDownLatch(1);
        var registered = new java.util.concurrent.CountDownLatch(1);
        var timeout = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        var cancellation = mock(java.util.concurrent.Future.class);
        var context =
                mock(
                        org.flexlb.balance.scheduler.RequestContext.class,
                        org.mockito.Mockito.RETURNS_DEEP_STUBS);
        var open = new java.util.concurrent.atomic.AtomicBoolean(true);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            assertTrue(
                                    call.<Long>getArgument(1) > 0L
                                            && call.<Long>getArgument(1) <= 20L);
                            boolean firstDeadline =
                                    timeout.compareAndSet(null, call.getArgument(0));
                            registered.countDown();
                            return firstDeadline
                                    ? cancellation
                                    : mock(java.util.concurrent.Future.class);
                        })
                .when(timing)
                .schedule(any(), anyLong());
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++)
                runtime.tasks()
                        .execute(
                                context,
                                () -> {
                                    entered.countDown();
                                    try {
                                        assertTrue(release.await(5, TimeUnit.SECONDS));
                                    } catch (InterruptedException e) {
                                        throw new AssertionError(e);
                                    }
                                });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var acquirer =
                    newPreemption(
                            dispatcher(channel),
                            SchedulerTestSupport.repository(f.requests()),
                            runtime.tasks(),
                            timing,
                            mock(RequestSchedulerReporter.class),
                            SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                            new DecodeResources.AdmissionCapacity(1L, 100L),
                            new PriorityDecodePreemptionPolicy());
            var outcome =
                    acquirer.preempt(
                            context,
                            new DefaultDecodePreemption.PreemptionCommand(
                                    f.endpoint(),
                                    incoming(),
                                    org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                    List.of(requestToPreempt(11L, 101L)),
                                    1000L,
                                    20L,
                                    open::get,
                                    "frozen ACK deadline"));
            verify(channel, org.mockito.Mockito.timeout(1000))
                    .cancelAsync(any(), anyInt(), any(), anyLong());
            verify(timing, org.mockito.Mockito.times(2)).schedule(any(), anyLong());
            ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
            assertTrue(
                    registered.await(2, TimeUnit.SECONDS),
                    "deadline must register before request workers are available");
            if (!completedInTime) {
                timeout.get().run();
            }
            terminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
            if (completedInTime) {
                timeout.get().run();
            }
            assertFalse(outcome.isDone(), "coordination remains queued after eligibility is fixed");
            open.set(admissionOpen);
            release.countDown();
            assertEquals(
                    completedInTime && admissionOpen, outcome.get(2, TimeUnit.SECONDS).committed());
            if (completedInTime && admissionOpen) {
                verify(f.endpoint()).commitPreemption(1L);
            } else {
                verify(f.endpoint(), never()).commitPreemption(anyLong());
                verify(f.endpoint()).abortPreemption(1L);
            }
            verify(cancellation).cancel(false);
        } finally {
            release.countDown();
        }
    }

    @Test
    void unresolvedNotFoundStillRejectsAsSoonAsTheOtherVictimResolves() throws Exception {
        var f = fixture();
        var firstTerminal = new CompletableFuture<PreemptionRequestResult>();
        var secondTerminal = new CompletableFuture<PreemptionRequestResult>();
        var first = claim(f.requests(), 11L, firstTerminal);
        var second = claim(f.requests(), 12L, secondTerminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenAnswer(
                        call ->
                                Optional.of(
                                        call.<DecodeResources.RequestReservation>getArgument(0)
                                                        .requestId()
                                                        .equals("11")
                                                ? first
                                                : second));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenAnswer(
                        call ->
                                CompletableFuture.completedFuture(
                                        call.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                                                ? EngineGrpcClient.CancelReply.ACCEPTED
                                                : EngineGrpcClient.CancelReply.NOT_FOUND));
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(f.requests()),
                        runtime.tasks(),
                        runtime.taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                f.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L), requestToPreempt(12L, 102L)),
                                1000L,
                                5000L,
                                () -> true,
                                "NOT_FOUND early rejection"));
        verify(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(f.requests()),
                        org.mockito.Mockito.timeout(1000))
                .applyEvent(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
        firstTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        var result = outcome.get(1, TimeUnit.SECONDS);
        assertFalse(result.committed());
        assertFalse(secondTerminal.isDone());
        secondTerminal.complete(
                new PreemptionRequestResult(
                        String.valueOf(12L), PreemptionRequestResult.Outcome.REQUEST_END));
        assertSame(result, outcome.join());
        verify(f.endpoint(), never()).commitPreemption(anyLong());
        verify(f.endpoint()).abortPreemption(1L);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void resourceReleaseUsesTheFrozenDeadlineWhileNotificationIsPaused(boolean expired)
            throws Exception {
        var timing =
                mock(
                        TaskTimer.class,
                        org.mockito.AdditionalAnswers.delegatesTo(runtime.taskTimer()));
        var f = fixture();
        var terminal = new CompletableFuture<PreemptionRequestResult>();
        var preemptionRegistration = claim(f.requests(), 11L, terminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenReturn(Optional.of(preemptionRegistration));
        var channel = mock(EngineGrpcClient.class);
        var ack = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(ack);
        var taskExecutor = org.mockito.Mockito.spy(runtime.tasks());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                runtime, "tasks", taskExecutor);
        var queued = new java.util.concurrent.LinkedBlockingQueue<Runnable>();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            queued.add(call.getArgument(1));
                            return null;
                        })
                .when(taskExecutor)
                .execute(any(), any());
        var timeout = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            timeout.compareAndSet(null, call.getArgument(0));
                            return mock(java.util.concurrent.Future.class);
                        })
                .when(timing)
                .schedule(any(), anyLong());
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(f.requests()),
                        runtime.tasks(),
                        timing,
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(1L, 100L),
                        new PriorityDecodePreemptionPolicy());
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                f.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                List.of(requestToPreempt(11L, 101L)),
                                1000L,
                                2000L,
                                () -> true,
                                "canonical propagation gap"));
        verify(channel, org.mockito.Mockito.timeout(1000))
                .cancelAsync(any(), anyInt(), any(), anyLong());
        ack.complete(EngineGrpcClient.CancelReply.ACCEPTED);
        Runnable coordinator = queued.poll(2, TimeUnit.SECONDS);
        assertTrue(coordinator != null);
        assertTrue(timeout.get() != null);
        if (expired) timeout.get().run();
        // Make the resource fact visible without publishing its listener notification.
        releasedResources.get(f.requests()).add(preemptionRegistration.reservation());
        if (!expired) timeout.get().run();
        coordinator.run();
        while (!outcome.isDone()) {
            var work = queued.poll(2, TimeUnit.SECONDS);
            assertTrue(work != null, "resource wait and admission must both finish");
            work.run();
        }
        assertEquals(
                !expired,
                outcome.get(2, TimeUnit.SECONDS).committed(),
                "a late resource release must not reopen a completed timeout");
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "false,false",
        "true,false",
        "false,true",
        "true,true"
    })
    void finishedAttemptReleasesObserversWithoutCompletingVictimSignals(
            boolean expire, boolean mixed) throws Exception {
        verifyResourceListenerRemoved(expire, mixed);
    }

    private void verifyResourceListenerRemoved(boolean expire, boolean mixed) throws Exception {
        var deadline = new CompletableFuture<Runnable>();
        // A Mockito stubbing retains its last used invocation, even after clearInvocations.
        // Keep the controllable timer free of recorded Runnable arguments for this GC contract.
        var timing =
                new TaskTimer() {
                    @Override
                    public java.util.concurrent.Future<?> schedule(Runnable work, long delayMs) {
                        deadline.complete(work);
                        return new CompletableFuture<Void>() {
                            @Override
                            public boolean cancel(boolean interrupt) {
                                deadline.obtrudeValue(null);
                                return super.cancel(interrupt);
                            }
                        };
                    }
                };
        var f = fixture();
        var firstTerminal = new CompletableFuture<PreemptionRequestResult>();
        var secondTerminal = new CompletableFuture<PreemptionRequestResult>();
        var first = claim(f.requests(), 11L, firstTerminal);
        var second = claim(f.requests(), 12L, secondTerminal);
        when(registrations.tryRegisterPreemption(any(), anyLong()))
                .thenAnswer(
                        call ->
                                Optional.of(
                                        call.<DecodeResources.RequestReservation>getArgument(0)
                                                        .requestId()
                                                        .equals("11")
                                                ? first
                                                : second));
        var channel = mock(EngineGrpcClient.class);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenAnswer(
                        call ->
                                CompletableFuture.completedFuture(
                                        call.<CancelRequestPB>getArgument(2).getRequestId() == 11L
                                                        && (expire || mixed)
                                                ? EngineGrpcClient.CancelReply.ACCEPTED
                                                : EngineGrpcClient.CancelReply.NOT_FOUND));
        var acquirer =
                newPreemption(
                        dispatcher(channel),
                        SchedulerTestSupport.repository(f.requests()),
                        runtime.tasks(),
                        timing,
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()),
                        new DecodeResources.AdmissionCapacity(2L, 100L),
                        new PriorityDecodePreemptionPolicy());
        java.util.function.BooleanSupplier admissionOpen = () -> true;
        var requestsToPreempt =
                mixed
                        ? List.of(requestToPreempt(11L, 101L), requestToPreempt(12L, 102L))
                        : List.of(requestToPreempt(11L, 101L));
        var outcome =
                preempt(
                        acquirer,
                        new DefaultDecodePreemption.PreemptionCommand(
                                f.endpoint(),
                                incoming(),
                                org.flexlb.dao.SchedulingMetadata.explicit(70, Long.MAX_VALUE),
                                requestsToPreempt,
                                1000L,
                                5000L,
                                admissionOpen,
                                "observer lifetime"));
        if (expire) {
            deadline.get(2, TimeUnit.SECONDS).run();
        } else if (mixed) {
            verify(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(
                                    f.requests()),
                            org.mockito.Mockito.timeout(1000))
                    .applyEvent(second, phaseEvent(PreemptionCancelPhase.NOT_FOUND_STALE));
            firstTerminal.complete(
                    new PreemptionRequestResult(
                            String.valueOf(11L), PreemptionRequestResult.Outcome.REQUEST_END));
        }
        assertFalse(outcome.get(2, TimeUnit.SECONDS).committed());
        var unresolved = mixed ? second : first;
        assertFalse(
                unresolved.completion().toCompletableFuture().isDone(),
                "ending the local wait must not manufacture remote cleanup evidence or cancel its"
                        + " signal");
        verify(f.endpoint(), never()).commitPreemption(anyLong());
        assertTrue(
                resourceListeners.get(f.requests()).isEmpty(),
                "ending the wait must remove its endpoint listener even if B never completes");
    }

    private Fixture fixture() {
        AbstractRequestScheduler requests = mock(AbstractRequestScheduler.class);
        org.flexlb.balance.scheduler.SchedulerTestServices.bind(requests, runtime);
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(requests)
                        .applyEvent(any(), any()))
                .thenAnswer(
                        call ->
                                call.getArgument(1) == null
                                        || call.getArgument(
                                                                1,
                                                                org.flexlb.balance.preemption
                                                                        .PreemptionCompletion
                                                                        .PreemptionEvent.class)
                                                        .phase()
                                                != PreemptionCancelPhase.CANCEL_UNKNOWN);
        DecodeEndpoint endpoint =
                org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        var released =
                java.util.concurrent.ConcurrentHashMap
                        .<DecodeResources.RequestReservation>newKeySet();
        var listeners = java.util.concurrent.ConcurrentHashMap.<Runnable>newKeySet();
        releasedResources.put(requests, released);
        resourceListeners.put(requests, listeners);
        when(endpoint.isPreemptionResourceReleased(anyLong(), any()))
                .thenAnswer(call -> released.contains(call.getArgument(1)));
        when(endpoint.watchEngineDispatchCapacity(any()))
                .thenAnswer(
                        call -> {
                            Runnable listener = call.getArgument(0);
                            listeners.add(listener);
                            return (Runnable) () -> listeners.remove(listener);
                        });
        WorkerStatus status = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getGenerationId()).thenReturn(9L);
        when(endpoint.beginPreemption(
                        anyLong(),
                        anyList(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any(DecodeResources.AdmissionCapacity.class)))
                .thenReturn(DecodeResources.PreemptionBeginResult.SUCCESS);
        when(endpoint.updatePreemption(anyLong(), any(DecodeResources.PreemptionUpdate.class)))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        when(endpoint.commitPreemption(anyLong()))
                .thenReturn(new DecodeResources.RequestReservation(9L, String.valueOf(20L), 100L));
        return new Fixture(requests, endpoint);
    }

    private PreemptionRegistration claim(
            AbstractRequestScheduler owner,
            long requestId,
            CompletableFuture<PreemptionRequestResult> terminal) {
        var context =
                mock(
                        org.flexlb.balance.scheduler.RequestContext.class,
                        org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getRequestId()).thenReturn(String.valueOf(requestId));
        when(context.snapshot().delivery()).thenReturn(null);
        when(context.snapshot().cancellation()).thenReturn(null);
        when(context.data().getRequestScheduler()).thenReturn(owner);
        var reservation =
                new DecodeResources.RequestReservation(
                        9L, String.valueOf(requestId), requestId == 11L ? 101L : 102L);
        var claim =
                new PreemptionRegistration(
                        context, 1L, reservation, new CancelTarget("10.0.0.1", 9090));
        terminal.whenComplete(
                (result, failure) -> {
                    if (failure == null) {
                        claim.complete(result);
                        // These protocol fixtures report resource release with request completion.
                        // Independent resource/request ordering is covered by the resource-wait
                        // test.
                        releasedResources.get(owner).add(reservation);
                        resourceListeners.get(owner).forEach(Runnable::run);
                    }
                });
        return claim;
    }

    private DefaultDecodePreemption newPreemption(
            org.flexlb.balance.scheduler.DeliveryCoordinator dispatcher,
            org.flexlb.balance.scheduler.RequestRepository repository,
            org.flexlb.balance.scheduler.RequestTaskExecutor executor,
            org.flexlb.balance.scheduler.TaskTimer timer,
            RequestSchedulerReporter reporter,
            ConfigService config,
            DecodeResources.AdmissionCapacity capacity,
            DecodePreemptionPolicy policy) {
        return new DefaultDecodePreemption(
                dispatcher, repository, executor, timer, reporter, config, capacity, policy) {
            @Override
            public Optional<PreemptionRegistration> tryRegisterPreemption(
                    DecodeResources.RequestReservation reservation, long preemptionId) {
                return registrations.tryRegisterPreemption(reservation, preemptionId);
            }

            @Override
            protected PreemptionCompletion completion() {
                var handler = org.mockito.Mockito.spy(new PreemptionCompletion());
                org.mockito.Mockito.doAnswer(
                                call -> {
                                    PreemptionRegistration attempt = call.getArgument(0);
                                    return SchedulerTestSupport.preemption(
                                                    attempt.owner().data().getRequestScheduler())
                                            .applyEvent(attempt, call.getArgument(1));
                                })
                        .when(handler)
                        .applyEvent(any(), any());
                return handler;
            }
        };
    }

    private org.flexlb.balance.scheduler.DeliveryCoordinator dispatcher(EngineGrpcClient client) {
        org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                runtime.dispatcher, "grpcClient", client);
        return (org.flexlb.balance.scheduler.DeliveryCoordinator)
                org.springframework.test.util.ReflectionTestUtils.getField(runtime, "dispatcher");
    }

    private static DecodeRequestView requestToPreempt(long requestId, long reservationToken) {
        return new DecodeRequestView(
                String.valueOf(requestId),
                30,
                64L,
                64L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true,
                reservationToken,
                false);
    }

    private CompletableFuture<DecodePreemption.PreemptionResult> preempt(
            DefaultDecodePreemption acquirer, DefaultDecodePreemption.PreemptionCommand command) {
        var context =
                mock(
                        org.flexlb.balance.scheduler.RequestContext.class,
                        org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(context.data().getRequestScheduler()).thenReturn(mock(AbstractRequestScheduler.class));
        return acquirer.preempt(context, command);
    }

    private static org.flexlb.dao.loadbalance.Request incoming() {
        return org.flexlb.dao.loadbalance.Request.builder()
                .requestId("20")
                .seqLen(64L)
                .maxNewTokens(0)
                .build();
    }

    private static org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent phaseEvent(
            PreemptionCancelPhase phase) {
        return new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                phase, false, null, null, null);
    }

    private record Fixture(AbstractRequestScheduler requests, DecodeEndpoint endpoint) {}
}
