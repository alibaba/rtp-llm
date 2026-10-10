package org.flexlb.httpserver;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.stream.Stream;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

class FlexlbServiceImplTest {

    private final java.util.concurrent.ScheduledExecutorService deadlineTimer =
            java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
    private RequestScheduler requestScheduler;
    private final AbstractRequestScheduler requestState = mock(AbstractRequestScheduler.class);
    private MasterStatusService masterStatusService;
    private EngineHealthReporter engineHealthReporter;
    private FlexlbGrpcForwarder grpcForwarder;
    private ConfigService configService;
    private DeliveryMetricsReporter deliveryMetricsReporter;
    private ServerScheduleLatencyRecorder serverLatencyRecorder;
    private RequestSchedulerReporter requestSchedulerReporter;
    private FlexlbServiceImpl service;
    private ch.qos.logback.classic.Logger pvLogger;
    private ListAppender<ILoggingEvent> pvAppender;

    @BeforeEach
    void setUp() {
        org.flexlb.telemetry.FlexlbTrace.configure(io.opentelemetry.api.OpenTelemetry.noop(), "");
        requestScheduler = mock(RequestScheduler.class);
        masterStatusService = mock(MasterStatusService.class);
        engineHealthReporter = mock(EngineHealthReporter.class);
        grpcForwarder = mock(FlexlbGrpcForwarder.class);
        deliveryMetricsReporter = mock(DeliveryMetricsReporter.class);
        serverLatencyRecorder = mock(ServerScheduleLatencyRecorder.class);

        configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = org.flexlb.mock.TestFlexlbConfigs.create();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        requestSchedulerReporter = mock(RequestSchedulerReporter.class);
        createService();

        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        pvAppender = new ListAppender<>();
        pvAppender.start();
        pvLogger.addAppender(pvAppender);
    }

    private void createService() {
        service = FlexlbServiceTestSupport.create(requestScheduler,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestState),
                masterStatusService, engineHealthReporter, grpcForwarder, configService,
                deliveryMetricsReporter, serverLatencyRecorder, requestSchedulerReporter);
    }

    @AfterEach
    void tearDown() {
        org.flexlb.telemetry.FlexlbTrace.configure(null, "");
        deadlineTimer.shutdownNow();
        pvLogger.detachAppender(pvAppender);
        pvAppender.stop();
    }

    @Test
    void testSchedule_localRouting() {
        FlexlbConfig requestConfig = org.flexlb.mock.TestFlexlbConfigs.create();
        when(configService.loadBalanceConfig()).thenReturn(requestConfig)
                .thenThrow(new IllegalStateException("configuration must only be read once"));
        org.mockito.Mockito.clearInvocations(configService);
        createService();
        // Given: not master, no consistency needed
        when(masterStatusService.isNeedConsistency()).thenReturn(false);

        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(any(RequestContext.class))).thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .setSeqLen(100)
                .setCacheKeyBlockSize(1024L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
        assertEquals(200, resp.getCode());
        ArgumentCaptor<RequestContext> contextCaptor = ArgumentCaptor.forClass(RequestContext.class);
        verify(requestScheduler).submit(contextCaptor.capture());
        org.junit.jupiter.api.Assertions.assertSame(requestConfig, contextCaptor.getValue().getConfig());
        assertEquals(requestConfig.getScheduler().getType(), contextCaptor.getValue().getConfig().getScheduler().getType());
        verify(configService).loadBalanceConfig();
        assertPvContains("\"scheduleOrigin\":\"LOCAL_STANDALONE\"");
        verify(serverLatencyRecorder).recordArrival(anyLong());
        verify(serverLatencyRecorder).recordCompletion(any(RequestContext.class), anyLong());
    }

    @Test
    void testSchedule_clientCancellationReleasesSchedulerOwnedRequest() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        CompletableFuture<Response> pendingRoute = new CompletableFuture<>();
        when(requestScheduler.submit(any(RequestContext.class))).thenAnswer(call -> {
            org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(call.getArgument(0, RequestContext.class), requestState);
            return pendingRoute;
        });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_356L)
                        .build();
        Context.CancellableContext inbound = Context.current().withCancellation();

        inbound.run(() -> service.schedule(request, observer));
        inbound.cancel(null);

        verify(requestState).onResponseUndeliverable(org.mockito.ArgumentMatchers.argThat(context -> context.getRequestId() == 12_356L));
        verifyNoInteractions(observer);

        Response lateRoute = new Response();
        lateRoute.setSuccess(true);
        lateRoute.setCode(200);
        pendingRoute.complete(lateRoute);

        verifyNoInteractions(observer);
    }

    @Test
    void testSchedule_alreadyCancelledContextCannotRaceAheadOfRegistration() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        CompletableFuture<Response> pendingRoute = new CompletableFuture<>();
        when(requestScheduler.submit(any(RequestContext.class))).thenAnswer(call -> {
            org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(call.getArgument(0, RequestContext.class), requestState);
            return pendingRoute;
        });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        Context.CancellableContext inbound = Context.current().withCancellation();
        inbound.cancel(null);

        inbound.run(() -> service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_357L)
                        .build(), observer));

        var inOrder = inOrder(requestScheduler, requestState);
        inOrder.verify(requestScheduler).submit(any(RequestContext.class));
        inOrder.verify(requestState).onResponseUndeliverable(org.mockito.ArgumentMatchers.argThat(context -> context.getRequestId() == 12_357L));
        verifyNoInteractions(observer);
    }

    @Test
    void testSchedule_preservesBothEnqueuedByMasterValues() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);

        for (boolean expected : new boolean[]{false, true}) {
            Response response = new Response();
            response.setSuccess(true);
            response.setCode(200);
            response.setEnqueuedByMaster(expected);
            when(requestScheduler.submit(any(RequestContext.class)))
                    .thenReturn(CompletableFuture.completedFuture(response));

            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                    mock(StreamObserver.class);
            service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId(expected ? 12_351L : 12_350L)
                    .build(), observer);

            ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                    ArgumentCaptor.forClass(
                            FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(captor.capture());
            verify(observer).onCompleted();
            assertEquals(expected, captor.getValue().getEnqueuedByMaster());
        }
    }

    @Test
    void testSchedule_serializesTypedAdmissionReason() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        Response response = Response.error(
                StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
        when(requestScheduler.submit(any(RequestContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(54321L)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        assertEquals(8430, captor.getValue().getCode());
        assertEquals(
                FlexlbScheduleProtocol.ScheduleFailureReasonPB.SAME_PRIORITY_AHEAD,
                captor.getValue().getAdmissionRejectReason());
        assertPvContains("\"code\":8430");
        assertPvContains("\"admissionRejectReason\":\"SAME_PRIORITY_AHEAD\"");
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void schedulingDiagnosticsAreLoggedOnlyForFailure(boolean success) throws Exception {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        Response response = success ? new Response() : Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
        if (success) {
            response.setSuccess(true);
            response.setCode(200);
        }
        when(requestScheduler.submit(any(RequestContext.class))).thenAnswer(invocation -> {
            RequestContext ctx = invocation.getArgument(0);
            ctx.setSchedulingDiagnostics(java.util.Map.of(
                    "cause", "decode engine slots exhausted",
                    "prefill", java.util.Map.of("queueDepth", 8, "higherPriorityCount", 3),
                    "decode", java.util.Map.of("engineLoad", 4, "kvAvailable", 512)));
            return CompletableFuture.completedFuture(response);
        });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(54322L).build(), observer);
        assertEquals(1, pvAppender.list.size());
        com.fasterxml.jackson.databind.JsonNode pv = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(pvAppender.list.get(0).getFormattedMessage());
        assertEquals(!success, pv.has("schedulingDiagnostics"));
        assertFalse(pv.path("response").has("schedulingDiagnostics"));
        if (!success) {
            assertEquals("decode engine slots exhausted", pv.at("/schedulingDiagnostics/cause").asText());
            assertEquals(8, pv.at("/schedulingDiagnostics/prefill/queueDepth").asInt());
            assertEquals(4, pv.at("/schedulingDiagnostics/decode/engineLoad").asInt());
            ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                    ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(captor.capture());
            assertEquals(8431, captor.getValue().getCode());
            assertEquals("admission capacity is temporarily exhausted", captor.getValue().getErrorMessage());
        }
    }

    @Test
    void testSchedule_forwardToMaster_success() {
        // Given: consistency needed, not master, forward succeeds
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB masterResponse = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(true)
                .setCode(200)
                .setEnqueuedByMaster(true)
                .build();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                                masterResponse, "10.0.0.2:7001")));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardScheduleToMaster(request);
        verify(requestScheduler, never()).submit(any());

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
        assertTrue(resp.getEnqueuedByMaster());
        assertTrue(pvAppender.list.isEmpty());
    }

    @Test
    void testSchedule_pendingMasterForwardReturnsWithoutHoldingRequestThread() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pendingForward =
                new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pendingForward);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_352L)
                        .build();

        assertTimeoutPreemptively(Duration.ofSeconds(1),
                () -> service.schedule(request, observer));

        verifyNoInteractions(observer);
        verify(requestScheduler, never()).submit(any());

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(true)
                        .setCode(200)
                        .build();
        assertTrue(pendingForward.complete(
                FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                        response, "10.0.0.2:7001")));
        assertFalse(pendingForward.complete(
                FlexlbGrpcForwarder.MasterForwardResult.failed(
                        "UNAVAILABLE", "10.0.0.2:7001")));

        verify(observer, times(1)).onNext(response);
        verify(observer, times(1)).onCompleted();
        verify(requestScheduler, never()).submit(any());
    }

    @Test
    @SuppressWarnings("unchecked")
    void testSchedule_callbackRegistrationFailureStillCompletesExactlyOnce() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> unusualStage =
                mock(CompletionStage.class);
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(true)
                        .setCode(200)
                        .build();
        FlexlbGrpcForwarder.MasterForwardResult forwardResult =
                FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                        response, "10.0.0.2:7001");
        when(unusualStage.whenComplete(any())).thenAnswer(invocation -> {
            BiConsumer<FlexlbGrpcForwarder.MasterForwardResult, Throwable> callback =
                    invocation.getArgument(0);
            callback.accept(forwardResult, null);
            throw new IllegalStateException("registration failed after callback");
        });
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(unusualStage);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12_354L)
                .build(), observer);

        verify(observer, times(1)).onNext(response);
        verify(observer, times(1)).onCompleted();
        verify(requestScheduler, never()).submit(any());
    }

    @Test
    void testSchedule_exceptionalMasterCompletionIsTerminal() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pendingForward =
                new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pendingForward);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12_353L)
                .build(), observer);
        pendingForward.completeExceptionally(Status.UNAVAILABLE.asRuntimeException());

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> response =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer, times(1)).onNext(response.capture());
        verify(observer, times(1)).onCompleted();
        assertFalse(response.getValue().getSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                response.getValue().getCode());
        verify(requestScheduler, never()).submit(any());
    }

    @Test
    void testSchedule_forwardObserverFailureDoesNotSendSecondResponse() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB masterResponse =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(true)
                        .setCode(200)
                        .build();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                                masterResponse, "10.0.0.2:7001")));
        when(grpcForwarder.forwardCompensatingCancelToMaster(any(), any(), any(io.opentelemetry.context.Context.class)))
                .thenAnswer(invocation -> {
                    assertFalse(Context.current().isCancelled());
                    return CompletableFuture.completedFuture(
                            FlexlbGrpcForwarder.CancelForwardResult.forwarded(
                                    FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                                            .setFound(true)
                                            .build(),
                                    "10.0.0.2:7001"));
                });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        doThrow(new RuntimeException("client disconnected"))
                .when(observer).onNext(any());
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_346L)
                        .build();

        var parent = io.opentelemetry.api.trace.SpanContext.createFromRemoteParent(
                "11111111111111111111111111111111", "2222222222222222",
                io.opentelemetry.api.trace.TraceFlags.getSampled(),
                io.opentelemetry.api.trace.TraceState.getDefault());
        var traceContext = io.opentelemetry.context.Context.root().with(io.opentelemetry.api.trace.Span.wrap(parent));
        Context.CancellableContext inbound = Context.current()
                .withValue(org.flexlb.interceptor.GrpcTraceInterceptor.OTEL_CONTEXT_KEY, traceContext)
                .withCancellation();
        inbound.cancel(null);
        inbound.run(() -> service.schedule(request, observer));

        verify(grpcForwarder, times(1)).forwardScheduleToMaster(any());
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelRequestPB> cancel =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelRequestPB.class);
        verify(grpcForwarder).forwardCompensatingCancelToMaster(
                cancel.capture(), org.mockito.ArgumentMatchers.eq("10.0.0.2:7001"),
                org.mockito.ArgumentMatchers.argThat(context -> {
                    var actual = io.opentelemetry.api.trace.Span.fromContext(context).getSpanContext();
                    return parent.getTraceId().equals(actual.getTraceId())
                            && parent.getSpanId().equals(actual.getSpanId());
                }));
        assertEquals(request.getRequestId(), cancel.getValue().getRequestId());
        assertEquals(
                FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED,
                cancel.getValue().getReason());
        verify(requestScheduler, never()).submit(any());
        verify(observer, times(1)).onNext(any());
        verify(observer, never()).onCompleted();
    }

    @Test
    void testSchedule_masterNotFoundRoutesLocallyAsFallback() {
        // No Master address was selected, so no RPC was attempted.
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.noMaster()));

        Response localResponse = new Response();
        localResponse.setSuccess(true);
        localResponse.setCode(200);
        when(requestScheduler.submit(any(RequestContext.class))).thenReturn(CompletableFuture.completedFuture(localResponse));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardScheduleToMaster(request);
        verify(requestScheduler).submit(any(RequestContext.class));

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
        assertPvContains("\"scheduleOrigin\":\"LOCAL_FALLBACK\"");
    }

    static Stream<Arguments> forwardFailureCases() {
        String host = "10.0.0.2:7001";
        return Stream.of(
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                Status.DEADLINE_EXCEEDED.asRuntimeException(), host)), 8511),
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                new CompletionException(new TimeoutException()), host)), 8511),
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                new TimeoutException() {}, host)), 8511),
                Arguments.of(
                        CompletableFuture.failedFuture(new ExecutionException(
                                new TimeoutException())), 8511),
                Arguments.of(
                        CompletableFuture.failedFuture(Status.DEADLINE_EXCEEDED.asRuntimeException()), 8511),
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                Status.UNAVAILABLE.asRuntimeException(), host)), 8511),
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                new IllegalStateException("DEADLINE_EXCEEDED"), host)), 8511),
                Arguments.of(
                        CompletableFuture.completedFuture(FlexlbGrpcForwarder.MasterForwardResult.failed(
                                "DEADLINE_EXCEEDED", host)), 8511));
    }

    @ParameterizedTest
    @MethodSource("forwardFailureCases")
    void unknownForwardOutcomeMustNotPermitFrontendReplay(
            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> result, int expectedCode) {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(result);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(90001).build(), observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> response =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(response.capture());
        assertEquals(expectedCode, response.getValue().getCode());
        assertFalse(response.getValue().getSuccess());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
        verifyNoInteractions(requestScheduler);
    }

    @Test
    void testSchedule_forwardFailureIsTerminalAndNeverRoutesLocally() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pendingForward =
                new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pendingForward);

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12348L)
                        .setGenerateTimeout(12_345L)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        Context.CancellableContext inbound = Context.current().withCancellation();
        inbound.run(() -> service.schedule(request, observer));
        inbound.cancel(null);
        inbound.run(() -> pendingForward.complete(
                FlexlbGrpcForwarder.MasterForwardResult.failed(
                        Status.CANCELLED.asRuntimeException(), "10.0.0.2:7001")));

        verify(requestScheduler, never()).submit(any());
        verify(grpcForwarder).forwardScheduleToMaster(request);
        verifyNoMoreInteractions(grpcForwarder);
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        assertFalse(captor.getValue().getSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                captor.getValue().getCode());
        assertPvContains("\"code\":8504");
        assertPvContains("\"scheduleOrigin\":\"FORWARD_FAILED\"");
        assertPvContains("\"requestExpiresAtMs\":");
        assertPvContains("\"realMasterHost\":\"10.0.0.2:7001\"");
    }

    @Test
    void testSchedule_unsentSelfTargetRoutesLocallyWithoutCancel() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.blocked(
                                "SELF_FORWARD_BLOCKED", "10.0.0.2:7001")));
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(response));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12_349L)
                .build(), mock(StreamObserver.class));

        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(requestScheduler, times(1)).submit(any());
    }

    @Test
    void rejectedForwardRetriesOnceAndRetainsOriginalDeadline() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pending = new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pending);
        Response success = new Response();
        success.setSuccess(true);
        success.setCode(200);
        when(requestScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(success));
        var inbound = Context.current().withDeadlineAfter(5, java.util.concurrent.TimeUnit.SECONDS,
                deadlineTimer);
        when(requestScheduler.submit(any())).thenAnswer(invocation -> {
            assertSame(inbound.getDeadline(), Context.current().getDeadline());
            return CompletableFuture.completedFuture(success);
        });
        try {
            inbound.run(() -> service.schedule(
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(81).build(),
                    mock(StreamObserver.class)));
            pending.complete(FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                    nodeRejected(), "old:7001")); // Complete outside the original Context.
            ArgumentCaptor<RequestContext> ctx = ArgumentCaptor.forClass(RequestContext.class);
            verify(requestScheduler).submit(ctx.capture());
            assertEquals(81, ctx.getValue().getRequestId());
            verify(grpcForwarder, times(1)).forwardScheduleToMaster(any());
            verify(grpcForwarder, never()).forwardCancelToMaster(any());
        } finally {
            inbound.cancel(null);
        }
    }

    @Test
    void cancelledCallerCannotStartLocalRetryFromACompletionOnAnotherThread() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pending = new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pending);
        var inbound = Context.current().withCancellation();
        inbound.run(() -> service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(82).build(),
                mock(StreamObserver.class)));
        inbound.cancel(null);
        pending.complete(FlexlbGrpcForwarder.MasterForwardResult.forwarded(nodeRejected(), "old:7001"));
        verify(requestScheduler, never()).submit(any());
    }

    @Test
    void ambiguousForwardFailureReturnsImmediatelyWithoutCancelOrLocalSchedule() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        for (var status : java.util.List.of(Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED)) {
            when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(CompletableFuture.completedFuture(
                    FlexlbGrpcForwarder.MasterForwardResult.failed(status.asRuntimeException(), "old:7001")));
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
            assertTimeoutPreemptively(Duration.ofSeconds(1), () -> service.schedule(
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(83).build(), observer));
            ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> response =
                    ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(response.capture());
            verify(observer).onCompleted();
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getValue().getCode());
            assertEquals("old:7001", response.getValue().getRealMasterHost());
        }
        verify(requestScheduler, never()).submit(any());
        verify(grpcForwarder, times(2)).forwardScheduleToMaster(any());
        verifyNoMoreInteractions(grpcForwarder);
    }

    @Test
    @SuppressWarnings("unchecked")
    void forwardCompletionDispatchesLocalRequestOnlyOnce() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> stage = new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(stage);
        CompletableFuture<Response> local = new CompletableFuture<>();
        when(requestScheduler.submit(any())).thenReturn(local);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(91).build(), observer);

        var rejection = FlexlbGrpcForwarder.MasterForwardResult.forwarded(nodeRejected(), "old:7001");
        assertTrue(stage.complete(rejection));
        assertFalse(stage.complete(rejection));
        verify(grpcForwarder, times(1)).forwardScheduleToMaster(any());
        verify(requestScheduler, times(1)).submit(any());
        verifyNoInteractions(observer);
        Response success = new Response();
        success.setSuccess(true);
        success.setCode(200);
        local.complete(success);
        verify(observer, times(1)).onNext(any());
        verify(observer, times(1)).onCompleted();
        verify(observer, never()).onError(any());
    }

    @Test
    void formerMasterDoesNotClaimAnOwnedRequestWasNeverAccepted() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestState).getRequestState(88L, 0L)).thenReturn(new RequestState(88L,
                RequestState.Phase.ACKNOWLEDGED, DeliveryClaimKind.BATCH_ENQUEUE,
                1001L, 10L, 20L, "already dispatched"));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(88).setForwardHop(1).build(), observer);
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> result =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(result.capture());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), result.getValue().getCode());
        assertTrue(result.getValue().hasLifecycle());
        verify(requestScheduler, never()).submit(any());
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void followerQueriesAndCancelsItsLocalOwnerBeforeConsultingTheLeader() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        RequestState owned = new RequestState(86L, RequestState.Phase.ACKNOWLEDGED,
                DeliveryClaimKind.BATCH_ENQUEUE, 1001L, 10L, 20L, "owned here");
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestState).getRequestState(86L, 1001L)).thenReturn(owned);
        when(requestScheduler.cancel(86L, 1001L, CancelReason.CLIENT_CANCELLED)).thenReturn(owned);
        service.getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setRequestId(86).setBatchId(1001).build(), mock(StreamObserver.class));
        service.cancel(FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(86).setBatchId(1001).build(), mock(StreamObserver.class));
        verify(requestScheduler).cancel(86L, 1001L, CancelReason.CLIENT_CANCELLED);
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void expiredInboundDeadlinePreventsLocalFallbackEvenWhenNoMasterExists() {
        when(masterStatusService.isNeedConsistency()).thenReturn(true);
        when(masterStatusService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(CompletableFuture.completedFuture(
                FlexlbGrpcForwarder.MasterForwardResult.noMaster()));
        var inbound = Context.current().withDeadlineAfter(-1, java.util.concurrent.TimeUnit.SECONDS, deadlineTimer);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        try {
            inbound.run(() -> service.schedule(
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(87).build(), observer));
            verify(requestScheduler, never()).submit(any());
            ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> result =
                    ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(result.capture());
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), result.getValue().getCode());
        } finally {
            inbound.cancel(null);
        }
    }

    @Test
    void localRetryRequiresAnExplicitUnacceptedNodeRejection() {
        RequestContext context = mock(RequestContext.class);
        var result = FlexlbGrpcForwarder.MasterForwardResult.forwarded(nodeRejected(), "old:7001");
        when(context.requestExpired(anyLong())).thenReturn(false);
        assertTrue(service.shouldScheduleLocally(context, result));
        when(context.requestExpired(anyLong())).thenReturn(true);
        assertFalse(service.shouldScheduleLocally(context, result));
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void businessErrorsAndContradictoryRejectionsCannotBeRetriedLocally() {
        RequestContext context = mock(RequestContext.class);
        for (var error : StrategyErrorType.values()) {
            if (error == StrategyErrorType.NOT_MASTER) {
                continue;
            }
            var response = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder().setCode(error.getErrorCode()).build();
            assertFalse(service.shouldScheduleLocally(context,
                    FlexlbGrpcForwarder.MasterForwardResult.forwarded(response, "old:7001")), error.name());
        }
        var rejected = nodeRejected().toBuilder();
        for (var response : java.util.List.of(
                rejected.clone().setSuccess(true).build(),
                rejected.clone().setCode(200).build(),
                rejected.clone().setEnqueuedByMaster(true).build(),
                rejected.clone().setLifecycle(FlexlbScheduleProtocol.RequestLifecyclePB.getDefaultInstance()).build(),
                rejected.clone().addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.getDefaultInstance()).build(),
                rejected.clone().setCode(999).build())) {
            assertFalse(service.shouldScheduleLocally(context,
                    FlexlbGrpcForwarder.MasterForwardResult.forwarded(response, "old:7001")));
        }
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void grpcStatusAloneNeverPermitsLocalReplay() {
        RequestContext context = mock(RequestContext.class);
        for (var code : Status.Code.values()) {
            var failure = FlexlbGrpcForwarder.MasterForwardResult.failed(
                    Status.fromCode(code).asRuntimeException(), "old:7001");
            assertFalse(service.shouldScheduleLocally(context, failure), code.name());
        }
        assertFalse(service.shouldScheduleLocally(context, null));
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void connectionEstablishmentErrorsPermitLocalRoutingWithoutCancel() {
        RequestContext context = mock(RequestContext.class);
        for (var cause : java.util.List.of(
                new java.net.ConnectException("connection refused"),
                new io.netty.channel.ConnectTimeoutException("connect timeout"),
                new java.net.UnknownHostException("master.invalid"),
                new java.nio.channels.UnresolvedAddressException())) {
            var failure = FlexlbGrpcForwarder.MasterForwardResult.failed(
                    Status.UNAVAILABLE.withCause(new RuntimeException(cause)).asRuntimeException(), "old:7001");
            when(context.requestExpired(anyLong())).thenReturn(false);
            assertTrue(service.shouldScheduleLocally(context, failure));
            when(context.requestExpired(anyLong())).thenReturn(true);
            assertFalse(service.shouldScheduleLocally(context, failure));
        }
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void resolutionFailuresDoNotOverrideCallerCancellationOrDeadline() {
        RequestContext context = mock(RequestContext.class);
        for (var cause : java.util.List.of(
                new java.net.UnknownHostException("master.invalid"),
                new java.nio.channels.UnresolvedAddressException())) {
            for (var status : java.util.List.of(Status.CANCELLED, Status.DEADLINE_EXCEEDED)) {
                var failure = FlexlbGrpcForwarder.MasterForwardResult.failed(
                        status.withCause(cause).asRuntimeException(), "old:7001");
                assertFalse(service.shouldScheduleLocally(context, failure));
            }
            var failure = FlexlbGrpcForwarder.MasterForwardResult.failed(
                    Status.UNAVAILABLE.withCause(cause).asRuntimeException(), "old:7001");
            try (var inbound = Context.current().withCancellation()) {
                inbound.cancel(null);
                inbound.run(() -> assertFalse(service.shouldScheduleLocally(context, failure)));
            }
        }
        verifyNoInteractions(grpcForwarder);
    }

    @Test
    void connectionResetAndRemoteUnavailableCannotMasqueradeAsConnectFailure() {
        RequestContext context = mock(RequestContext.class);
        for (var error : java.util.List.of(
                Status.UNAVAILABLE.withCause(new java.net.SocketException("Connection reset")).asRuntimeException(),
                Status.UNAVAILABLE.withDescription("Connection refused").asRuntimeException(),
                Status.UNAVAILABLE.withDescription("Unable to resolve host master.invalid").asRuntimeException(),
                Status.UNAVAILABLE.withCause(new java.net.NoRouteToHostException("No route to host")).asRuntimeException(),
                Status.DEADLINE_EXCEEDED.asRuntimeException())) {
            assertFalse(service.shouldScheduleLocally(context,
                    FlexlbGrpcForwarder.MasterForwardResult.failed(error, "old:7001")));
        }
        verifyNoInteractions(grpcForwarder);
        verify(requestScheduler, never()).submit(any());
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleResponsePB nodeRejected() {
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(StrategyErrorType.NOT_MASTER.getErrorCode()).build();
    }

    @Test
    void testSchedule_exceptionHandling() {
        // Given: route throws exception
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        when(requestScheduler.submit(any(RequestContext.class))).thenReturn(CompletableFuture.failedFuture(new RuntimeException("test error")));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertFalse(resp.getSuccess());
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), resp.getCode());
        assertTrue(resp.getErrorMessage().contains("test error"));
    }

    @Test
    void testSchedule_entryErrorMarksServerSpanFromInterceptorContext() {
        // buildContext() throws before ctx is assigned, so completeSchedule() gets
        // a null RequestContext. The SERVER span must still carry the internal
        // error, recovered from the interceptor's gRPC-scoped context.
        io.opentelemetry.api.GlobalOpenTelemetry.resetForTest();
        RecordingExporter exporter = new RecordingExporter();
        io.opentelemetry.sdk.trace.SdkTracerProvider provider =
                io.opentelemetry.sdk.trace.SdkTracerProvider.builder()
                        .setSampler(io.opentelemetry.sdk.trace.samplers.Sampler.alwaysOn())
                        .addSpanProcessor(
                                io.opentelemetry.sdk.trace.export.SimpleSpanProcessor.create(exporter))
                        .build();
        io.opentelemetry.sdk.OpenTelemetrySdk sdk =
                io.opentelemetry.sdk.OpenTelemetrySdk.builder().setTracerProvider(provider).build();
        io.opentelemetry.api.GlobalOpenTelemetry.set(sdk);
        org.flexlb.telemetry.FlexlbTrace.configure(sdk, "");
        try {
            io.opentelemetry.api.trace.Span serverSpan =
                    org.flexlb.telemetry.FlexlbTrace.startServer(
                            "rtp_llm.flexlb.schedule", io.opentelemetry.context.Context.root());

            doThrow(new IllegalStateException("request reporting failed"))
                    .when(requestSchedulerReporter).reportRequest(org.mockito.ArgumentMatchers.anyInt());

            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                            .setRequestId(778899L)
                            .build();
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                    mock(StreamObserver.class);

            // Publish the SERVER span the way GrpcTraceInterceptor does, then run
            // schedule() inside that gRPC context.
            io.grpc.Context.current()
                    .withValue(org.flexlb.interceptor.GrpcTraceInterceptor.OTEL_CONTEXT_KEY,
                            org.flexlb.telemetry.FlexlbTrace.withSpan(
                                    serverSpan, io.opentelemetry.context.Context.root()))
                    .run(() -> service.schedule(request, observer));

            ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                    ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(captor.capture());
            verify(observer).onCompleted();
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
            assertFalse(resp.getSuccess());
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), resp.getCode());

            // The interceptor owns the span lifecycle; nothing is exported yet.
            assertEquals(0, exporter.spans.size());
            org.flexlb.telemetry.FlexlbTrace.finish(serverSpan, null);

            assertEquals(1, exporter.spans.size());
            io.opentelemetry.sdk.trace.data.SpanData span = exporter.spans.get(0);
            assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR,
                    span.getStatus().getStatusCode());
            assertEquals("FLEXLB_INTERNAL_ERROR",
                    span.getAttributes().get(
                            io.opentelemetry.api.common.AttributeKey.stringKey("error.type")));
            assertEquals((long) StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    span.getAttributes().get(
                            io.opentelemetry.api.common.AttributeKey.longKey("flexlb.schedule.code")));
            assertTrue(span.getEvents().isEmpty());
        } finally {
            org.flexlb.telemetry.FlexlbTrace.configure(null, "");
            sdk.close();
            io.opentelemetry.api.GlobalOpenTelemetry.resetForTest();
        }
    }

    @Test
    void forwardingFailureIsNotMisclassifiedAsAdmissionRejection() {
        RecordingExporter exporter = new RecordingExporter();
        try (var provider = io.opentelemetry.sdk.trace.SdkTracerProvider.builder()
                .addSpanProcessor(io.opentelemetry.sdk.trace.export.SimpleSpanProcessor.create(exporter)).build()) {
            var span = provider.get("test").spanBuilder("schedule").startSpan();
            when(masterStatusService.isNeedConsistency()).thenReturn(true);
            when(masterStatusService.isMaster()).thenReturn(false);
            when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(CompletableFuture.completedFuture(
                    FlexlbGrpcForwarder.MasterForwardResult.failed("FORWARD_HOP_LIMIT", "10.0.0.2:7001")));
            Context.current().withValue(org.flexlb.interceptor.GrpcTraceInterceptor.OTEL_CONTEXT_KEY,
                    io.opentelemetry.context.Context.root().with(span)).run(() -> service.schedule(
                            FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(779900L).build(),
                            mock(StreamObserver.class)));
            org.flexlb.telemetry.FlexlbTrace.finishWithGrpcStatus(span, "OK", 0, true);
            assertEquals(1, exporter.spans.size());
            var data = exporter.spans.get(0);
            assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, data.getStatus().getStatusCode());
            assertEquals("FLEXLB_FORWARD_FAILED", data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.stringKey("error.type")));
            assertEquals((long) StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.longKey("flexlb.schedule.code")));
            assertTrue(data.getEvents().isEmpty());
            verify(requestScheduler, never()).submit(any());
        }
    }

    @Test
    void forwardedWireEndpointsAreRecordedBeforePublicationWithoutInternalResponse() {
        verifyTraceAtResponse(true, false, true, true);
    }

    @Test
    void pdfusionWireResponseDoesNotInventDecodeEndpoint() {
        verifyTraceAtResponse(true, true, true, true);
    }

    @Test
    void localResponseReadsStageTimesBeforeSpanEnds() {
        verifyTraceAtResponse(false, false, true, false);
    }

    @Test
    void workerCompletionWithoutAckDoesNotInventRpcResponseTiming() {
        verifyTraceAtResponse(false, false, true, true);
    }

    @Test
    void failedResponseDoesNotPublishCandidateEndpoints() {
        verifyTraceAtResponse(false, false, false, true);
    }

    private void verifyTraceAtResponse(boolean forwarded, boolean fusion, boolean success, boolean missingAck) {
        RecordingExporter exporter = new RecordingExporter();
        try (var provider = io.opentelemetry.sdk.trace.SdkTracerProvider.builder()
                .addSpanProcessor(io.opentelemetry.sdk.trace.export.SimpleSpanProcessor.create(exporter)).build()) {
            var span = provider.get("test").spanBuilder("schedule")
                    .setSpanKind(io.opentelemetry.api.trace.SpanKind.SERVER).startSpan();
            var traceContext = io.opentelemetry.context.Context.root().with(span);
            var captured = new java.util.concurrent.atomic.AtomicReference<RequestContext>();
            org.mockito.Mockito.doAnswer(inv -> {
                RequestContext ctx = inv.getArgument(0);
                captured.set(ctx);
                long now = System.nanoTime();
                org.springframework.test.util.ReflectionTestUtils.setField(ctx, "serviceStartNanos", now - 40_000_000L);
                ctx.setRouteSubmittedNanos(now - 30_000_000L);
                ctx.setBatchDispatchedNanos(now - 10_000_000L);
                if (!missingAck) {
                    ctx.setAckAtNanos(now - 1_000_000L);
                }
                return null;
            }).when(engineHealthReporter).reportArriveDelayTime(any());
            var wire = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(success).setCode(success ? 200 : 500).setEnqueuedByMaster(true)
                    .addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                            .setRole(fusion ? "PDFUSION" : "PREFILL").setServerIp("10.0.0.10").setHttpPort(8000));
            if (!fusion) {
                wire.addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole("DECODE").setServerIp("10.0.0.20").setHttpPort(9000));
            }
            CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> remote = new CompletableFuture<>();
            CompletableFuture<Response> local = new CompletableFuture<>();
            when(masterStatusService.isNeedConsistency()).thenReturn(forwarded);
            when(masterStatusService.isMaster()).thenReturn(false);
            when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(remote);
            when(requestScheduler.submit(any())).thenReturn(local);
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = new StreamObserver<>() {
                public void onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB value) {
                    org.flexlb.telemetry.FlexlbTrace.finishWithGrpcStatus(span, "OK", 0, true);
                }
                public void onError(Throwable error) { throw new AssertionError(error); }
                public void onCompleted() { }
            };
            Context.current().withValue(org.flexlb.interceptor.GrpcTraceInterceptor.OTEL_CONTEXT_KEY, traceContext)
                    .run(() -> service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                            .setRequestId(991L).build(), observer));
            CompletableFuture.runAsync(() -> {
                if (forwarded) {
                    org.junit.jupiter.api.Assertions.assertNull(captured.get().getResponse());
                    remote.complete(FlexlbGrpcForwarder.MasterForwardResult.forwarded(wire.build(), "10.0.0.2:7001"));
                } else {
                    Response response = new Response();
                    response.setSuccess(success);
                    response.setCode(success ? 200 : 500);
                    response.setEnqueuedByMaster(true);
                    response.setServerStatus(wire.getServerStatusList().stream().map(server -> {
                        var status = new org.flexlb.dao.loadbalance.ServerStatus();
                        status.setRole(org.flexlb.dao.route.RoleType.fromString(server.getRole()));
                        status.setServerIp(server.getServerIp());
                        status.setHttpPort(server.getHttpPort());
                        return status;
                    }).toList());
                    local.complete(response);
                }
            }).join();
            assertEquals(1, exporter.spans.size());
            var data = exporter.spans.getFirst();
            assertEquals(success ? io.opentelemetry.api.trace.StatusCode.OK
                    : io.opentelemetry.api.trace.StatusCode.ERROR, data.getStatus().getStatusCode());
            assertEquals(success ? "10.0.0.10:8000" : null, data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.stringKey("rtp_llm.prefill_address")));
            assertEquals(success && !fusion ? "10.0.0.20:9000" : null, data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.stringKey("rtp_llm.decode_address")));
            assertEquals(forwarded ? null : 20L, data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.batch_wait_ms")));
            assertEquals(forwarded ? null : 10L, data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.route_submit_ms")));
            if (forwarded || missingAck) {
                org.junit.jupiter.api.Assertions.assertNull(data.getAttributes().get(
                        io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.ack_to_response_ms")));
            }
            org.junit.jupiter.api.Assertions.assertNull(data.getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.enqueue_batch_ms")));
        }
    }

    private static final class RecordingExporter
            implements io.opentelemetry.sdk.trace.export.SpanExporter {
        private final java.util.List<io.opentelemetry.sdk.trace.data.SpanData> spans =
                new java.util.ArrayList<>();

        @Override
        public io.opentelemetry.sdk.common.CompletableResultCode export(
                java.util.Collection<io.opentelemetry.sdk.trace.data.SpanData> batch) {
            spans.addAll(batch);
            return io.opentelemetry.sdk.common.CompletableResultCode.ofSuccess();
        }

        @Override
        public io.opentelemetry.sdk.common.CompletableResultCode flush() {
            return io.opentelemetry.sdk.common.CompletableResultCode.ofSuccess();
        }

        @Override
        public io.opentelemetry.sdk.common.CompletableResultCode shutdown() {
            return io.opentelemetry.sdk.common.CompletableResultCode.ofSuccess();
        }
    }

    @Test
    void testSchedule_observerFailureStillWritesPvRecord() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(any(RequestContext.class)))
                .thenAnswer(call -> {
                    org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(call.getArgument(0, RequestContext.class), requestState);
                    return CompletableFuture.completedFuture(response);
                });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        doThrow(new RuntimeException("client disconnected"))
                .when(observer).onNext(any());

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(88_001L)
                .build(), observer);

        verify(observer, times(1)).onNext(any());
        verify(observer, never()).onCompleted();
        verify(requestState).onResponseUndeliverable(org.mockito.ArgumentMatchers.argThat(context -> context.getRequestId() == 88_001L));
        assertPvContains("\"requestId\":88001");
        assertPvContains("\"scheduleOrigin\":\"LOCAL_STANDALONE\"");
    }

    @Test
    void testSchedule_buildContextPreservesCacheKeyBlockSize() {
        // Given: not master, no consistency needed
        when(masterStatusService.isNeedConsistency()).thenReturn(false);

        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);

        ArgumentCaptor<RequestContext> ctxCaptor = ArgumentCaptor.forClass(RequestContext.class);
        when(requestScheduler.submit(ctxCaptor.capture())).thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(99999L)
                .setSeqLen(2048)
                .setCacheKeyBlockSize(1024L)
                .addBlockCacheKeys(100L)
                .addBlockCacheKeys(200L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then: verify cacheKeyBlockSize is propagated to Request
        RequestContext capturedCtx = ctxCaptor.getValue();
        Request capturedRequest = capturedCtx.getRequest();
        assertEquals(1024L, capturedRequest.getCacheKeyBlockSize());
        assertEquals(2, capturedRequest.getBlockCacheKeys().size());
        assertEquals(100L, capturedRequest.getBlockCacheKeys().get(0));
        assertEquals(200L, capturedRequest.getBlockCacheKeys().get(1));
        assertEquals(2048L, capturedRequest.getSeqLen());
        assertEquals(Request.DEFAULT_GENERATE_TIMEOUT_MS,
                capturedRequest.getGenerateTimeout());
        assertEquals(capturedCtx.getStartTime() + 3_600_000L,
                capturedCtx.getRequestExpiresAtMs());
    }

    @Test
    void queueTimeoutComesFromFlexlbConfigAndOverridesCallerTimeout() {
        FlexlbConfig queueConfig = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","queueTimeoutMs":7777,
                    "ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"},
                  "requestLifecycle":{"request":{"timeoutMs":3600000},"decision":{"lifetime":2}}
                }
                """);
        when(configService.loadBalanceConfig()).thenReturn(queueConfig);
        createService();
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        ArgumentCaptor<RequestContext> context = ArgumentCaptor.forClass(RequestContext.class);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(context.capture())).thenReturn(
                CompletableFuture.completedFuture(response));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(100_001L)
                .setGenerateTimeout(1L)
                .setRequestTimeMs(1L)
                .build(), mock(StreamObserver.class));

        RequestContext captured = context.getValue();
        assertEquals(captured.getStartTime() + 7777L, captured.getRequestExpiresAtMs());
    }

    @Test
    void directModeHasNoSchedulingTimeout() {
        FlexlbConfig directConfig = ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"},
                  "requestLifecycle":{"request":{"timeoutMs":3600000},"decision":{"lifetime":2}}
                }
                """);
        when(configService.loadBalanceConfig()).thenReturn(directConfig);
        createService();
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        ArgumentCaptor<RequestContext> context = ArgumentCaptor.forClass(RequestContext.class);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(context.capture())).thenReturn(
                CompletableFuture.completedFuture(response));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(100_002L)
                .setGenerateTimeout(1L)
                .setRequestTimeMs(1L)
                .build(), mock(StreamObserver.class));

        assertEquals(Long.MAX_VALUE, context.getValue().getRequestExpiresAtMs());
    }

    @Test
    void testSchedule_returnsBatchIdAndLifecycle() {
        when(masterStatusService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(requestScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(response));
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestState).getRequestState(700L, 0)).thenReturn(
                new RequestState(700L, RequestState.Phase.ACKNOWLEDGED,
                        DeliveryClaimKind.BATCH_ENQUEUE, 1001L, 10L, 20L,
                        "engine acknowledged batch"));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(700L)
                .build(), observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        assertEquals(FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_ACKNOWLEDGED,
                captor.getValue().getLifecycle().getState());
        assertEquals(1001L, captor.getValue().getLifecycle().getBatchId());
    }

    @Test
    void testGetRequestState_rejectsStaleBatchIdAsNotFound() {
        when(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestState).getRequestState(702L, 1002L)).thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> observer = mock(StreamObserver.class);

        service.getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setRequestId(702L)
                .setBatchId(1002L)
                .build(), observer);

        ArgumentCaptor<FlexlbScheduleProtocol.GetRequestStateResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.GetRequestStateResponsePB.class);
        verify(observer).onNext(captor.capture());
        assertFalse(captor.getValue().getFound());
    }

    private void assertPvContains(String expected) {
        assertEquals(1, pvAppender.list.size());
        assertTrue(pvAppender.list.get(0).getFormattedMessage().contains(expected),
                pvAppender.list.get(0).getFormattedMessage());
    }

}
