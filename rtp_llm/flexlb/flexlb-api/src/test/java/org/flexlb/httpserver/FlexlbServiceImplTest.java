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
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
import static org.mockito.Mockito.when;

class FlexlbServiceImplTest {

    private RouteService routeService;
    private LBStatusConsistencyService lbStatusConsistencyService;
    private EngineHealthReporter engineHealthReporter;
    private ActiveRequestCounter activeRequestCounter;
    private FlexlbGrpcForwarder grpcForwarder;
    private ConfigService configService;
    private BatchSchedulerReporter batchSchedulerReporter;
    private ServerScheduleLatencyRecorder serverLatencyRecorder;
    private ActiveRequestCounter.RequestToken requestToken;
    private FlexlbServiceImpl service;
    private ch.qos.logback.classic.Logger pvLogger;
    private ListAppender<ILoggingEvent> pvAppender;

    @BeforeEach
    void setUp() {
        org.flexlb.telemetry.FlexlbTrace.configureEnabled(true);
        routeService = mock(RouteService.class);
        lbStatusConsistencyService = mock(LBStatusConsistencyService.class);
        engineHealthReporter = mock(EngineHealthReporter.class);
        activeRequestCounter = mock(ActiveRequestCounter.class);
        grpcForwarder = mock(FlexlbGrpcForwarder.class);
        batchSchedulerReporter = mock(BatchSchedulerReporter.class);
        serverLatencyRecorder = mock(ServerScheduleLatencyRecorder.class);

        configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        requestToken = mock(ActiveRequestCounter.RequestToken.class);
        when(activeRequestCounter.acquire()).thenReturn(requestToken);

        service = new FlexlbServiceImpl(
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                activeRequestCounter,
                grpcForwarder,
                configService,
                batchSchedulerReporter,
                serverLatencyRecorder,
                mock(RequestSchedulerReporter.class)
        );

        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        pvAppender = new ListAppender<>();
        pvAppender.start();
        pvLogger.addAppender(pvAppender);
    }

    @AfterEach
    void tearDown() {
        org.flexlb.telemetry.FlexlbTrace.configureEnabled(false);
        pvLogger.detachAppender(pvAppender);
        pvAppender.stop();
    }

    @Test
    void testSchedule_localRouting() {
        // Given: not master, no consistency needed
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);

        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.completedFuture(response));

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
        assertPvContains("\"scheduleOrigin\":\"LOCAL_STANDALONE\"");
        verify(serverLatencyRecorder).recordArrival(anyLong());
        verify(serverLatencyRecorder).recordCompletion(any(BalanceContext.class), anyLong());
    }

    @Test
    void testSchedule_clientCancellationReleasesSchedulerOwnedRequest() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        CompletableFuture<Response> pendingRoute = new CompletableFuture<>();
        when(routeService.route(any(BalanceContext.class))).thenReturn(pendingRoute);
        when(routeService.cancelRequest(12_356L, 0L, CancelReason.CLIENT_CANCELLED))
                .thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_356L)
                        .build();
        Context.CancellableContext inbound = Context.current().withCancellation();

        inbound.run(() -> service.schedule(request, observer));
        inbound.cancel(null);

        verify(routeService).cancelRequest(
                12_356L, 0L, CancelReason.CLIENT_CANCELLED);
        verifyNoInteractions(observer);
        verify(requestToken).close();

        Response lateRoute = new Response();
        lateRoute.setSuccess(true);
        lateRoute.setCode(200);
        pendingRoute.complete(lateRoute);

        verifyNoInteractions(observer);
        verify(requestToken, times(1)).close();
    }

    @Test
    void testSchedule_alreadyCancelledContextCannotRaceAheadOfRegistration() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        CompletableFuture<Response> pendingRoute = new CompletableFuture<>();
        when(routeService.route(any(BalanceContext.class))).thenReturn(pendingRoute);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        Context.CancellableContext inbound = Context.current().withCancellation();
        inbound.cancel(null);

        inbound.run(() -> service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12_357L)
                        .build(), observer));

        var inOrder = inOrder(routeService);
        inOrder.verify(routeService).route(any(BalanceContext.class));
        inOrder.verify(routeService).cancelRequest(
                12_357L, 0L, CancelReason.CLIENT_CANCELLED);
        verifyNoInteractions(observer);
        verify(requestToken).close();
    }

    @Test
    void testSchedule_preservesBothEnqueuedByMasterValues() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);

        for (boolean expected : new boolean[]{false, true}) {
            Response response = new Response();
            response.setSuccess(true);
            response.setCode(200);
            response.setEnqueuedByMaster(expected);
            when(routeService.route(any(BalanceContext.class)))
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
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = Response.error(
                StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
        when(routeService.route(any(BalanceContext.class)))
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

    @Test
    void testSchedule_forwardToMaster_success() {
        // Given: consistency needed, not master, forward succeeds
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);

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
        verify(routeService, never()).route(any());

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
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
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
        verify(requestToken, never()).close();
        verify(routeService, never()).route(any());

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
        verify(requestToken, times(1)).close();
        verify(routeService, never()).route(any());
    }

    @Test
    @SuppressWarnings("unchecked")
    void testSchedule_callbackRegistrationFailureStillCompletesExactlyOnce() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
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
        verify(requestToken, times(1)).close();
        verify(routeService, never()).route(any());
    }

    @Test
    void testSchedule_exceptionalMasterCompletionIsTerminal() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
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
        verify(requestToken, times(1)).close();
        verify(routeService, never()).route(any());
    }

    @Test
    void testSchedule_forwardObserverFailureDoesNotSendSecondResponse() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
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
        verify(routeService, never()).route(any());
        verify(observer, times(1)).onNext(any());
        verify(observer, never()).onCompleted();
        verify(requestToken, times(1)).close();
    }

    @Test
    void testSchedule_masterNotFoundRoutesLocallyAsFallback() {
        // No Master address was selected, so no RPC was attempted.
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.noMaster()));

        Response localResponse = new Response();
        localResponse.setSuccess(true);
        localResponse.setCode(200);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.completedFuture(localResponse));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardScheduleToMaster(request);
        verify(routeService).route(any(BalanceContext.class));

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
        assertPvContains("\"scheduleOrigin\":\"LOCAL_FALLBACK\"");
    }

    @Test
    void testSchedule_forwardFailureIsTerminalAndNeverRoutesLocally() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.MasterForwardResult> pendingForward =
                new CompletableFuture<>();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(pendingForward);
        when(grpcForwarder.forwardCompensatingCancelToMaster(any(), any(), any(io.opentelemetry.context.Context.class)))
                .thenAnswer(invocation -> {
                    // The schedule RPC inherited the cancelled inbound Context. Its
                    // reconciliation must not inherit that cancellation as well.
                    assertFalse(Context.current().isCancelled());
                    return CompletableFuture.completedFuture(
                            FlexlbGrpcForwarder.CancelForwardResult.noMaster());
                });

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
                        "CANCELLED", "10.0.0.2:7001")));

        verify(routeService, never()).route(any());
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelRequestPB> cancel =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelRequestPB.class);
        verify(grpcForwarder).forwardCompensatingCancelToMaster(
                cancel.capture(), org.mockito.ArgumentMatchers.eq("10.0.0.2:7001"),
                any(io.opentelemetry.context.Context.class));
        assertEquals(request.getRequestId(), cancel.getValue().getRequestId());
        assertEquals(
                FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED,
                cancel.getValue().getReason());
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        assertFalse(captor.getValue().getSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                captor.getValue().getCode());
        assertPvContains("\"code\":8511");
        assertPvContains("\"scheduleOrigin\":\"FORWARD_FAILED\"");
        assertPvContains("\"requestExpiresAtMs\":");
        assertPvContains("\"realMasterHost\":\"10.0.0.2:7001\"");
    }

    @Test
    void testSchedule_guardFailureDoesNotStartCancellationReconciliation() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.failed(
                                "FORWARD_HOP_LIMIT", "10.0.0.2:7001")));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12_349L)
                .build(), mock(StreamObserver.class));

        verify(grpcForwarder, never())
                .forwardCompensatingCancelToMaster(any(), any(), any(io.opentelemetry.context.Context.class));
        verify(routeService, never()).route(any());
    }

    @Test
    void testSchedule_exceptionHandling() {
        // Given: route throws exception
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.failedFuture(new RuntimeException("test error")));

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
        assertEquals(500, resp.getCode());
        assertTrue(resp.getErrorMessage().contains("test error"));
    }

    @Test
    void testSchedule_entryErrorMarksServerSpanFromInterceptorContext() {
        // buildContext() throws before ctx is assigned, so completeSchedule() gets
        // a null BalanceContext. The SERVER span must still carry the internal
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
        try {
            io.opentelemetry.api.trace.Span serverSpan =
                    org.flexlb.telemetry.FlexlbTrace.startServer(
                            "rtp_llm.flexlb.schedule", io.opentelemetry.context.Context.root());

            // buildContext() reads loadBalanceConfig(); make it throw.
            when(configService.loadBalanceConfig())
                    .thenThrow(new IllegalStateException("config unavailable"));

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
            assertEquals(500, resp.getCode());

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
            assertEquals(500L,
                    span.getAttributes().get(
                            io.opentelemetry.api.common.AttributeKey.longKey("flexlb.schedule.code")));
            assertTrue(span.getEvents().isEmpty());
        } finally {
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
            when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
            when(lbStatusConsistencyService.isMaster()).thenReturn(false);
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
            verify(routeService, never()).route(any());
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
            var captured = new java.util.concurrent.atomic.AtomicReference<BalanceContext>();
            org.mockito.Mockito.doAnswer(inv -> {
                BalanceContext ctx = inv.getArgument(0);
                captured.set(ctx);
                long now = System.nanoTime();
                ctx.setServiceStartNanos(now - 40_000_000L);
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
            when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(forwarded);
            when(lbStatusConsistencyService.isMaster()).thenReturn(false);
            when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(remote);
            when(routeService.route(any())).thenReturn(local);
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
            assertEquals(success ? io.opentelemetry.api.trace.StatusCode.OK : io.opentelemetry.api.trace.StatusCode.ERROR,
                    data.getStatus().getStatusCode());
            assertEquals(success ? "10.0.0.10:8000" : null,
                    data.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey("rtp_llm.prefill_address")));
            assertEquals(success && !fusion ? "10.0.0.20:9000" : null,
                    data.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey("rtp_llm.decode_address")));
            assertEquals(forwarded ? null : 20L,
                    data.getAttributes().get(io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.batch_wait_ms")));
            assertEquals(forwarded ? null : 10L,
                    data.getAttributes().get(io.opentelemetry.api.common.AttributeKey.longKey("rtp_llm.route_submit_ms")));
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
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        doThrow(new RuntimeException("client disconnected"))
                .when(observer).onNext(any());

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(88_001L)
                .build(), observer);

        verify(observer, times(1)).onNext(any());
        verify(observer, never()).onCompleted();
        verify(routeService).cancelRequest(
                88_001L, 0L, CancelReason.CLIENT_CANCELLED);
        assertPvContains("\"requestId\":88001");
        assertPvContains("\"scheduleOrigin\":\"LOCAL_STANDALONE\"");
    }

    @Test
    void testSchedule_buildContextPreservesCacheKeyBlockSize() {
        // Given: not master, no consistency needed
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);

        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);

        ArgumentCaptor<BalanceContext> ctxCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        when(routeService.route(ctxCaptor.capture())).thenReturn(CompletableFuture.completedFuture(response));

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
        BalanceContext capturedCtx = ctxCaptor.getValue();
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
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        when(configService.loadBalanceConfig()).thenReturn(queueConfig);
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(context.capture())).thenReturn(
                CompletableFuture.completedFuture(response));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(100_001L)
                .setGenerateTimeout(1L)
                .setRequestTimeMs(1L)
                .build(), mock(StreamObserver.class));

        BalanceContext captured = context.getValue();
        assertEquals(captured.getStartTime() + 7777L, captured.getRequestExpiresAtMs());
    }

    @Test
    void directModeHasNoSchedulingTimeout() {
        FlexlbConfig directConfig = ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        when(configService.loadBalanceConfig()).thenReturn(directConfig);
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(context.capture())).thenReturn(
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
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any())).thenReturn(CompletableFuture.completedFuture(response));
        when(routeService.getRequestState(700L, 0)).thenReturn(
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
        when(routeService.getRequestState(702L, 1002L)).thenReturn(null);
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
