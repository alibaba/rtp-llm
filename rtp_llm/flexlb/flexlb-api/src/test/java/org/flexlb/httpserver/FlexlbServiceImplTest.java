package org.flexlb.httpserver;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.RequestLifecycleState;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.List;
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
    private SessionPlacementStore sessionPlacementStore;
    private FlexlbServiceImpl service;
    private ch.qos.logback.classic.Logger pvLogger;
    private ListAppender<ILoggingEvent> pvAppender;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        lbStatusConsistencyService = mock(LBStatusConsistencyService.class);
        engineHealthReporter = mock(EngineHealthReporter.class);
        activeRequestCounter = mock(ActiveRequestCounter.class);
        grpcForwarder = mock(FlexlbGrpcForwarder.class);
        batchSchedulerReporter = mock(BatchSchedulerReporter.class);
        serverLatencyRecorder = mock(ServerScheduleLatencyRecorder.class);
        sessionPlacementStore = mock(SessionPlacementStore.class);

        configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        var sessionAffinity = new org.flexlb.config.RoutingConfig.SessionAffinityConfig();
        sessionAffinity.setTtlMs(1_800_000L);
        sessionAffinity.setMaxExtraTtftMs(40L);
        flexlbConfig.getRouter().getRoles().getPrefill().setSessionAffinity(sessionAffinity);
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
                mock(PrioritySchedulerReporter.class),
                sessionPlacementStore
        );

        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        pvAppender = new ListAppender<>();
        pvAppender.start();
        pvLogger.addAppender(pvAppender);
    }

    @AfterEach
    void tearDown() {
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
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        doThrow(new RuntimeException("client disconnected"))
                .when(observer).onNext(any());

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12_346L)
                .build(), observer);

        verify(grpcForwarder, times(1)).forwardScheduleToMaster(any());
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
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.failed(
                                "DEADLINE_EXCEEDED", "10.0.0.2:7001")));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(12348L)
                        .setGenerateTimeout(12_345L)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        verify(routeService, never()).route(any());
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
    void testSchedule_buildContextPreservesSessionRoutingHint() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        ArgumentCaptor<BalanceContext> ctxCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        when(routeService.route(ctxCaptor.capture()))
                .thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(100_001L)
                        .setModel("kimi-k3")
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB
                                .newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_example")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.ESTABLISHED))
                        .build();

        service.schedule(request, mock(StreamObserver.class));

        Request captured = ctxCaptor.getValue().getRequest();
        assertEquals(1, captured.getSessionSchemaVersion());
        assertEquals("isess_v1_example", captured.getInferenceSessionId());
        assertEquals(Request.SessionState.ESTABLISHED, captured.getInferenceSessionState());
    }

    @Test
    void testSchedule_missingSessionHintKeepsAffinityDisabled() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        ArgumentCaptor<BalanceContext> ctxCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        when(routeService.route(ctxCaptor.capture()))
                .thenReturn(CompletableFuture.completedFuture(response));

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(100_002L)
                .build(), mock(StreamObserver.class));

        Request captured = ctxCaptor.getValue().getRequest();
        assertEquals(0, captured.getSessionSchemaVersion());
        assertEquals("", captured.getInferenceSessionId());
        assertEquals(Request.SessionState.UNSPECIFIED, captured.getInferenceSessionState());
    }

    @Test
    void successfulLocalDeliveryRecordsPrefillSessionPlacement() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.2");
        prefill.setHttpPort(8080);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(prefill));
        when(routeService.route(any(BalanceContext.class)))
                .thenAnswer(invocation -> {
                    BalanceContext context = invocation.getArgument(0);
                    context.setConfig(configService.loadBalanceConfig());
                    context.getRequest().setSessionPlacementEpoch(1L);
                    return CompletableFuture.completedFuture(response);
                });
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(100_003L)
                        .setModel("kimi-k3")
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB
                                .newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_example")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.NEW))
                        .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        service.schedule(request, observer);

        verify(sessionPlacementStore).record(
                "kimi-k3", "isess_v1_example", "10.0.0.2:8080", 1L);
        InOrder publicationOrder = inOrder(sessionPlacementStore, observer);
        publicationOrder.verify(sessionPlacementStore).record(
                "kimi-k3", "isess_v1_example", "10.0.0.2:8080", 1L);
        publicationOrder.verify(observer).onNext(any());
    }

    @Test
    void sessionPlacementFailureDoesNotFailSuccessfulScheduleResponse() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.2");
        prefill.setHttpPort(8080);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(prefill));
        when(routeService.route(any(BalanceContext.class)))
                .thenAnswer(invocation -> {
                    BalanceContext context = invocation.getArgument(0);
                    context.setConfig(configService.loadBalanceConfig());
                    context.getRequest().setSessionPlacementEpoch(1L);
                    return CompletableFuture.completedFuture(response);
                });
        doThrow(new IllegalStateException("placement unavailable"))
                .when(sessionPlacementStore)
                .record("kimi-k3", "isess_v1_example", "10.0.0.2:8080", 1L);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(100_005L)
                        .setModel("kimi-k3")
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB
                                .newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_example")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.ESTABLISHED))
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        verify(observer).onNext(any());
        verify(observer).onCompleted();
    }

    @Test
    void responseObserverFailureDoesNotSkipSessionPlacement() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.2");
        prefill.setHttpPort(8080);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(prefill));
        when(routeService.route(any(BalanceContext.class)))
                .thenAnswer(invocation -> {
                    BalanceContext context = invocation.getArgument(0);
                    context.setConfig(configService.loadBalanceConfig());
                    context.getRequest().setSessionPlacementEpoch(1L);
                    return CompletableFuture.completedFuture(response);
                });
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        doThrow(new IllegalStateException("client closed")).when(observer).onNext(any());
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(100_006L)
                        .setModel("kimi-k3")
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB
                                .newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_example")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.ESTABLISHED))
                        .build();

        service.schedule(request, observer);

        verify(sessionPlacementStore).record(
                "kimi-k3", "isess_v1_example", "10.0.0.2:8080", 1L);
    }

    @Test
    void successfulLocalDeliveryRecordsPdfusionSessionPlacement() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        ServerStatus pdfusion = new ServerStatus();
        pdfusion.setRole(RoleType.PDFUSION);
        pdfusion.setServerIp("10.0.0.3");
        pdfusion.setHttpPort(8081);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(pdfusion));
        when(routeService.route(any(BalanceContext.class)))
                .thenAnswer(invocation -> {
                    BalanceContext context = invocation.getArgument(0);
                    context.setConfig(configService.loadBalanceConfig());
                    context.getRequest().setSessionPlacementEpoch(2L);
                    return CompletableFuture.completedFuture(response);
                });
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(100_004L)
                        .setModel("kimi-k3")
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB
                                .newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_pdfusion")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.ESTABLISHED))
                        .build();

        service.schedule(request, mock(StreamObserver.class));

        verify(sessionPlacementStore).record(
                "kimi-k3", "isess_v1_pdfusion", "10.0.0.3:8081", 2L);
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
                new RequestLifecycleSnapshot(700L, RequestLifecycleState.ACKNOWLEDGED,
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
