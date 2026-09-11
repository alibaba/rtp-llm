package org.flexlb.httpserver;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.google.protobuf.CodedOutputStream;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.cache.match.CacheAwareService;
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
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.optimizer.OptimizerClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;
import org.mockito.InOrder;
import org.mockito.Mockito;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
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
    private CacheAwareService cacheAwareService;
    private ActiveRequestCounter.RequestToken requestToken;
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
        cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.prepareBlockCacheKeys(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(null));

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
                mock(RequestSchedulerReporter.class),
                cacheAwareService,
                mock(OptimizerClient.class)
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
                .setRequestId(String.valueOf(12345L))
                .addInputIds(1)
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
        assertFalse(pvAppender.list.get(0).getFormattedMessage().contains("\"admissionRejectReason\""));
        verify(serverLatencyRecorder).recordArrival(anyLong());
        verify(serverLatencyRecorder).recordCompletion(any(BalanceContext.class), anyLong());
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void deliveryToResponseMetricUsesSelectedWorker(RoleType role) {
        ServerStatus worker = new ServerStatus();
        worker.setRole(role);
        worker.setServerIp("10.0.0.1");
        worker.setHttpPort(8080);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(worker));
        when(routeService.route(any())).thenAnswer(invocation -> {
            BalanceContext context = invocation.getArgument(0);
            context.setAckAtMs(System.currentTimeMillis() - 10L);
            context.setResponse(response);
            return CompletableFuture.completedFuture(response);
        });
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("delivery-metric").addInputIds(1).setSeqLen(1).build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(request, observer);
        verify(batchSchedulerReporter).reportAckToResponseTimeMs(
                ArgumentMatchers.eq(role.name()), ArgumentMatchers.eq(worker.getMetricIpPort()), anyLong());
        verify(observer).onCompleted();
    }

    @Test
    void pendingRoutePublishesResponseAndTimingsBeforePv() throws Exception {
        CompletableFuture<Response> pending = new CompletableFuture<>();
        when(routeService.route(any())).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            pending.whenComplete((value, failure) -> ctx.setResponse(value));
            ctx.setServiceStartNanos(System.nanoTime() - 1_000_000L);
            ctx.getRequest().clearInputIds();
            return pending;
        });
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("pv-pending").setSeqLen(2).addInputIds(11).addInputIds(22)
                .setRequestTimeMs(System.currentTimeMillis() - 100L).build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(request, observer);
        assertTrue(pvAppender.list.isEmpty());
        var worker = new org.flexlb.dao.loadbalance.ServerStatus();
        worker.setServerIp("10.0.0.1");
        worker.setRole(org.flexlb.dao.route.RoleType.PREFILL);
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(worker));
        pending.complete(response);

        assertEquals(1, pvAppender.list.size());
        var json = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(pvAppender.list.getFirst().getFormattedMessage());
        assertEquals("10.0.0.1", json.path("response").path("server_status").get(0).path("server_ip").asText());
        assertEquals(2, json.path("seqLen").asInt());
        assertFalse(json.has("inputIdsCount"));
        assertFalse(json.has("requestMessageBytes"));
        assertTrue(json.path("totalUs").asLong() >= 1_000L);
        assertTrue(json.path("arrivalMs").asLong() >= 100L);
        assertFalse(json.has("reqParseUs"));
        assertFalse(json.has("requestBodyBytes"));
        assertFalse(json.has("admissionRejectReason"));
        assertFalse(json.path("response").has("admission_reject_reason"));
        ArgumentCaptor<BalanceContext> payloadContext = ArgumentCaptor.forClass(BalanceContext.class);
        verify(engineHealthReporter).reportRequestPayload(payloadContext.capture());
        verify(engineHealthReporter).reportArriveDelayTime(payloadContext.getValue());
        assertEquals((long) request.getSerializedSize(), payloadContext.getValue().getRequestMessageBytes());
        assertEquals(2L, payloadContext.getValue().getInputIdsCount());
        assertNull(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.getDescriptor()
                .findFieldByName("real_master_host"));
        verify(cacheAwareService).updateFromRoutedRequest(any(), any());
        verify(observer).onCompleted();
    }

    @Test
    void expiredRpcWritesOneTerminalPvEvenWhenRouteCompletesLater() {
        var timer = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
        CompletableFuture<Response> pending = new CompletableFuture<>();
        when(routeService.route(any())).thenReturn(pending);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        try (Context.CancellableContext inbound = Context.current().withDeadlineAfter(
                -1L, java.util.concurrent.TimeUnit.SECONDS, timer)) {
            inbound.run(() -> service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId("pv-deadline").addInputIds(1).build(), observer));
            pending.complete(new Response());
            assertPvContains("\"requestState\":\"REQUEST_STATE_TIMED_OUT\"");
            assertPvContains("\"success\":false");
            assertPvContains("\"code\":" + StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode());
            verifyNoInteractions(observer);
            verify(requestToken, times(1)).close();
            verify(engineHealthReporter).reportRequestPayload(any());
        } finally {
            timer.shutdownNow();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void invalidIdentityStillWritesEntryFailurePv(boolean observerThrows) {
        doThrow(new IllegalStateException("completion monitor unavailable")).when(serverLatencyRecorder)
                .recordCompletion(any(), anyLong());
        doThrow(new IllegalStateException("payload monitor unavailable")).when(engineHealthReporter)
                .reportRequestPayload(any());
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        RuntimeException deliveryError = new IllegalStateException("observer closed");
        if (observerThrows) {
            doThrow(deliveryError).when(observer).onError(any());
        }
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .addInputIds(1).addInputIds(2).setRequestTimeMs(System.currentTimeMillis() - 100).build();
        if (observerThrows) {
            assertEquals(deliveryError, org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                    () -> service.schedule(request, observer)));
        } else {
            service.schedule(request, observer);
        }
        ArgumentCaptor<Throwable> error = ArgumentCaptor.forClass(Throwable.class);
        verify(observer).onError(error.capture());
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(error.getValue()).getCode());
        assertEquals("Missing request ID", Status.fromThrowable(error.getValue()).getDescription());
        verify(observer, never()).onNext(any());
        verify(observer, never()).onCompleted();
        verifyNoInteractions(activeRequestCounter, routeService, grpcForwarder);
        verify(serverLatencyRecorder).recordArrival(anyLong());
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        verify(serverLatencyRecorder).recordCompletion(context.capture(), anyLong());
        assertEquals(2L, context.getValue().getInputIdsCount());
        assertEquals((long) request.getSerializedSize(), context.getValue().getRequestMessageBytes());
        verify(engineHealthReporter).reportRequestPayload(context.getValue());
        assertTrue(context.getValue().getRequestArrivalDelayMs() >= 100);
        assertEquals(1, pvAppender.list.size());
        assertPvContains("\"scheduleOrigin\":\"ENTRY_ERROR\"");
        assertPvContains("\"success\":false");
        assertPvContains("\"code\":" + StrategyErrorType.INVALID_REQUEST.getErrorCode());
        assertFalse(pvAppender.list.getFirst().getFormattedMessage().contains("\"requestId\""));
    }

    @Test
    void requestInitializationFailureStillFinalizesEntryContext() {
        when(configService.loadBalanceConfig()).thenThrow(new IllegalStateException("config unavailable"));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("entry-config-failure").addInputIds(1).build(), observer);
        verify(observer).onCompleted();
        verify(requestToken).close();
        verify(serverLatencyRecorder).recordArrival(anyLong());
        verify(serverLatencyRecorder).recordCompletion(any(BalanceContext.class), anyLong());
        verifyNoInteractions(routeService, grpcForwarder);
        assertEquals(1, pvAppender.list.size());
        assertPvContains("\"requestId\":\"entry-config-failure\"");
        assertPvContains("\"scheduleOrigin\":\"ENTRY_ERROR\"");
        assertPvContains("\"success\":false");
    }

    @Test
    void testSchedule_acceptsEmptyCacheKeysWithoutInputIds() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("short-prompt")
                        .setSeqLen(100)
                        .setCacheKeyBlockSize(1024)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
        assertTrue(captor.getValue().getSuccess());
        verify(cacheAwareService).prepareBlockCacheKeys(any(BalanceContext.class));
        verify(routeService).route(any(BalanceContext.class));
    }

    @Test
    void testSchedule_forwardsEmptyCacheKeysWithoutInputIds() {
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

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("short-prompt-forward")
                        .setSeqLen(100)
                        .setCacheKeyBlockSize(1024)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        assertTrue(captor.getValue().getSuccess());
        verify(grpcForwarder).forwardScheduleToMaster(request);
        verify(cacheAwareService, never()).prepareBlockCacheKeys(any(BalanceContext.class));
        verify(routeService, never()).route(any(BalanceContext.class));
    }

    @Test
    void testSchedule_acceptsBlockCacheKeysWithoutInputIds() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
        ArgumentCaptor<BalanceContext> contextCaptor =
                ArgumentCaptor.forClass(BalanceContext.class);

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("block-id-request")
                        .addBlockCacheKeys(101L)
                        .addBlockCacheKeys(202L)
                        .setSeqLen(2)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        verify(cacheAwareService).prepareBlockCacheKeys(contextCaptor.capture());
        assertEquals(List.of(101L, 202L),
                contextCaptor.getValue().getRequest().getBlockCacheKeys());
        verify(routeService).route(contextCaptor.getValue());
        verify(observer).onCompleted();
    }

    @Test
    void testSchedule_propagatesInputIdsToCachePreparation() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
        ArgumentCaptor<BalanceContext> contextCaptor =
                ArgumentCaptor.forClass(BalanceContext.class);

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("input-id-request")
                        .addInputIds(11)
                        .addInputIds(22)
                        .addInputIds(33)
                        .setSeqLen(3)
                        .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request, observer);

        verify(cacheAwareService).prepareBlockCacheKeys(contextCaptor.capture());
        var inputIds = contextCaptor.getValue().getRequest().getInputIds();
        assertEquals(3, inputIds.size());
        assertEquals(11, inputIds.getInt(0));
        assertEquals(22, inputIds.getInt(1));
        assertEquals(33, inputIds.getInt(2));
        verify(routeService).route(contextCaptor.getValue());
        verify(observer).onCompleted();
    }

    @Test
    void testSchedule_clientCancellationReleasesSchedulerOwnedRequest() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        CompletableFuture<Response> pendingRoute = new CompletableFuture<>();
        when(routeService.route(any(BalanceContext.class))).thenReturn(pendingRoute);
        when(routeService.cancelRequest("12356", 0L, CancelReason.CLIENT_CANCELLED))
                .thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("12356")
                        .addInputIds(1)
                        .build();
        Context.CancellableContext inbound = Context.current().withCancellation();

        inbound.run(() -> service.schedule(request, observer));
        inbound.cancel(null);

        verify(routeService).cancelRequest(
                "12356", 0L, CancelReason.CLIENT_CANCELLED);
        verifyNoInteractions(observer);
        verify(requestToken, never()).close();
        assertTrue(pvAppender.list.isEmpty());
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        verify(routeService).route(context.capture());
        context.getValue().recordCacheQuery("KVCM", 45);

        Response lateRoute = new Response();
        lateRoute.setSuccess(true);
        lateRoute.setCode(200);
        pendingRoute.complete(lateRoute);

        verifyNoInteractions(observer);
        verify(requestToken, times(1)).close();
        assertPvContains("\"requestState\":\"REQUEST_STATE_CANCELLED\"");
        assertPvContains("\"success\":false");
        assertPvContains("\"code\":8504");
        assertPvContains("\"cacheMatchUs\":45");
        assertEquals(1, pvAppender.list.size());
        verify(serverLatencyRecorder, times(1)).recordCompletion(any(), anyLong());
    }

    @Test
    void alreadyCancelledContextSkipsHashAndRoutingAndFinalizesOnce() {
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        try (Context.CancellableContext inbound = Context.current().withCancellation()) {
            inbound.cancel(null);
            inbound.run(() -> service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId("12357").addInputIds(1).build(), observer));
        }
        verify(cacheAwareService, never()).prepareBlockCacheKeys(any());
        verify(routeService, never()).route(any());
        verifyNoInteractions(observer);
        verify(requestToken, times(1)).close();
        assertEquals(1, pvAppender.list.size());
        assertPvContains("\"requestState\":\"REQUEST_STATE_CANCELLED\"");
    }

    @Test
    void cancellationDuringHashWaitsForHashTelemetryAndSkipsRouting() {
        CompletableFuture<Void> hash = new CompletableFuture<>();
        when(cacheAwareService.prepareBlockCacheKeys(any())).thenReturn(hash);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        try (Context.CancellableContext inbound = Context.current().withCancellation()) {
            inbound.run(() -> service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId("cancel-hash").addInputIds(1).build(), observer));
            inbound.cancel(null);
            assertTrue(pvAppender.list.isEmpty());
            verify(requestToken, never()).close();
            ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
            verify(cacheAwareService).prepareBlockCacheKeys(context.capture());
            context.getValue().recordBlockHashTiming(12, 34);
            hash.complete(null);
        }
        verify(routeService, never()).route(any());
        verifyNoInteractions(observer);
        verify(requestToken, times(1)).close();
        assertEquals(1, pvAppender.list.size());
        assertFalse(pvAppender.list.getFirst().getFormattedMessage().contains("\"hashUs\""));
        assertPvContains("\"requestState\":\"REQUEST_STATE_CANCELLED\"");
    }

    @Test
    void cancellationRacingSchedulerRegistrationIsReconciledBeforeFinalization() {
        CompletableFuture<Response> route = new CompletableFuture<>();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        try (Context.CancellableContext inbound = Context.current().withCancellation()) {
            when(routeService.route(any())).thenAnswer(call -> {
                inbound.cancel(null);
                return route;
            });
            inbound.run(() -> service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId("registration-race").addInputIds(1).build(), observer));
            verify(routeService, times(2)).cancelRequest("registration-race", 0L, CancelReason.CLIENT_CANCELLED);
            assertTrue(pvAppender.list.isEmpty());
            verify(requestToken, never()).close();
            route.complete(new Response());
        }
        verifyNoInteractions(observer);
        verify(requestToken, times(1)).close();
        assertEquals(1, pvAppender.list.size());
    }

    @Test
    void telemetryFailureStillWritesPvAndReleasesRequestToken() {
        when(routeService.route(any())).thenReturn(CompletableFuture.completedFuture(new Response()));
        Mockito.doThrow(new IllegalStateException("monitor unavailable")).when(serverLatencyRecorder)
                .recordCompletion(any(), anyLong());
        doThrow(new IllegalStateException("payload monitor unavailable")).when(engineHealthReporter)
                .reportRequestPayload(any());
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("monitor-failure").addInputIds(1).build(), observer);
        verify(observer).onCompleted();
        verify(engineHealthReporter, times(1)).reportRequestPayload(any());
        verify(requestToken, times(1)).close();
        assertEquals(1, pvAppender.list.size());
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
                    .setRequestId(expected ? "12351" : "12350")
                    .addInputIds(1)
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
                        .setRequestId(String.valueOf(54321L))
                        .addInputIds(1)
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
                .setRequestId(String.valueOf(12345L))
                .addInputIds(1)
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
                        .setRequestId(String.valueOf(12_352L))
                        .addInputIds(1)
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
                .setRequestId(String.valueOf(12_354L))
                .addInputIds(1)
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
                .setRequestId(String.valueOf(12_353L))
                .addInputIds(1)
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
                .setRequestId(String.valueOf(12_346L))
                .addInputIds(1)
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
                .setRequestId(String.valueOf(12345L))
                .addInputIds(1)
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
        when(grpcForwarder.forwardCancelToMaster(any())).thenAnswer(invocation -> {
            // The schedule RPC inherited the cancelled inbound Context. Its
            // reconciliation must not inherit that cancellation as well.
            assertFalse(Context.current().isCancelled());
            return CompletableFuture.completedFuture(
                    FlexlbGrpcForwarder.CancelForwardResult.noMaster());
        });

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(String.valueOf(12348L))
                        .addInputIds(1)
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
        verify(grpcForwarder).forwardCancelToMaster(cancel.capture());
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
        assertFalse(pvAppender.list.getFirst().getFormattedMessage().contains("realMasterHost"));
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
                .setRequestId("12349")
                .addInputIds(1)
                .build(), mock(StreamObserver.class));

        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(routeService, never()).route(any());
    }

    @Test
    void testSchedule_exceptionHandling() {
        // Given: route throws exception
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.failedFuture(new RuntimeException("test error")));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(String.valueOf(12345L))
                .addInputIds(1)
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
                .setRequestId(String.valueOf(88_001L))
                .addInputIds(1)
                .build(), observer);

        verify(observer, times(1)).onNext(any());
        verify(observer, never()).onCompleted();
        verify(routeService).cancelRequest(
                "88001", 0L, CancelReason.CLIENT_CANCELLED);
        assertPvContains("\"requestId\":\"88001\"");
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
                .setRequestId(String.valueOf(99999L))
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
        InOrder localRouteOrder = Mockito.inOrder(
                cacheAwareService, routeService);
        localRouteOrder.verify(cacheAwareService).prepareBlockCacheKeys(capturedCtx);
        localRouteOrder.verify(routeService).route(capturedCtx);
    }

    @Test
    void queueTimeoutComesFromFlexlbConfigAndOverridesCallerTimeout() {
        FlexlbConfig queueConfig = FlexlbConfigMerger.mergeWithDefaults("""
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
                .setRequestId(String.valueOf(100_001L))
                .addInputIds(1)
                .setGenerateTimeout(1L)
                .setRequestTimeMs(1L)
                .build(), mock(StreamObserver.class));

        BalanceContext captured = context.getValue();
        assertEquals(captured.getStartTime() + 7777L, captured.getRequestExpiresAtMs());
    }

    @Test
    void directModeHasNoSchedulingTimeout() {
        FlexlbConfig directConfig = FlexlbConfigMerger.mergeWithDefaults("""
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
                .setRequestId(String.valueOf(100_002L))
                .addInputIds(1)
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
        when(routeService.getRequestState("700", 0)).thenReturn(
                new RequestState("700", RequestState.Phase.ACKNOWLEDGED,
                        DeliveryClaimKind.BATCH_ENQUEUE, 1001L, 10L, 20L,
                        "engine acknowledged batch"));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(String.valueOf(700L))
                .addInputIds(1)
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
        when(routeService.getRequestState("702", 1002L)).thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> observer = mock(StreamObserver.class);

        service.getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setRequestId(String.valueOf(702L))
                .setBatchId(1002L)
                .build(), observer);

        ArgumentCaptor<FlexlbScheduleProtocol.GetRequestStateResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.GetRequestStateResponsePB.class);
        verify(observer).onNext(captor.capture());
        assertFalse(captor.getValue().getFound());
    }

    @ParameterizedTest
    @ValueSource(strings = {"req-abc-001", "00123", "请求-测试", "9223372036854775808"})
    void originalIdentitySurvivesScheduleStateAndCancel(String requestId) {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response routed = new Response();
        routed.setSuccess(true);
        routed.setCode(200);
        when(routeService.route(any())).thenReturn(CompletableFuture.completedFuture(routed));
        RequestState snapshot = new RequestState(requestId,
                RequestState.Phase.ACKNOWLEDGED, DeliveryClaimKind.BATCH_ENQUEUE,
                1001L, 10L, 20L, "accepted");
        when(routeService.getRequestState(requestId, 0L)).thenReturn(snapshot);
        when(routeService.getRequestState(requestId, 1001L)).thenReturn(snapshot);
        when(routeService.cancelRequest(ArgumentMatchers.eq(requestId),
                ArgumentMatchers.eq(1001L), any())).thenReturn(snapshot);

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> scheduled = mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId).addInputIds(1).build(), scheduled);
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        verify(routeService).route(context.capture());
        assertEquals(requestId, context.getValue().getRequestId());
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> scheduleResult =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(scheduled).onNext(scheduleResult.capture());
        assertEquals(requestId, scheduleResult.getValue().getLifecycle().getRequestId());

        StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> state = mock(StreamObserver.class);
        service.getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setRequestId(requestId).setBatchId(1001L).build(), state);
        ArgumentCaptor<FlexlbScheduleProtocol.GetRequestStateResponsePB> stateResult =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.GetRequestStateResponsePB.class);
        verify(state).onNext(stateResult.capture());
        assertEquals(requestId, stateResult.getValue().getLifecycle().getRequestId());

        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> cancelled = mock(StreamObserver.class);
        service.cancel(FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(requestId).setBatchId(1001L).build(), cancelled);
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> cancelResult =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(cancelled).onNext(cancelResult.capture());
        assertEquals(requestId, cancelResult.getValue().getLifecycle().getRequestId());
    }

    @Test
    void schedulesOldIntegerWireIdAsOriginalDecimalString() throws Exception {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        Response routed = new Response();
        routed.setSuccess(true);
        routed.setCode(200);
        when(routeService.route(any())).thenReturn(CompletableFuture.completedFuture(routed));
        var bytes = new ByteArrayOutputStream();
        var wire = CodedOutputStream.newInstance(bytes);
        wire.writeInt64(1, 123);
        wire.writeInt32(16, 1);
        wire.flush();
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(bytes.toByteArray());
        assertEquals("", request.getRequestId());
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(request, observer);
        ArgumentCaptor<BalanceContext> context = ArgumentCaptor.forClass(BalanceContext.class);
        verify(routeService).route(context.capture());
        assertEquals("123", context.getValue().getRequestId());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
    }

    private void assertPvContains(String expected) {
        assertEquals(1, pvAppender.list.size());
        assertTrue(pvAppender.list.get(0).getFormattedMessage().contains(expected),
                pvAppender.list.get(0).getFormattedMessage());
    }

}
