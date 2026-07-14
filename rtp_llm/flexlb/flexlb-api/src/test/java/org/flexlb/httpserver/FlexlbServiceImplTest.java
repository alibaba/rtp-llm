package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class FlexlbServiceImplTest {

    private RouteService routeService;
    private LBStatusConsistencyService lbStatusConsistencyService;
    private EngineHealthReporter engineHealthReporter;
    private ActiveRequestCounter activeRequestCounter;
    private FlexlbGrpcForwarder grpcForwarder;
    private ConfigService configService;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        lbStatusConsistencyService = mock(LBStatusConsistencyService.class);
        engineHealthReporter = mock(EngineHealthReporter.class);
        activeRequestCounter = mock(ActiveRequestCounter.class);
        grpcForwarder = mock(FlexlbGrpcForwarder.class);

        configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        ActiveRequestCounter.RequestToken token = mock(ActiveRequestCounter.RequestToken.class);
        when(activeRequestCounter.acquire()).thenReturn(token);

        service = new FlexlbServiceImpl(
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                activeRequestCounter,
                grpcForwarder,
                configService
        );
    }

    @Test
    void testSchedule_localRouting() {
        // Given: not master, no consistency needed
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);

        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.completedFuture(response));

        EngineRpcService.FlexlbScheduleRequestPB request = EngineRpcService.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .setSeqLen(100)
                .setCacheKeyBlockSize(1024L)
                .build();

        StreamObserver<EngineRpcService.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        ArgumentCaptor<EngineRpcService.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(EngineRpcService.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());

        EngineRpcService.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
        assertEquals(200, resp.getCode());
    }

    @Test
    void testSchedule_forwardToMaster_success() {
        // Given: consistency needed, not master, forward succeeds
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);

        EngineRpcService.FlexlbScheduleResponsePB masterResponse = EngineRpcService.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(true)
                .setCode(200)
                .build();
        when(grpcForwarder.forwardToMaster(any())).thenReturn(masterResponse);

        EngineRpcService.FlexlbScheduleRequestPB request = EngineRpcService.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<EngineRpcService.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardToMaster(request);
        verify(routeService, never()).route(any());

        ArgumentCaptor<EngineRpcService.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(EngineRpcService.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());

        EngineRpcService.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
    }

    @Test
    void testSchedule_forwardToMaster_fallbackToLocal() {
        // Given: consistency needed, not master, forward fails (returns null)
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardToMaster(any())).thenReturn(null);

        Response localResponse = new Response();
        localResponse.setSuccess(true);
        localResponse.setCode(200);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.completedFuture(localResponse));

        EngineRpcService.FlexlbScheduleRequestPB request = EngineRpcService.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<EngineRpcService.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardToMaster(request);
        verify(routeService).route(any(BalanceContext.class));

        ArgumentCaptor<EngineRpcService.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(EngineRpcService.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());

        EngineRpcService.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
    }

    @Test
    void testSchedule_exceptionHandling() {
        // Given: route throws exception
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        when(routeService.route(any(BalanceContext.class))).thenReturn(CompletableFuture.failedFuture(new RuntimeException("test error")));

        EngineRpcService.FlexlbScheduleRequestPB request = EngineRpcService.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<EngineRpcService.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        ArgumentCaptor<EngineRpcService.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(EngineRpcService.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        EngineRpcService.FlexlbScheduleResponsePB resp = captor.getValue();
        assertFalse(resp.getSuccess());
        assertEquals(500, resp.getCode());
        assertTrue(resp.getErrorMessage().contains("test error"));
    }

    @Test
    void testCancel_success() {
        // Given
        doNothing().when(routeService).cancelByRequestId(12345L);

        EngineRpcService.CancelRequestPB request = EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<EngineRpcService.EmptyPB> observer = mock(StreamObserver.class);

        // When
        service.cancel(request, observer);

        // Then
        verify(routeService).cancelByRequestId(12345L);
        verify(observer).onNext(any(EngineRpcService.EmptyPB.class));
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
    }

    @Test
    void testCancel_exceptionHandling() {
        // Given
        doThrow(new RuntimeException("cancel error")).when(routeService).cancelByRequestId(12345L);

        EngineRpcService.CancelRequestPB request = EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<EngineRpcService.EmptyPB> observer = mock(StreamObserver.class);

        // When
        service.cancel(request, observer);

        // Then
        verify(routeService).cancelByRequestId(12345L);
        verify(observer).onError(any());
        verify(observer, never()).onNext(any());
        verify(observer, never()).onCompleted();
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

        EngineRpcService.FlexlbScheduleRequestPB request = EngineRpcService.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(99999L)
                .setSeqLen(2048)
                .setCacheKeyBlockSize(1024L)
                .addBlockCacheKeys(100L)
                .addBlockCacheKeys(200L)
                .build();

        StreamObserver<EngineRpcService.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

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
    }
}
