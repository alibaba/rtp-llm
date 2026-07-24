package org.flexlb.httpserver;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.RequestLifecycleState;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

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
    private BatchSchedulerReporter batchSchedulerReporter;
    private ServerScheduleLatencyRecorder serverLatencyRecorder;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
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

        ActiveRequestCounter.RequestToken token = mock(ActiveRequestCounter.RequestToken.class);
        when(activeRequestCounter.acquire()).thenReturn(token);

        service = new FlexlbServiceImpl(
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                activeRequestCounter,
                grpcForwarder,
                configService,
                batchSchedulerReporter,
                serverLatencyRecorder
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
        verify(serverLatencyRecorder).recordArrival(anyLong());
        verify(serverLatencyRecorder).recordCompletion(any(BalanceContext.class), anyLong());
    }

    @Test
    void testSchedule_forwardToMaster_success() {
        // Given: consistency needed, not master, forward succeeds
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB masterResponse = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(true)
                .setCode(200)
                .build();
        when(grpcForwarder.forwardToMaster(any())).thenReturn(masterResponse);

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardToMaster(request);
        verify(routeService, never()).route(any());

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
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

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then
        verify(grpcForwarder).forwardToMaster(request);
        verify(routeService).route(any(BalanceContext.class));

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertTrue(resp.getSuccess());
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
    void testCancel_success() {
        // Given
        FlexlbScheduleProtocol.FlexlbCancelRequestPB request = FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer = mock(StreamObserver.class);

        // When
        service.cancel(request, observer);

        // Then
        verify(routeService).cancelByRequestId(eq(12345L), any(), eq(0L));
        verify(observer).onNext(any(FlexlbScheduleProtocol.FlexlbCancelResponsePB.class));
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
    }

    @Test
    void testCancel_followerForwardsToMaster() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        FlexlbScheduleProtocol.FlexlbCancelRequestPB request =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestId(12346L)
                        .setBatchId(7001L)
                        .build();
        FlexlbScheduleProtocol.FlexlbCancelResponsePB forwarded =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder().setFound(true).build();
        when(grpcForwarder.forwardCancelToMaster(request)).thenReturn(forwarded);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer = mock(StreamObserver.class);

        service.cancel(request, observer);

        verify(grpcForwarder).forwardCancelToMaster(request);
        verify(routeService).cancelByRequestId(12346L, CancelReason.CLIENT_CANCELLED,
                7001L);
        verify(observer).onNext(forwarded);
        verify(observer).onCompleted();
    }

    @Test
    void testCancel_exceptionHandling() {
        // Given
        doThrow(new RuntimeException("cancel error")).when(routeService)
                .cancelByRequestId(eq(12345L), any(), eq(0L));

        FlexlbScheduleProtocol.FlexlbCancelRequestPB request = FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer = mock(StreamObserver.class);

        // When
        service.cancel(request, observer);

        // Then
        verify(routeService).cancelByRequestId(eq(12345L), any(), eq(0L));
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
                        1001L, 10L, 20L, "engine acknowledged batch"));
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

    // ---- gRPC status classification (CANCELLED / DEADLINE_EXCEEDED false-error fix) ----

    @Test
    void grpcStatusCode_recognisesDirectCancelledStatusRuntimeException() {
        // Mirrors the cancellationListener in schedule(): a bare StatusRuntimeException
        // carrying CANCELLED, delivered to whenComplete without wrapping.
        StatusRuntimeException ex =
                Status.CANCELLED.withDescription("gRPC context cancelled").asRuntimeException();
        assertEquals(Status.Code.CANCELLED, FlexlbServiceImpl.grpcStatusCode(ex));
    }

    @Test
    void grpcStatusCode_unwrapsCompletionExceptionToFindDeadlineExceeded() {
        // A route future may fail with an exception wrapped in CompletionException; the
        // classifier must traverse the cause chain to recover the underlying gRPC code.
        StatusRuntimeException inner =
                Status.DEADLINE_EXCEEDED.withDescription("deadline exceeded").asRuntimeException();
        Throwable wrapped = new CompletionException(inner);
        assertEquals(Status.Code.DEADLINE_EXCEEDED, FlexlbServiceImpl.grpcStatusCode(wrapped));
    }

    @Test
    void grpcStatusCode_returnsNullForNonGrpcThrowable() {
        // A plain non-gRPC exception must yield null so it stays on the ERROR path.
        assertNull(FlexlbServiceImpl.grpcStatusCode(new RuntimeException("test error")));
    }

    @Test
    void testSchedule_clientCancellationProducesCancelledErrorCode() {
        // Given: route fails with a gRPC CANCELLED (as raised by the cancellation listener)
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        StatusRuntimeException cancelled =
                Status.CANCELLED.withDescription("gRPC context cancelled").asRuntimeException();
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.failedFuture(cancelled));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        // When
        service.schedule(request, observer);

        // Then: response carries the REQUEST_CANCELLED error code (8504), not a generic 500
        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertFalse(resp.getSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), resp.getCode());
    }

    @Test
    void testSchedule_clientCancellationLogsWarnNotError() {
        // CANCELLED must downgrade to WARN (no throwable) + DEBUG (with throwable), never ERROR.
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        StatusRuntimeException cancelled =
                Status.CANCELLED.withDescription("gRPC context cancelled").asRuntimeException();
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.failedFuture(cancelled));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        try (LogCapture capture = new LogCapture(Level.DEBUG)) {
            service.schedule(request, observer);

            assertTrue(capture.hasEvent(Level.WARN, "client cancelled/timeout"),
                    "CANCELLED should log WARN");
            assertTrue(capture.hasEventWithThrowable(Level.DEBUG),
                    "CANCELLED stack should be available at DEBUG");
            assertFalse(capture.hasEvent(Level.ERROR, null),
                    "CANCELLED must not log ERROR");
        }

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), captor.getValue().getCode());
    }

    @Test
    void testSchedule_nonCancelledGrpcErrorLogsError() {
        // A non-CANCELLED / non-DEADLINE_EXCEEDED gRPC failure must stay on the ERROR path.
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        StatusRuntimeException unknown =
                Status.UNKNOWN.withDescription("engine boom").asRuntimeException();
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.failedFuture(unknown));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12346L)
                .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        try (LogCapture capture = new LogCapture(Level.DEBUG)) {
            service.schedule(request, observer);

            assertTrue(capture.hasEventWithThrowable(Level.ERROR),
                    "Non-cancelled gRPC error should log ERROR with stack");
            assertFalse(capture.hasEvent(Level.WARN, "client cancelled/timeout"),
                    "Non-cancelled gRPC error must not take the cancelled/timeout WARN path");
        }
    }

    @Test
    void testSchedule_deadlineExceededProducesCancelledErrorCodeAndMessage() {
        // Deadline expiry shares the REQUEST_CANCELLED response code but its message must
        // distinguish the timeout from an active client cancel.
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);
        StatusRuntimeException deadlineExceeded =
                Status.DEADLINE_EXCEEDED.withDescription("deadline exceeded").asRuntimeException();
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.failedFuture(deadlineExceeded));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12347L)
                .build();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);

        service.schedule(request, observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer).onNext(captor.capture());
        verify(observer).onCompleted();

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB resp = captor.getValue();
        assertFalse(resp.getSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), resp.getCode());
        assertTrue(resp.getErrorMessage().contains("deadline exceeded"),
                "Message should distinguish deadline expiry, got: " + resp.getErrorMessage());
    }

    /**
     * Captures {@code flexlbLogger} events via a logback {@link ListAppender} for log-level
     * assertions. Uses logback's native API (already on the classpath via spring-boot-starter),
     * so no extra test dependency is required. Restores the previous level on close.
     */
    private static final class LogCapture implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger;
        private final Level prevLevel;
        private final ListAppender<ILoggingEvent> appender = new ListAppender<>();

        LogCapture(Level level) {
            logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
            prevLevel = logger.getLevel();
            logger.setLevel(level);
            appender.start();
            logger.addAppender(appender);
        }

        List<ILoggingEvent> events() {
            return appender.list;
        }

        boolean hasEvent(Level level, String fragment) {
            return events().stream().anyMatch(e -> e.getLevel().equals(level)
                    && (fragment == null || e.getFormattedMessage().contains(fragment)));
        }

        boolean hasEventWithThrowable(Level level) {
            return events().stream().anyMatch(e -> e.getLevel().equals(level)
                    && e.getThrowableProxy() != null);
        }

        @Override
        public void close() {
            logger.detachAppender(appender);
            logger.setLevel(prevLevel);
        }
    }

}
