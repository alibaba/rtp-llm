package org.flexlb.engine.grpc;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.handler.codec.http2.Http2Error;
import io.netty.handler.codec.http2.Http2Exception;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.net.SocketException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineGrpcClientCompletionTest {
    private final ThreadPoolExecutor executor =
            new ThreadPoolExecutor(1, 1, 1, TimeUnit.MINUTES, new LinkedBlockingQueue<>());
    private final GrpcReporter reporter = mock(GrpcReporter.class);
    private TestClient client;
    private Server server;

    @AfterEach
    void close() throws InterruptedException {
        if (client != null) {
            client.shutdownChannelPool();
        }
        executor.shutdownNow();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = Status.Code.class,
            names = {"INTERNAL", "RESOURCE_EXHAUSTED", "UNAVAILABLE"})
    void ordinaryGrpcErrorsReachCallerUnchangedWithoutRetry(Status.Code code) throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(
                observer -> {
                    calls.incrementAndGet();
                    observer.onError(Status.fromCode(code).asRuntimeException());
                });
        ExecutionException failure =
                assertThrows(
                        ExecutionException.class,
                        () ->
                                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 1000)
                                        .get(5, TimeUnit.SECONDS));
        var rpcFailure = assertInstanceOf(StatusRuntimeException.class, failure.getCause());
        assertEquals(code, rpcFailure.getStatus().getCode());
        assertEquals(GrpcFailure.Kind.RPC_FAILURE, GrpcFailure.classify(rpcFailure));
        assertEquals(1, calls.get());
        verify(reporter)
                .reportCallFailure(
                        eq("worker"),
                        eq(false),
                        eq("CALL_RESULT"),
                        argThat(
                                detail ->
                                        detail.kind() == GrpcFailure.Kind.RPC_FAILURE
                                                && detail.grpcCode() == code),
                        any());
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"Connection reset", "Broken pipe"})
    void recoveredTransportFailureOnlyReportsSuccess(String message) throws Exception {
        AtomicInteger calls = new AtomicInteger();
        SocketException cause = new SocketException(message);
        startServer(
                observer -> {
                    if (calls.incrementAndGet() == 1) {
                        observer.onError(Status.UNAVAILABLE.withCause(cause).asRuntimeException());
                    } else {
                        succeed(observer);
                    }
                });

        try (var logs = new RpcFailureLogs()) {
            assertEquals(
                    EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                    client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                            .get(5, TimeUnit.SECONDS));
            assertTrue(logs.events.isEmpty(), "a recovered RPC must not emit a failure log");
        }

        assertEquals(2, calls.get());
        verify(reporter, never())
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
        verify(reporter).reportCallMetrics(eq("GetWorkerStatus"), anyLong(), eq(true));
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void failureDiagnosticsCannotReplaceFinalRpcFailure(boolean memoryError) throws Exception {
        AtomicInteger calls = new AtomicInteger();
        SocketException lastCause = new SocketException("Broken pipe");
        startServer(
                observer -> {
                    if (calls.incrementAndGet() == 1) {
                        observer.onError(transportFailure());
                    } else {
                        observer.onError(
                                Status.UNAVAILABLE.withCause(lastCause).asRuntimeException());
                    }
                });
        Throwable diagnosticFailure =
                memoryError
                        ? new OutOfMemoryError("metrics allocation failed")
                        : new IllegalStateException("metrics backend failed");
        doThrow(diagnosticFailure)
                .when(reporter)
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());

        try (var logs = new MetricWarnings()) {
            ExecutionException failure =
                    assertThrows(
                            ExecutionException.class,
                            () ->
                                    client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                                            .get(5, TimeUnit.SECONDS));
            assertSame(lastCause, failure.getCause().getCause());
            assertEquals(1, failure.getCause().getSuppressed().length);
            assertEquals(2, calls.get());
            verify(reporter, times(1))
                    .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
            assertEquals(1, logs.messages.size());
            assertTrue(logs.messages.element().contains("stage=CALL_FAILURE"));
        }
    }

    @Test
    void deliveryMetricAllocationFailureCannotChangeSuccessfulRpcOutcome() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new OutOfMemoryError("delivery metric allocation failed"))
                .when(reporter)
                .reportDeliveryDelay(anyString(), anyBoolean(), any());

        assertEquals(
                EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                        .get(5, TimeUnit.SECONDS));
        verify(reporter, never())
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
    }

    @Test
    void metricsFailureDoesNotReplaceSuccessOrStrandFuture() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new IllegalStateException("metrics backend unavailable"))
                .when(reporter)
                .reportCallMetrics(anyString(), anyLong(), anyBoolean());

        try (var logs = new MetricWarnings()) {
            for (int i = 0; i < 3; i++) {
                assertEquals(
                        EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                        client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 1000)
                                .get(5, TimeUnit.SECONDS));
            }
            assertEquals(
                    1, logs.messages.size(), "repeated metric failures must not flood the log");
            String warning = logs.messages.element();
            assertTrue(warning.contains("stage=CALL_METRICS"));
            assertTrue(warning.contains("endpoint=127.0.0.1:8081"));
            assertTrue(warning.contains("errorType=IllegalStateException"));
            assertTrue(warning.contains("metrics backend unavailable"));
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void metricsErrorCannotStrandAnAcceptedResponse(boolean retry) throws Exception {
        var response =
                new java.util.concurrent.CompletableFuture<
                        StreamObserver<EngineRpcService.WorkerStatusPB>>();
        var calls = new AtomicInteger();
        startServer(
                observer -> {
                    if (retry && calls.incrementAndGet() == 1) {
                        observer.onError(transportFailure());
                    } else {
                        response.complete(observer);
                    }
                });
        doThrow(new LinkageError("metrics implementation unavailable"))
                .when(reporter)
                .reportCallMetrics(anyString(), anyLong(), anyBoolean());

        var result = client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000);
        succeed(response.get(5, TimeUnit.SECONDS));

        assertEquals(
                EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                result.get(5, TimeUnit.SECONDS));
    }

    @Test
    void unreadableMetricFailureDoesNotStrandFuture() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(
                        new IllegalStateException() {
                            @Override
                            public String getMessage() {
                                throw new IllegalStateException("exception details unavailable");
                            }
                        })
                .when(reporter)
                .reportCallMetrics(anyString(), anyLong(), anyBoolean());

        assertEquals(
                EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 1000)
                        .get(5, TimeUnit.SECONDS));
    }

    @Test
    void metricsFailureWarningIsBoundedAndSingleLine() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new IllegalStateException("bad\nmetrics\r" + "x".repeat(1000)))
                .when(reporter)
                .reportCallMetrics(anyString(), anyLong(), anyBoolean());
        try (var logs = new MetricWarnings()) {
            client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 1000)
                    .get(5, TimeUnit.SECONDS);
            assertEquals(1, logs.messages.size());
            String warning = logs.messages.element();
            assertTrue(warning.contains("stage=CALL_METRICS"));
            assertTrue(warning.contains("error=bad metrics "));
            assertTrue(warning.length() < 500);
            assertTrue(!warning.contains("\n") && !warning.contains("\r"));
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cacheResponseMetricsDoNotCalculateSerializedSize(boolean retry) throws Exception {
        var response =
                spy(EngineRpcService.CacheStatusPB.newBuilder().putCacheKeys(1L, true).build());
        var calls = new AtomicInteger();
        String name = InProcessServerBuilder.generateName();
        server =
                InProcessServerBuilder.forName(name)
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getCacheStatus(
                                            EngineRpcService.CacheVersionPB request,
                                            StreamObserver<EngineRpcService.CacheStatusPB>
                                                    observer) {
                                        if (retry && calls.incrementAndGet() == 1) {
                                            observer.onError(transportFailure());
                                        } else {
                                            observer.onNext(response);
                                            observer.onCompleted();
                                        }
                                    }
                                })
                        .build()
                        .start();
        client = new TestClient(name, executor, reporter);

        assertEquals(
                response,
                client.getCacheStatusAsync(
                                "127.0.0.1",
                                8081,
                                EngineRpcService.CacheVersionPB.getDefaultInstance(),
                                5000)
                        .get(5, TimeUnit.SECONDS));
        verify(response, never()).getSerializedSize();
        verify(reporter)
                .reportCallMetrics(
                        eq(AbstractGrpcClient.ServiceType.CACHE_STATUS.getOperationName()),
                        anyLong(),
                        eq(retry));
    }

    @Test
    void deadlineFailureCompletesFutureWithoutRetry() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(observer -> calls.incrementAndGet());

        ExecutionException failure =
                assertThrows(
                        ExecutionException.class,
                        () ->
                                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 40)
                                        .get(5, TimeUnit.SECONDS));
        assertEquals(
                Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure.getCause()).getCode());
        assertTrue(calls.get() <= 1);
    }

    @Test
    void connectionMetricFailureIsRecordedAndDoesNotPreventRetry() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(
                observer -> {
                    if (calls.incrementAndGet() == 1) {
                        observer.onError(transportFailure());
                    } else {
                        succeed(observer);
                    }
                });

        doThrow(new IllegalStateException("connection metrics unavailable"))
                .when(reporter)
                .reportConnectionDuration(anyString(), anyLong());
        try (var logs = new MetricWarnings()) {
            client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                    .get(5, TimeUnit.SECONDS);
            assertEquals(2, calls.get());
            assertEquals(1, logs.messages.size());
            assertTrue(logs.messages.element().contains("stage=CONNECTION_DURATION"));
            assertTrue(logs.messages.element().contains("connection metrics unavailable"));
        }
    }

    @Test
    void failedRetryIsTerminal() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        SocketException firstCause = new SocketException("Connection reset");
        SocketException lastCause = new SocketException("Broken pipe");
        startServer(
                observer -> {
                    observer.onError(
                            Status.UNAVAILABLE
                                    .withCause(
                                            calls.incrementAndGet() == 1 ? firstCause : lastCause)
                                    .asRuntimeException());
                });

        try (var logs = new RpcFailureLogs()) {
            ExecutionException failure =
                    assertThrows(
                            ExecutionException.class,
                            () ->
                                    client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                                            .get(5, TimeUnit.SECONDS));
            assertEquals(2, calls.get());
            assertSame(lastCause, failure.getCause().getCause());
            assertEquals(1, failure.getCause().getSuppressed().length);
            assertSame(firstCause, failure.getCause().getSuppressed()[0].getCause());
            assertEquals(1, logs.events.size());
            var loggedFailure = logs.events.element().getThrowableProxy();
            assertEquals("Broken pipe", loggedFailure.getCause().getMessage());
            assertEquals(1, loggedFailure.getSuppressed().length);
            assertEquals(
                    "Connection reset", loggedFailure.getSuppressed()[0].getCause().getMessage());
        }
        verifyFinalFailure(true, "CALL_RESULT", GrpcFailure.Kind.BROKEN_PIPE);
        verify(reporter, times(1))
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
    }

    @Test
    void initialChannelCreationErrorCompletesTheReturnedFuture() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        OutOfMemoryError localFailure = new OutOfMemoryError("direct allocation failed");
        client.failChannelCreation(1, localFailure);

        CompletableFuture<?> result =
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000);

        ExecutionException failure =
                assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(localFailure, failure.getCause());
        verifyFinalFailure(false, "CHANNEL_SETUP", GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED);
    }

    @Test
    void synchronousCallStartRejectionIsReportedAtItsActualBoundary() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        RejectedExecutionException localFailure =
                new RejectedExecutionException("client executor rejected");
        client.callStartFailure = localFailure;

        ExecutionException failure =
                assertThrows(
                        ExecutionException.class,
                        () ->
                                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                                        .get(5, TimeUnit.SECONDS));

        assertSame(localFailure, failure.getCause());
        verifyFinalFailure(false, "CALL_START", GrpcFailure.Kind.LOCAL_EXECUTOR_REJECTED);
        assertEquals(1, client.channelCreations.get());
    }

    @Test
    void asynchronousRetryCreationErrorCompletesFutureAndPreservesFirstFailure() throws Exception {
        var response = new CompletableFuture<StreamObserver<EngineRpcService.WorkerStatusPB>>();
        startServer(response::complete);
        OutOfMemoryError localFailure = new OutOfMemoryError("direct allocation failed");
        client.failChannelCreation(2, localFailure);
        SocketException transportFailure = new SocketException("first attempt failed");

        CompletableFuture<?> result =
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000);
        response.get(5, TimeUnit.SECONDS)
                .onError(Status.UNAVAILABLE.withCause(transportFailure).asRuntimeException());

        ExecutionException failure =
                assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(localFailure, failure.getCause());
        assertEquals(2, client.channelCreations.get());
        assertEquals(1, localFailure.getSuppressed().length);
        assertSame(transportFailure, localFailure.getSuppressed()[0].getCause());
        verifyFinalFailure(true, "CHANNEL_SETUP", GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED);
        verify(reporter, times(1))
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
    }

    @Test
    void http2FailureCausedByLocalOutOfMemoryIsNotRetried() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        OutOfMemoryError localFailure = new OutOfMemoryError("direct allocation failed");
        Http2Exception http2Failure =
                new Http2Exception(
                        Http2Error.INTERNAL_ERROR, "opaque transport error", localFailure);
        startServer(
                observer -> {
                    calls.incrementAndGet();
                    observer.onError(Status.INTERNAL.withCause(http2Failure).asRuntimeException());
                });

        ExecutionException failure =
                assertThrows(
                        ExecutionException.class,
                        () ->
                                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                                        .get(5, TimeUnit.SECONDS));

        assertSame(http2Failure, failure.getCause().getCause());
        assertSame(localFailure, failure.getCause().getCause().getCause());
        assertEquals(1, calls.get());
        assertEquals(1, client.channelCreations.get());
    }

    @Test
    void connectionResetTextWithoutTransportCauseIsNotRetried() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(
                observer -> {
                    calls.incrementAndGet();
                    observer.onError(
                            Status.UNAVAILABLE
                                    .withDescription("Connection reset; http2 exception")
                                    .asRuntimeException());
                });

        assertThrows(
                ExecutionException.class,
                () ->
                        client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                                .get(5, TimeUnit.SECONDS));

        assertEquals(1, calls.get());
        verifyFinalFailure(false, "CALL_RESULT", GrpcFailure.Kind.RPC_FAILURE);
    }

    @ParameterizedTest
    @EnumSource(
            value = AbstractGrpcClient.ServiceType.class,
            names = {"BATCH_ENQUEUE", "ENGINE_CANCEL"})
    void ambiguousMutationsReportFailureWithoutReplay(AbstractGrpcClient.ServiceType serviceType)
            throws Exception {
        AtomicInteger calls = new AtomicInteger();
        String name = InProcessServerBuilder.generateName();
        server =
                InProcessServerBuilder.forName(name)
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void enqueueBatch(
                                            EngineRpcService.EnqueueBatchRequestPB request,
                                            StreamObserver<EngineRpcService.EnqueueBatchResponsePB>
                                                    observer) {
                                        calls.incrementAndGet();
                                        observer.onError(transportFailure());
                                    }

                                    @Override
                                    public void cancel(
                                            EngineRpcService.CancelRequestPB request,
                                            StreamObserver<EngineRpcService.CancelResponsePB>
                                                    observer) {
                                        calls.incrementAndGet();
                                        observer.onError(transportFailure());
                                    }
                                })
                        .build()
                        .start();
        client = new TestClient(name, executor, reporter);

        CompletableFuture<?> result =
                serviceType == AbstractGrpcClient.ServiceType.BATCH_ENQUEUE
                        ? client.batchEnqueueAsync(
                                "127.0.0.1",
                                8081,
                                EngineRpcService.EnqueueBatchRequestPB.getDefaultInstance(),
                                5000)
                        : client.cancelAsync(
                                "127.0.0.1",
                                8081,
                                EngineRpcService.CancelRequestPB.newBuilder()
                                        .setReason(
                                                EngineRpcService.RequestCancelReasonPB
                                                        .REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED)
                                        .build(),
                                5000);
        assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));

        assertEquals(1, calls.get());
        assertEquals(1, client.channelCreations.get());
        verify(reporter)
                .reportCallFailure(
                        eq(serviceType.getSuffix()),
                        eq(false),
                        eq("CALL_RESULT"),
                        argThat(detail -> detail.kind() == GrpcFailure.Kind.SOCKET_FAILURE),
                        any());
    }

    @Test
    void connectionMetricErrorDoesNotStrandTheRetry() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(
                observer -> {
                    if (calls.incrementAndGet() == 1) {
                        observer.onError(transportFailure());
                    } else {
                        succeed(observer);
                    }
                });
        doThrow(new OutOfMemoryError("metric allocation failed"))
                .when(reporter)
                .reportConnectionDuration(anyString(), anyLong());

        assertEquals(
                EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                        .get(5, TimeUnit.SECONDS));
        assertEquals(2, calls.get());
    }

    @Test
    void loggingErrorDoesNotPreventAnRpcFromCompleting() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        var logger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        Level previousLevel = logger.getLevel();
        AppenderBase<ILoggingEvent> failingAppender =
                new AppenderBase<>() {
                    @Override
                    protected void append(ILoggingEvent event) {
                        if (event.getLevel() == Level.DEBUG) {
                            throw new OutOfMemoryError("log allocation failed");
                        }
                    }
                };
        failingAppender.start();
        logger.addAppender(failingAppender);
        logger.setLevel(Level.DEBUG);
        try {
            assertEquals(
                    EngineRpcService.WorkerStatusPB.getDefaultInstance(),
                    client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000)
                            .get(5, TimeUnit.SECONDS));
        } finally {
            logger.detachAppender(failingAppender);
            logger.setLevel(previousLevel);
            failingAppender.stop();
        }
    }

    private static StatusRuntimeException transportFailure() {
        return Status.UNAVAILABLE
                .withCause(new SocketException("transport failed"))
                .asRuntimeException();
    }

    private void verifyFinalFailure(boolean retry, String phase, GrpcFailure.Kind kind) {
        verify(reporter)
                .reportCallFailure(
                        eq("worker"),
                        eq(retry),
                        eq(phase),
                        argThat(detail -> detail.kind() == kind),
                        any());
    }

    private void startServer(Consumer<StreamObserver<EngineRpcService.WorkerStatusPB>> handler)
            throws Exception {
        String name = InProcessServerBuilder.generateName();
        server =
                InProcessServerBuilder.forName(name)
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getWorkerStatus(
                                            EngineRpcService.StatusVersionPB request,
                                            StreamObserver<EngineRpcService.WorkerStatusPB>
                                                    observer) {
                                        handler.accept(observer);
                                    }
                                })
                        .build()
                        .start();
        client = new TestClient(name, executor, reporter);
    }

    private static EngineRpcService.StatusVersionPB request() {
        return EngineRpcService.StatusVersionPB.getDefaultInstance();
    }

    private static void succeed(StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
        observer.onNext(EngineRpcService.WorkerStatusPB.getDefaultInstance());
        observer.onCompleted();
    }

    private static final class RpcFailureLogs implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        private final Level previousLevel = logger.getLevel();
        private final ConcurrentLinkedQueue<ILoggingEvent> events = new ConcurrentLinkedQueue<>();
        private final AppenderBase<ILoggingEvent> appender =
                new AppenderBase<>() {
                    @Override
                    protected void append(ILoggingEvent event) {
                        if (event.getMessage().startsWith("event=engine_grpc_call_failed ")) {
                            events.add(event);
                        }
                    }
                };

        private RpcFailureLogs() {
            appender.start();
            logger.addAppender(appender);
            logger.setLevel(Level.DEBUG);
        }

        @Override
        public void close() {
            logger.detachAppender(appender);
            logger.setLevel(previousLevel);
            appender.stop();
        }
    }

    private static final class MetricWarnings implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        private final ConcurrentLinkedQueue<String> messages = new ConcurrentLinkedQueue<>();
        private final AppenderBase<ILoggingEvent> appender =
                new AppenderBase<>() {
                    @Override
                    protected void append(ILoggingEvent event) {
                        if (event.getLevel() == Level.WARN
                                && event.getMessage().startsWith("event=grpc_metrics_failed ")) {
                            messages.add(event.getFormattedMessage());
                        }
                    }
                };

        private MetricWarnings() {
            appender.start();
            logger.addAppender(appender);
        }

        @Override
        public void close() {
            logger.detachAppender(appender);
            appender.stop();
        }
    }

    private static final class TestClient extends EngineGrpcClient {
        private final String name;
        private final AtomicInteger channelCreations = new AtomicInteger();
        private int failingChannelCreation;
        private Error channelCreationFailure;
        private RuntimeException callStartFailure;

        private TestClient(String name, ThreadPoolExecutor executor, GrpcReporter reporter) {
            super(
                    mock(CustomNameResolver.class),
                    executor,
                    mock(EventLoopGroup.class),
                    reporter,
                    20,
                    5000);
            this.name = name;
        }

        private void failChannelCreation(int attempt, Error failure) {
            failingChannelCreation = attempt;
            channelCreationFailure = failure;
        }

        @Override
        protected ManagedChannel createChannel(String channelKey) {
            if (channelCreations.incrementAndGet() == failingChannelCreation) {
                throw channelCreationFailure;
            }
            if (callStartFailure != null) {
                ManagedChannel channel = mock(ManagedChannel.class);
                when(channel.newCall(any(), any())).thenThrow(callStartFailure);
                return channel;
            }
            // Test-only injection of transport causes; real remote gRPC errors never transmit Java
            // causes.
            return InProcessChannelBuilder.forName(name)
                    .executor(getExecutor())
                    .propagateCauseWithStatus(true)
                    .intercept(GrpcCallTrace.INTERCEPTOR)
                    .build();
        }
    }
}
