package org.flexlb.engine.grpc;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class EngineGrpcClientCompletionTest {
    private final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1, 1, 1, TimeUnit.MINUTES, new LinkedBlockingQueue<>());
    private final GrpcReporter reporter = mock(GrpcReporter.class);
    private TestClient client;
    private Server server;

    @AfterEach
    void close() throws InterruptedException {
        if (client != null) {
            client.shutdown();
        }
        executor.shutdownNow();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void metricsFailureDoesNotReplaceSuccessOrStrandFuture() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new IllegalStateException("metrics backend unavailable"))
                .when(reporter).reportCallMetrics(anyString(), anyLong(), anyBoolean());

        try (var logs = new MetricWarnings()) {
            for (int i = 0; i < 3; i++) {
                assertEquals(EngineRpcService.WorkerStatusPB.getDefaultInstance(), client.getWorkerStatusAsync(
                        "127.0.0.1", 8081, request(), 1000).get(5, TimeUnit.SECONDS));
            }
            assertEquals(1, logs.messages.size(), "repeated metric failures must not flood the log");
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
        var response = new java.util.concurrent.CompletableFuture<StreamObserver<EngineRpcService.WorkerStatusPB>>();
        var calls = new AtomicInteger();
        startServer(observer -> {
            if (retry && calls.incrementAndGet() == 1) {
                observer.onError(Status.UNAVAILABLE.withDescription("Connection reset").asRuntimeException());
            } else {
                response.complete(observer);
            }
        });
        doThrow(new LinkageError("metrics implementation unavailable"))
                .when(reporter).reportCallMetrics(anyString(), anyLong(), anyBoolean());

        var result = client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000);
        succeed(response.get(5, TimeUnit.SECONDS));

        assertEquals(EngineRpcService.WorkerStatusPB.getDefaultInstance(), result.get(5, TimeUnit.SECONDS));
    }

    @Test
    void unreadableMetricFailureDoesNotStrandFuture() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new IllegalStateException() {
            @Override
            public String getMessage() {
                throw new IllegalStateException("exception details unavailable");
            }
        }).when(reporter).reportCallMetrics(anyString(), anyLong(), anyBoolean());

        assertEquals(EngineRpcService.WorkerStatusPB.getDefaultInstance(), client.getWorkerStatusAsync(
                "127.0.0.1", 8081, request(), 1000).get(5, TimeUnit.SECONDS));
    }

    @Test
    void metricsFailureWarningIsBoundedAndSingleLine() throws Exception {
        startServer(EngineGrpcClientCompletionTest::succeed);
        doThrow(new IllegalStateException("bad\nmetrics\r" + "x".repeat(1000)))
                .when(reporter).reportCallMetrics(anyString(), anyLong(), anyBoolean());
        try (var logs = new MetricWarnings()) {
            client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 1000).get(5, TimeUnit.SECONDS);
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
        var response = spy(EngineRpcService.CacheStatusPB.newBuilder()
                .putCacheKeys(1L, true).build());
        var calls = new AtomicInteger();
        String name = InProcessServerBuilder.generateName();
        server = InProcessServerBuilder.forName(name).directExecutor().addService(
                new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                               StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                        if (retry && calls.incrementAndGet() == 1) {
                            observer.onError(Status.UNAVAILABLE.withDescription("Connection reset").asRuntimeException());
                        } else {
                            observer.onNext(response);
                            observer.onCompleted();
                        }
                    }
                }).build().start();
        client = new TestClient(name, executor, reporter);

        assertEquals(response, client.getCacheStatusAsync("127.0.0.1", 8081,
                EngineRpcService.CacheVersionPB.getDefaultInstance(), 5000).get(5, TimeUnit.SECONDS));
        verify(response, never()).getSerializedSize();
        verify(reporter).reportCallMetrics(
                eq("GetCacheStatus"),
                anyLong(), eq(retry));
    }

    @Test
    void deadlineFailureCompletesFutureWithoutRetry() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(observer -> calls.incrementAndGet());

        ExecutionException failure = assertThrows(ExecutionException.class, () ->
                client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 40)
                        .get(5, TimeUnit.SECONDS));
        assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure.getCause()).getCode());
        assertTrue(calls.get() <= 1);
    }

    @Test
    void connectionMetricFailureIsRecordedAndDoesNotPreventRetry() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(observer -> {
            if (calls.incrementAndGet() == 1) {
                observer.onError(Status.UNAVAILABLE.withDescription("Connection reset").asRuntimeException());
            } else {
                succeed(observer);
            }
        });

        doThrow(new IllegalStateException("connection metrics unavailable"))
                .when(reporter).reportConnectionDuration(anyString(), anyLong());
        try (var logs = new MetricWarnings()) {
            client.getWorkerStatusAsync("127.0.0.1", 8081, request(), 5000).get(5, TimeUnit.SECONDS);
            assertEquals(2, calls.get());
            assertEquals(1, logs.messages.size());
            assertTrue(logs.messages.element().contains("stage=CONNECTION_DURATION"));
            assertTrue(logs.messages.element().contains("connection metrics unavailable"));
        }
    }

    @Test
    void failedRetryIsTerminal() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        startServer(observer -> {
            calls.incrementAndGet();
            observer.onError(Status.UNAVAILABLE.withDescription("Connection reset").asRuntimeException());
        });

        assertThrows(ExecutionException.class, () -> client.getWorkerStatusAsync(
                "127.0.0.1", 8081, request(), 5000).get(5, TimeUnit.SECONDS));
        assertEquals(2, calls.get());
    }

    private void startServer(Consumer<StreamObserver<EngineRpcService.WorkerStatusPB>> handler) throws Exception {
        String name = InProcessServerBuilder.generateName();
        server = InProcessServerBuilder.forName(name).directExecutor().addService(
                new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                                StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                        handler.accept(observer);
                    }
                }).build().start();
        client = new TestClient(name, executor, reporter);
    }

    private static EngineRpcService.StatusVersionPB request() {
        return EngineRpcService.StatusVersionPB.getDefaultInstance();
    }

    private static void succeed(StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
        observer.onNext(EngineRpcService.WorkerStatusPB.getDefaultInstance());
        observer.onCompleted();
    }

    private static final class MetricWarnings implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger = (ch.qos.logback.classic.Logger)
                org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        private final ConcurrentLinkedQueue<String> messages = new ConcurrentLinkedQueue<>();
        private final AppenderBase<ILoggingEvent> appender = new AppenderBase<>() {
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
        private TestClient(String name, ThreadPoolExecutor executor, GrpcReporter reporter) {
            super(mock(EngineAddressResolver.class), new GrpcChannelFactory(executor, mock(EventLoopGroup.class)) {
                @Override
                public ManagedChannel create(GrpcTarget target) {
                    return InProcessChannelBuilder.forName(name).executor(executor).build();
                }
            }, reporter, mock(CacheMatchConfiguration.class));
        }
    }
}
