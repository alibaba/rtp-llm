package org.flexlb.engine.grpc;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.constant.MetricConstant.GRPC_CALL_COUNT;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_CONNECTION_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_RESPONSE_SIZE;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class EngineGrpcClientMonitoringTest {
    @ParameterizedTest
    @CsvSource({"true,false", "false,false", "true,true", "false,true"})
    void monitoringFailureDoesNotStrandSuccessfulRpc(boolean immediateReply, boolean monitorError) throws Exception {
        FlexMonitor monitor = failingMonitor(monitorError);
        var expected = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(51L).build();
        var pending = new AtomicReference<StreamObserver<EngineRpcService.EnqueueBatchResponsePB>>();
        var service = new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                     StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
                if (immediateReply) {
                    observer.onNext(expected);
                    observer.onCompleted();
                } else {
                    pending.set(observer);
                }
            }
        };
        try (var fixture = new Fixture(monitor, service)) {
            var result = fixture.client.batchEnqueueAsync("127.0.0.1", 8081,
                    EngineRpcService.EnqueueBatchRequestPB.getDefaultInstance());
            if (!immediateReply) {
                assertFalse(result.isDone());
                assertNotNull(pending.get());
                pending.get().onNext(expected);
                pending.get().onCompleted();
            }
            assertEquals(expected, assertDoesNotThrow(() -> result.get(2, TimeUnit.SECONDS),
                    "monitor failure must not leave a successful RPC pending"));
            verify(monitor).report(eq(GRPC_CALL_COUNT), any(FlexMetricTags.class), eq(1.0));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void monitoringFailureDoesNotPreventConnectionRetry(boolean monitorError) throws Exception {
        FlexMonitor monitor = failingMonitor(monitorError);
        AtomicInteger attempts = new AtomicInteger();
        var expected = EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).setStatusVersion(7L).build();
        var service = statusService(attempts, expected, null);
        try (var fixture = new Fixture(monitor, service)) {
            var result = fixture.client.getWorkerStatusAsync("127.0.0.1", 8081,
                    EngineRpcService.StatusVersionPB.getDefaultInstance(), 5000L);
            assertEquals(expected, assertDoesNotThrow(() -> result.get(2, TimeUnit.SECONDS),
                    "connection metrics must not block retry or completion"));
            assertEquals(2, attempts.get());
            verify(monitor).report(eq(GRPC_CONNECTION_DURATION), any(FlexMetricTags.class), anyDouble());
            verify(monitor).report(eq(GRPC_CALL_COUNT), any(FlexMetricTags.class), eq(1.0));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void monitoringFailurePreservesRetryFailure(boolean monitorError) throws Exception {
        AtomicInteger attempts = new AtomicInteger();
        var service = statusService(attempts, null, Status.PERMISSION_DENIED.withDescription("engine rejected request"));
        try (var fixture = new Fixture(failingMonitor(monitorError), service)) {
            var result = fixture.client.getWorkerStatusAsync("127.0.0.1", 8081,
                    EngineRpcService.StatusVersionPB.getDefaultInstance(), 5000L);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(Status.Code.PERMISSION_DENIED, Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(2, attempts.get());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationAndChannelMetricsCannotInterruptCallers(boolean monitorError) {
        FlexMonitor monitor = failingMonitor(monitorError);
        doThrow(monitorFailure(monitorError)).when(monitor).register(anyString(), any(FlexMetricType.class));
        doThrow(monitorFailure(monitorError)).when(monitor)
                .register(anyString(), any(FlexMetricType.class), any(FlexPriorityType.class));
        GrpcReporter reporter = new GrpcReporter(monitor);
        assertDoesNotThrow(reporter::init);
        verify(monitor, times(2)).register(anyString(), any(FlexMetricType.class));
        verify(monitor, times(3)).register(anyString(), any(FlexMetricType.class), any(FlexPriorityType.class));
        assertDoesNotThrow(() -> reporter.reportChannelPoolSize(3));
    }

    @Test
    void oneFailedMetricDoesNotSkipRemainingCallMetrics() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        doThrow(new IllegalStateException("metric backend unavailable")).when(monitor)
                .report(eq(GRPC_CALL_DURATION), any(FlexMetricTags.class), anyDouble());
        assertDoesNotThrow(() -> new GrpcReporter(monitor).reportCallMetrics("EnqueueBatch", 10L, 20, false));
        verify(monitor).report(eq(GRPC_RESPONSE_SIZE), any(FlexMetricTags.class), eq(20.0));
        verify(monitor).report(eq(GRPC_CALL_COUNT), any(FlexMetricTags.class), eq(1.0));
    }

    private static FlexMonitor failingMonitor(boolean monitorError) {
        FlexMonitor monitor = mock(FlexMonitor.class);
        doThrow(monitorFailure(monitorError)).when(monitor).report(anyString(), any(), anyDouble());
        return monitor;
    }

    private static Throwable monitorFailure(boolean monitorError) {
        return monitorError ? new AssertionError("metric backend failure")
                : new IllegalStateException("metric backend failure");
    }

    private static RpcServiceGrpc.RpcServiceImplBase statusService(AtomicInteger attempts,
            EngineRpcService.WorkerStatusPB response, Status retryFailure) {
        return new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                        StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                if (attempts.incrementAndGet() == 1) {
                    observer.onError(Status.UNAVAILABLE.withDescription("Connection reset").asRuntimeException());
                } else if (retryFailure != null) {
                    observer.onError(retryFailure.asRuntimeException());
                } else {
                    observer.onNext(response);
                    observer.onCompleted();
                }
            }
        };
    }

    private static final class Fixture implements AutoCloseable {
        private final Server server;
        private final EngineGrpcClient client;

        private Fixture(FlexMonitor monitor, RpcServiceGrpc.RpcServiceImplBase service) throws Exception {
            String serverName = InProcessServerBuilder.generateName();
            server = InProcessServerBuilder.forName(serverName).directExecutor().addService(service).build().start();
            client = new EngineGrpcClient(mock(CustomNameResolver.class), mock(ThreadPoolExecutor.class),
                    mock(EventLoopGroup.class), new GrpcReporter(monitor), 20, 5000L) {
                @Override
                protected ManagedChannel createChannel(String channelKey) {
                    return InProcessChannelBuilder.forName(serverName).directExecutor().build();
                }
            };
        }

        @Override
        public void close() throws Exception {
            try {
                client.shutdownChannelPool();
            } finally {
                server.shutdownNow();
                server.awaitTermination(2, TimeUnit.SECONDS);
            }
        }
    }
}
