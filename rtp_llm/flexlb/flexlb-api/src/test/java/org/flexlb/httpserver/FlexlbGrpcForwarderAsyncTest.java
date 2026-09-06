package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.interceptor.GrpcTraceInterceptor;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbGrpcForwarderAsyncTest {

    private static final String MASTER_HTTP_ADDRESS = "10.0.0.2:7001";
    private static final String MASTER_CHANNEL_KEY = "10.0.0.2:7003";

    @Test
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void asyncScheduleTracesActualGrpcSuccessRejectionAndTransportError() throws Exception {
        for (int outcome = 0; outcome < 5; outcome++) {
            AtomicReference<StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB>> response = new AtomicReference<>();
            AtomicReference<SpanContext> serverContext = new AtomicReference<>();
            CountDownLatch received = new CountDownLatch(1);
            try (TraceCapture capture = new TraceCapture();
                 RpcFixture fixture = RpcFixture.start((request, observer) -> {
                     serverContext.set(FlexlbTrace.spanContext(GrpcTraceInterceptor.getOtelContext()));
                     response.set(observer);
                     received.countDown();
                 })) {
                FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, mock(EngineHealthReporter.class));
                CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending;
                try (var scope = traceParent().makeCurrent()) {
                    pending = forwarder.forwardScheduleToMaster(request(901L));
                }
                assertTrue(received.await(3, TimeUnit.SECONDS));
                assertTrue(capture.spans.isEmpty(), "async return must not end the CLIENT span");
                assertEquals("one", serverContext.get().getTraceState().get("vendor"));
                int selected = outcome;
                java.util.concurrent.CompletableFuture.runAsync(() -> {
                    if (selected == 2) {
                        response.get().onError(Status.UNAVAILABLE.asRuntimeException());
                    } else {
                        response.get().onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                                .setSuccess(selected == 0)
                                .setCode(selected == 0 ? 200 : selected == 1 ? 8430 : selected == 3 ? 500 : 8513)
                                .build());
                        response.get().onCompleted();
                    }
                }).join();
                await(pending);
                assertTrue(capture.ended.await(3, TimeUnit.SECONDS));
                assertEquals(2, capture.spans.size());
                SpanData client = capture.client();
                SpanData server = capture.spans.stream().filter(s -> s.getKind()
                        == io.opentelemetry.api.trace.SpanKind.SERVER).findFirst().orElseThrow();
                assertEquals("rtp_llm.flexlb.forward_schedule", client.getName());
                assertEquals("1111111111111111", client.getParentSpanId());
                assertEquals(client.getSpanId(), server.getParentSpanId());
                assertEquals("11111111111111111111111111111111", server.getTraceId());
                assertEquals("901", client.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey("request_id")));
                assertEquals(outcome == 0 ? io.opentelemetry.api.trace.StatusCode.OK : io.opentelemetry.api.trace.StatusCode.ERROR,
                        client.getStatus().getStatusCode());
                assertEquals(outcome == 2 ? "UNAVAILABLE" : "OK", client.getAttributes().get(
                        io.opentelemetry.api.common.AttributeKey.stringKey(FlexlbTrace.RPC_RESPONSE_STATUS_CODE)));
                String[] errorTypes = {null, "FLEXLB_BUSINESS_REJECTED", "UNAVAILABLE",
                        "FLEXLB_INTERNAL_ERROR", "FLEXLB_SCHEDULE_FAILED"};
                assertEquals(errorTypes[outcome], client.getAttributes().get(
                        io.opentelemetry.api.common.AttributeKey.stringKey(FlexlbTrace.ERROR_TYPE)));
                forwarder.shutdown();
            }
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void explicitCancellationEndsForwardClientOnlyOnce() throws Exception {
        CountDownLatch received = new CountDownLatch(1);
        try (TraceCapture capture = new TraceCapture();
             RpcFixture fixture = RpcFixture.start((request, observer) -> received.countDown())) {
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, mock(EngineHealthReporter.class));
            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending;
            try (var scope = traceParent().makeCurrent()) {
                pending = forwarder.forwardScheduleToMaster(request(902L));
            }
            assertTrue(received.await(3, TimeUnit.SECONDS));
            pending.toCompletableFuture().cancel(true);
            pending.toCompletableFuture().cancel(true);
            assertTrue(capture.ended.await(3, TimeUnit.SECONDS));
            assertEquals(2, capture.spans.size());
            assertEquals("CANCELLED", capture.client().getAttributes().get(
                    io.opentelemetry.api.common.AttributeKey.stringKey(FlexlbTrace.RPC_RESPONSE_STATUS_CODE)));
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void compensatingCancelKeepsExplicitTraceOutsideCancelledGrpcContext() throws Exception {
        try (TraceCapture capture = new TraceCapture();
             RpcFixture fixture = RpcFixture.startCancel((request, observer) -> {
                 assertFalse(Context.current().isCancelled());
                 assertEquals("one", FlexlbTrace.spanContext(GrpcTraceInterceptor.getOtelContext()).getTraceState().get("vendor"));
                 observer.onNext(FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder().setFound(true).build());
                 observer.onCompleted();
             })) {
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, mock(EngineHealthReporter.class));
            Context.CancellableContext cancelled = Context.current().withCancellation();
            cancelled.cancel(null);
            var result = cancelled.call(() -> Context.ROOT.call(() -> forwarder.forwardCompensatingCancelToMaster(
                    FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder().setRequestId(903L).build(),
                    MASTER_HTTP_ADDRESS, traceParent()))).toCompletableFuture().get(3, TimeUnit.SECONDS);
            assertTrue(result.response().getFound());
            assertTrue(capture.ended.await(3, TimeUnit.SECONDS));
            assertEquals("rtp_llm.flexlb.cancel", capture.client().getName());
            assertEquals("1111111111111111", capture.client().getParentSpanId());
            assertEquals("11111111111111111111111111111111", capture.client().getTraceId());
            forwarder.shutdown();
        }
    }

    private static io.opentelemetry.context.Context traceParent() {
        return io.opentelemetry.context.Context.root().with(Span.wrap(SpanContext.create(
                "11111111111111111111111111111111", "1111111111111111", TraceFlags.getSampled(),
                TraceState.builder().put("vendor", "one").build())));
    }

    private static final class TraceCapture implements SpanExporter, AutoCloseable {
        final java.util.List<SpanData> spans = new java.util.concurrent.CopyOnWriteArrayList<>();
        final CountDownLatch ended = new CountDownLatch(2);
        final OpenTelemetrySdk sdk;

        TraceCapture() {
            FlexlbTrace.configureEnabled(true);
            GlobalOpenTelemetry.resetForTest();
            sdk = OpenTelemetrySdk.builder().setTracerProvider(SdkTracerProvider.builder()
                    .addSpanProcessor(SimpleSpanProcessor.create(this)).build()).build();
            GlobalOpenTelemetry.set(sdk);
        }

        SpanData client() {
            return spans.stream().filter(s -> s.getKind() == io.opentelemetry.api.trace.SpanKind.CLIENT)
                    .findFirst().orElseThrow();
        }

        public CompletableResultCode export(java.util.Collection<SpanData> batch) {
            spans.addAll(batch);
            batch.forEach(ignored -> ended.countDown());
            return CompletableResultCode.ofSuccess();
        }
        public CompletableResultCode flush() { return CompletableResultCode.ofSuccess(); }
        public CompletableResultCode shutdown() { return CompletableResultCode.ofSuccess(); }
        public void close() {
            FlexlbTrace.configureEnabled(false);
            sdk.close();
            GlobalOpenTelemetry.resetForTest();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void forwardsScheduleAsynchronouslyAndSetsOneHop() throws Exception {
        AtomicReference<FlexlbScheduleProtocol.FlexlbScheduleRequestPB> forwarded =
                new AtomicReference<>();
        try (RpcFixture fixture = RpcFixture.start((request, observer) -> {
            forwarded.set(request);
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true)
                    .setCode(200)
                    .build());
            observer.onCompleted();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(101L)));

            assertNotNull(result.response());
            assertTrue(result.response().getSuccess());
            assertEquals(1, forwarded.get().getForwardHop());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "200");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void unavailableIsTerminalAndDoesNotDiscardTheChannel() throws Exception {
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                observer.onError(Status.UNAVAILABLE
                        .withDescription("leader unavailable")
                        .asRuntimeException()))) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(102L)));

            assertTrue(result.masterFound());
            assertEquals("UNAVAILABLE", result.failure());
            assertFalse(fixture.channel.isShutdown());
            assertTrue(channels(forwarder).containsKey(MASTER_CHANNEL_KEY));
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void inboundCancellationCancelsTheForwardedRpc() throws Exception {
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                masterReceivedRequest.countDown())) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            Context.CancellableContext inbound = Context.current().withCancellation();

            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending =
                    inbound.call(() -> forwarder.forwardScheduleToMaster(request(103L)));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            assertFalse(pending.toCompletableFuture().isDone());

            inbound.cancel(null);
            FlexlbGrpcForwarder.MasterForwardResult result = await(pending);

            assertEquals("CANCELLED", result.failure());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void inboundDeadlineBoundsTheForwardedRpc() throws Exception {
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        ScheduledExecutorService deadlineTimer = Executors.newSingleThreadScheduledExecutor();
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                masterReceivedRequest.countDown())) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            Context.CancellableContext inbound = Context.current()
                    .withDeadlineAfter(500, TimeUnit.MILLISECONDS, deadlineTimer);

            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending =
                    inbound.call(() -> forwarder.forwardScheduleToMaster(request(104L)));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            FlexlbGrpcForwarder.MasterForwardResult result = await(pending);

            assertEquals("DEADLINE_EXCEEDED", result.failure());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
            inbound.cancel(null);
        } finally {
            deadlineTimer.shutdownNow();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void monitoringFailureCannotLoseOrDuplicateTheResponse() throws Exception {
        try (RpcFixture fixture = RpcFixture.start((request, observer) -> {
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true)
                    .setCode(200)
                    .build());
            observer.onCompleted();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            doThrow(new RuntimeException("monitor unavailable"))
                    .when(reporter)
                    .reportForwardToMasterResult("10.0.0.2", "200");
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(105L)));

            assertNotNull(result.response());
            assertTrue(result.response().getSuccess());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "200");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void forwardsCancelWithoutBlockingAndSetsOneHop() throws Exception {
        AtomicReference<FlexlbScheduleProtocol.FlexlbCancelRequestPB> forwarded =
                new AtomicReference<>();
        AtomicReference<StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB>>
                masterObserver = new AtomicReference<>();
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        try (RpcFixture fixture = RpcFixture.startCancel((request, observer) -> {
            forwarded.set(request);
            masterObserver.set(observer);
            masterReceivedRequest.countDown();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            AtomicReference<CompletionStage<FlexlbGrpcForwarder.CancelForwardResult>>
                    pending = new AtomicReference<>();

            assertTimeoutPreemptively(Duration.ofSeconds(1), () ->
                    pending.set(forwarder.forwardCancelToMaster(
                            FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                    .setRequestId(106L)
                                    .build())));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            assertFalse(pending.get().toCompletableFuture().isDone());
            assertEquals(1, forwarded.get().getForwardHop());

            masterObserver.get().onNext(
                    FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                            .setFound(false)
                            .build());
            masterObserver.get().onCompleted();
            FlexlbGrpcForwarder.CancelForwardResult result = pending.get()
                    .toCompletableFuture()
                    .get(5, TimeUnit.SECONDS);

            assertNotNull(result.response());
            assertFalse(result.response().getFound());
            verify(reporter, times(1)).reportForwardToMasterResult(
                    "10.0.0.2", "CANCEL_NOT_FOUND");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void compensatingCancelUsesCapturedMasterAfterElectionChanges() throws Exception {
        AtomicReference<FlexlbScheduleProtocol.FlexlbCancelRequestPB> forwarded =
                new AtomicReference<>();
        try (RpcFixture fixture = RpcFixture.startCancel((request, observer) -> {
            forwarded.set(request);
            observer.onNext(FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                    .setFound(true)
                    .build());
            observer.onCompleted();
        })) {
            FlexlbGrpcForwarder forwarder = forwarder(
                    fixture.channel, mock(EngineHealthReporter.class), "10.0.0.9:7001");

            FlexlbGrpcForwarder.CancelForwardResult result = awaitCancel(
                    forwarder.forwardCompensatingCancelToMaster(
                            cancelRequest(107L), MASTER_HTTP_ADDRESS, 1000L));

            assertNotNull(result.response());
            assertTrue(result.response().getFound());
            assertEquals(MASTER_HTTP_ADDRESS, result.masterHost());
            assertEquals(1, forwarded.get().getForwardHop());
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void compensatingCancelHasIndependentFiniteDeadline() throws Exception {
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        try (RpcFixture fixture = RpcFixture.startCancel((request, observer) ->
                masterReceivedRequest.countDown())) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.CancelForwardResult result = awaitCancel(
                    forwarder.forwardCompensatingCancelToMaster(
                            cancelRequest(108L), MASTER_HTTP_ADDRESS, 100L));

            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            assertEquals("DEADLINE_EXCEEDED", result.failure());
            assertEquals(MASTER_HTTP_ADDRESS, result.masterHost());
            verify(reporter).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
        }
    }

    private static FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest(
            long requestId) {
        return FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(requestId)
                .build();
    }

    private static FlexlbGrpcForwarder.CancelForwardResult awaitCancel(
            CompletionStage<FlexlbGrpcForwarder.CancelForwardResult> result) throws Exception {
        return result.toCompletableFuture().get(5, TimeUnit.SECONDS);
    }

    private static FlexlbGrpcForwarder forwarder(
            ManagedChannel channel,
            EngineHealthReporter reporter) throws Exception {
        return forwarder(channel, reporter, MASTER_HTTP_ADDRESS);
    }

    private static FlexlbGrpcForwarder forwarder(
            ManagedChannel channel,
            EngineHealthReporter reporter,
            String currentMaster) throws Exception {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getMasterHostIpPort()).thenReturn(currentMaster);
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(new org.flexlb.config.FlexlbConfig());
        FlexlbGrpcForwarder forwarder = new FlexlbGrpcForwarder(
                consistency,
                config,
                reporter,
                mock(EventLoopGroup.class),
                mock(Executor.class));
        channels(forwarder).put(MASTER_CHANNEL_KEY, channel);
        return forwarder;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long requestId) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .build();
    }

    private static FlexlbGrpcForwarder.MasterForwardResult await(
            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> result) throws Exception {
        return result.toCompletableFuture().get(5, TimeUnit.SECONDS);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(
            FlexlbGrpcForwarder forwarder) throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }

    @FunctionalInterface
    private interface ScheduleHandler {
        void schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer);
    }

    @FunctionalInterface
    private interface CancelHandler {
        void cancel(
                FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer);
    }

    private static final class RpcFixture implements AutoCloseable {
        private final Server server;
        private final ManagedChannel channel;

        private RpcFixture(Server server, ManagedChannel channel) {
            this.server = server;
            this.channel = channel;
        }

        static RpcFixture start(ScheduleHandler handler) throws Exception {
            return startService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override
                public void schedule(
                        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
                    handler.schedule(request, observer);
                }
            });
        }

        static RpcFixture startCancel(CancelHandler handler) throws Exception {
            return startService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override
                public void cancel(
                        FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer) {
                    handler.cancel(request, observer);
                }
            });
        }

        private static RpcFixture startService(
                FlexlbServiceGrpc.FlexlbServiceImplBase service) throws Exception {
            Server server = NettyServerBuilder.forPort(0)
                    .directExecutor()
                    .intercept(new GrpcTraceInterceptor())
                    .addService(service)
                    .build()
                    .start();
            ManagedChannel channel = NettyChannelBuilder
                    .forAddress("127.0.0.1", server.getPort())
                    .directExecutor()
                    .usePlaintext()
                    .build();
            return new RpcFixture(server, channel);
        }

        @Override
        public void close() throws Exception {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
