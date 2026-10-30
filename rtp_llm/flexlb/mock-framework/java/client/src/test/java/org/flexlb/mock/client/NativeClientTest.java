package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.google.protobuf.ByteString;
import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;

class NativeClientTest {
    static GenerateInputPB input(long id) {
        return GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2, 3))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setCanUsePdSeparation(true).setReturnIncremental(true)).build();
    }
    static FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setSeqLen(3).setMaxNewTokens(3).setPriority(20)
            .setGenerateInput(input(id).toByteString()).build();
    }
    static GenerateOutputsPB token(long id, int count, boolean finished) {
        return GenerateOutputsPB.newBuilder().setRequestId(id).setFlattenOutput(FlattenOutputPB.newBuilder()
            .addFinished(finished).addAuxInfo(AuxInfoPB.newBuilder().setOutputLen(count).setStepOutputLen(1).setFirstTokenCostTimeUs(1))
            .setOutputIds(TensorPB.newBuilder().setDataType(TensorPB.DataType.INT32).addShape(1).addShape(1).addShape(1)
                .setInt32Data(ByteString.copyFrom(new byte[]{1, 0, 0, 0})))).build();
    }
    static final class Harness implements AutoCloseable {
        final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
        final BlockingQueue<FlexlbCancelRequestPB> cancellations = new LinkedBlockingQueue<>();
        final CountDownLatch scheduleEntered = new CountDownLatch(1), scheduleCancelled = new CountDownLatch(1), streamCancelled = new CountDownLatch(1);
        final AtomicInteger generates = new AtomicInteger(), fetches = new AtomicInteger(), schedules = new AtomicInteger();
        final Server engine, master;
        final com.sun.net.httpserver.HttpServer heartbeat;
        volatile String leader;
        volatile int heartbeatStatus = 200;
        String http() { return "http://127.0.0.1:" + heartbeat.getAddress().getPort(); }
        Map<String, String> endpoint() { return Map.of("role", "master", "rpc", "127.0.0.1:" + master.getPort(), "http", http()); }
        final NativeClient client;
        volatile Status scheduleFailure;
        volatile boolean batched;
        volatile int scheduleDelay, tokenDelay;
        volatile Function<FlexlbScheduleResponsePB, FlexlbScheduleResponsePB> route = Function.identity();
        volatile BiConsumer<Long, StreamObserver<GenerateOutputsPB>> output = (id, response) -> {
            response.onNext(token(id, 1, false)); response.onNext(token(id, 2, false)); response.onNext(token(id, 3, true)); response.onCompleted();
        };
        final AtomicReference<GenerateInputPB> generated = new AtomicReference<>();
        final AtomicBoolean engineSawDeadline = new AtomicBoolean();
        private static com.sun.net.httpserver.HttpServer heartbeatListener() throws Exception {
            for (int i = 0; i < 32; i++) {
                var http = com.sun.net.httpserver.HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
                int rpc = http.getAddress().getPort() + 2;
                if (rpc <= 65535) try (var probe = new ServerSocket()) {
                    probe.setReuseAddress(true);
                    probe.bind(new InetSocketAddress("127.0.0.1", rpc));
                    return http;
                } catch (BindException occupied) { /* Try another native HTTP/gRPC pair. */ }
                http.stop(0);
            }
            throw new IllegalStateException("No free test Master port pair");
        }
        Harness() throws Exception {
            engine = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(new RpcServiceGrpc.RpcServiceImplBase() {
                private void stream(long id, StreamObserver<GenerateOutputsPB> response) {
                    engineSawDeadline.set(Context.current().getDeadline() != null);
                    Context.current().addListener(c -> streamCancelled.countDown(), Runnable::run);
                    workers.submit(() -> { try { Thread.sleep(tokenDelay); output.accept(id, response); }
                        catch (InterruptedException cancelled) { Thread.currentThread().interrupt(); } });
                }
                @Override public void generateStreamCall(GenerateInputPB input, StreamObserver<GenerateOutputsPB> response) {
                    generates.incrementAndGet(); generated.set(input); stream(input.getRequestId(), response);
                }
                @Override public void fetchResponse(FetchRequestPB input, StreamObserver<GenerateOutputsPB> response) {
                    fetches.incrementAndGet(); stream(input.getRequestId(), response);
                }
            }).build().start();
            heartbeat = heartbeatListener();
            leader = "127.0.0.1:" + heartbeat.getAddress().getPort();
            heartbeat.createContext("/rtp_llm/master/info", exchange -> {
                byte[] body = ("{\"real_master_host\":\"" + leader + "\",\"queue_length\":0}").getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(heartbeatStatus, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            heartbeat.start();
            master = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", heartbeat.getAddress().getPort() + 2)).addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override public void schedule(FlexlbScheduleRequestPB input, StreamObserver<FlexlbScheduleResponsePB> response) {
                    schedules.incrementAndGet(); scheduleEntered.countDown();
                    if (scheduleFailure != null) { response.onError(scheduleFailure.asRuntimeException()); return; }
                    Context.current().addListener(c -> scheduleCancelled.countDown(), Runnable::run);
                    workers.submit(() -> {
                        try {
                            Thread.sleep(scheduleDelay);
                            var result = FlexlbScheduleResponsePB.newBuilder().setSuccess(true).setCode(200).setEnqueuedByMaster(batched)
                                .setLifecycle(RequestLifecyclePB.newBuilder().setRequestId(input.getRequestId()).setBatchId(9))
                                .addServerStatus(FlexlbServerStatusPB.newBuilder().setRole("PREFILL").setServerIp("127.0.0.1").setGrpcPort(engine.getPort())).build();
                            response.onNext(route.apply(result)); response.onCompleted();
                        } catch (InterruptedException cancelled) { Thread.currentThread().interrupt(); }
                    });
                }
                @Override public void cancel(FlexlbCancelRequestPB input, StreamObserver<FlexlbCancelResponsePB> response) {
                    cancellations.add(input); response.onNext(FlexlbCancelResponsePB.newBuilder().setFound(true).build()); response.onCompleted();
                }
            }).build().start();
            client = new NativeClient("127.0.0.1:" + master.getPort());
        }
        NativeClient.Result result(NativeClient.Call call) throws Exception { return call.completion().toCompletableFuture().get(4, TimeUnit.SECONDS); }
        @Override public void close() throws Exception {
            client.close(); heartbeat.stop(0); workers.shutdownNow(); engine.shutdownNow(); master.shutdownNow();
            engine.awaitTermination(3, TimeUnit.SECONDS); master.awaitTermination(3, TimeUnit.SECONDS);
        }
    }
    @Test void choosesGenerateAndMeasuresReceivedTokenRatherThanEngineReportedTime() throws Exception {
        try (var h = new Harness()) {
            h.scheduleDelay = 30; h.tokenDelay = 40;
            var call = h.client.send(request(1), Duration.ofSeconds(2));
            assertTrue(call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS) >= TimeUnit.MILLISECONDS.toNanos(60));
            var result = h.result(call); assertEquals(NativeClient.Outcome.success, result.outcome());
            assertEquals(3, result.outputTokens()); assertNotNull(result.ttftNanos()); assertTrue(result.elapsedNanos() >= result.ttftNanos());
            assertEquals(1, h.generates.get()); assertEquals(0, h.fetches.get());
            assertEquals(RoleAddrPB.RoleType.PREFILL, h.generated.get().getGenerateConfig().getRoleAddrs(0).getRole());
        }
    }

    @Test void finalLengthMustRespectIgnoreEosAndMinimumButAllowsAnActualEos() throws Exception {
        for (int mode = 0; mode < 3; mode++) try (var h = new Harness()) {
            h.output = (id, response) -> {
                response.onNext(token(id, 1, false)); response.onNext(token(id, 2, true)); response.onCompleted();
            };
            var request = request(1).toBuilder(); var input = GenerateInputPB.parseFrom(request.getGenerateInput()).toBuilder();
            input.getGenerateConfigBuilder().setIgnoreEos(mode == 0).setMinNewTokens(mode == 1 ? 3 : 0);
            request.setGenerateInput(input.build().toByteString());
            var result = h.result(h.client.send(request.build(), Duration.ofSeconds(2)));
            assertEquals(mode == 2 ? NativeClient.Outcome.success : NativeClient.Outcome.error, result.outcome());
        }
    }
    @Test void enqueuedRouteAlwaysFetchesAndConsumesToFinalCompletion() throws Exception {
        try (var h = new Harness()) {
            h.batched = true;
            var result = h.result(h.client.send(request(2), Duration.ofSeconds(2)));
            assertEquals(NativeClient.Outcome.success, result.outcome()); assertEquals(3, result.outputTokens());
            assertEquals(1, h.fetches.get()); assertEquals(0, h.generates.get());
        }
    }
    @Test void scheduleRejectionHasNoTokenSampleAndDoesNotOpenAnEngineStream() throws Exception {
        try (var h = new Harness()) {
            h.route = r -> r.toBuilder().setSuccess(false).setCode(429).clearLifecycle().build();
            var call = h.client.send(request(3), Duration.ofSeconds(2)); var result = h.result(call);
            assertEquals(NativeClient.Outcome.error, result.outcome()); assertEquals(429, result.errorCode()); assertNull(result.ttftNanos());
            assertThrows(ExecutionException.class, () -> call.firstToken().toCompletableFuture().get());
            assertEquals(0, h.generates.get()); assertEquals(0, h.fetches.get());
        }
    }
    @Test void successFlagWithFailureCodeCannotPass() throws Exception {
        try (var h = new Harness()) {
            h.route = r -> r.toBuilder().setCode(500).build();
            assertEquals(NativeClient.Outcome.error, h.result(h.client.send(request(4), Duration.ofSeconds(2))).outcome());
            assertEquals(0, h.generates.get());
        }
    }
    @Test void emptyOrTruncatedStreamsFailWithoutFabricatingAFirstToken() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> response.onCompleted();
            var result = h.result(h.client.send(request(5), Duration.ofSeconds(2)));
            assertNull(result.ttftNanos()); assertEquals("DATA_LOSS", result.grpcCode());
            h.output = (id, response) -> { response.onNext(token(id, 1, false)); response.onCompleted(); };
            result = h.result(h.client.send(request(6), Duration.ofSeconds(2)));
            assertNotNull(result.ttftNanos()); assertEquals(NativeClient.Outcome.error, result.outcome()); assertEquals(1, result.outputTokens());
        }
    }
    @Test void malformedFrameAndFramesAfterFinishedFail() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> { response.onNext(GenerateOutputsPB.newBuilder().setRequestId(id).build()); response.onCompleted(); };
            assertEquals("DATA_LOSS", h.result(h.client.send(request(7), Duration.ofSeconds(2))).grpcCode());
            h.output = (id, response) -> { response.onNext(token(id, 1, true)); response.onNext(token(id, 2, true)); response.onCompleted(); };
            assertEquals("DATA_LOSS", h.result(h.client.send(request(8), Duration.ofSeconds(2))).grpcCode());
        }
    }
    @Test void cancelDuringScheduleCancelsBothTheRpcAndMasterOwnershipWithoutRetry() throws Exception {
        try (var h = new Harness()) {
            h.scheduleDelay = 1000;
            var call = h.client.send(request(9), Duration.ofSeconds(2)); assertTrue(h.scheduleEntered.await(2, TimeUnit.SECONDS)); call.cancel();
            assertEquals(NativeClient.Outcome.cancelled, h.result(call).outcome()); assertTrue(h.scheduleCancelled.await(2, TimeUnit.SECONDS));
            var cancel = h.cancellations.poll(2, TimeUnit.SECONDS); assertNotNull(cancel); assertEquals(9, cancel.getRequestId());
            assertEquals(CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED, cancel.getReason()); assertEquals(1, h.schedules.get());
        }
    }
    @Test void timeoutBeforeFirstTokenNeverCreatesATtftSampleAndCancelsAdmission() throws Exception {
        try (var h = new Harness()) {
            h.scheduleDelay = 1000;
            var result = h.result(h.client.send(request(10), Duration.ofMillis(80)));
            assertEquals(NativeClient.Outcome.timeout, result.outcome()); assertNull(result.ttftNanos());
            var cancel = h.cancellations.poll(2, TimeUnit.SECONDS); assertNotNull(cancel);
            assertEquals(CancelReasonPB.CANCEL_REASON_DEADLINE_EXCEEDED, cancel.getReason());
        }
    }
    @Test void cancellingAfterFirstTokenKeepsTheObservedTtftAndCancelsEngineAndMaster() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> response.onNext(token(id, 1, false));
            var call = h.client.send(request(11), Duration.ofSeconds(2));
            long ttft = call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS); call.cancel();
            var result = h.result(call); assertEquals(ttft, result.ttftNanos()); assertEquals(NativeClient.Outcome.cancelled, result.outcome());
            assertTrue(h.streamCancelled.await(2, TimeUnit.SECONDS));
            assertEquals(9, Objects.requireNonNull(h.cancellations.poll(2, TimeUnit.SECONDS)).getBatchId());
        }
    }
    @Test void nativeTrailerBusinessErrorIsPreservedWithoutATokenSample() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> {
                var details = new Metadata(); details.put(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER),
                    ErrorDetailsPB.newBuilder().setErrorCode(8429).build().toByteArray());
                response.onError(Status.RESOURCE_EXHAUSTED.asRuntimeException(details));
            };
            var result = h.result(h.client.send(request(12), Duration.ofSeconds(2)));
            assertEquals(8429, result.errorCode()); assertEquals("RESOURCE_EXHAUSTED", result.grpcCode()); assertNull(result.ttftNanos());
        }
    }
    @Test void independentConcurrentCallsShareChannelsAndCancelOnlyTheirOwnRequest() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> { response.onNext(token(id, 1, id != 13)); if (id != 13) response.onCompleted(); };
            var held = h.client.send(request(13), Duration.ofSeconds(2)); held.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertThrows(IllegalArgumentException.class, () -> h.client.send(request(13), Duration.ofSeconds(2)));
            var others = new ArrayList<NativeClient.Call>();
            for (long id = 14; id < 34; id++) others.add(h.client.send(request(id), Duration.ofSeconds(2)));
            held.cancel(); for (var call : others) assertEquals(NativeClient.Outcome.success, h.result(call).outcome());
            assertEquals(NativeClient.Outcome.cancelled, h.result(held).outcome());
        }
    }
    @Test void closeCancelsActiveCallsAndRejectsFurtherSends() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> response.onNext(token(id, 1, false));
            var call = h.client.send(request(34), Duration.ofSeconds(2)); call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
            h.client.close(); assertEquals(NativeClient.Outcome.cancelled, h.result(call).outcome());
            assertNotNull(h.cancellations.poll(2, TimeUnit.SECONDS));
            assertThrows(IllegalStateException.class, () -> h.client.send(request(35), Duration.ofSeconds(2)));
        }
    }
    @Test void deadlineCoversSchedulingAndStreamingAndPropagatesToEngine() throws Exception {
        try (var h = new Harness()) {
            h.scheduleDelay = 40; h.output = (id, response) -> response.onNext(token(id, 1, false));
            var call = h.client.send(request(37), Duration.ofMillis(180));
            call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
            var result = h.result(call); assertEquals(NativeClient.Outcome.timeout, result.outcome());
            assertTrue(h.engineSawDeadline.get()); assertNotNull(result.ttftNanos());
            assertTrue(result.elapsedNanos() < TimeUnit.SECONDS.toNanos(1));
            assertTrue(h.streamCancelled.await(2, TimeUnit.SECONDS));
        }
    }
    @Test void cumulativeAndMultiTokenFramesCountActualReceivedPayload() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> {
                for (int count = 1; count <= 3; count++) {
                    var frame = token(id, count, count == 3).toBuilder();
                    frame.getFlattenOutputBuilder().getOutputIdsBuilder().setShape(2, count).setInt32Data(ByteString.copyFrom(new byte[count * 4]));
                    response.onNext(frame.build());
                }
                response.onCompleted();
            };
            var nativeInput = input(38).toBuilder(); nativeInput.getGenerateConfigBuilder().setReturnIncremental(false);
            var cumulative = h.result(h.client.send(request(38).toBuilder().setGenerateInput(nativeInput.build().toByteString()).build(), Duration.ofSeconds(2)));
            assertEquals(NativeClient.Outcome.success, cumulative.outcome()); assertEquals(3, cumulative.outputTokens());
            h.output = (id, response) -> {
                var frame = token(id, 3, true).toBuilder();
                frame.getFlattenOutputBuilder().getOutputIdsBuilder().setShape(2, 3).setInt32Data(ByteString.copyFrom(new byte[12]));
                response.onNext(frame.build()); response.onCompleted();
            };
            assertEquals(NativeClient.Outcome.success, h.result(h.client.send(request(39), Duration.ofSeconds(2))).outcome());
        }
    }
    @Test void firstTokenCallbackMayCancelImmediatelyWithoutLosingTheObservedToken() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> response.onNext(token(id, 1, false));
            var call = h.client.send(request(40), Duration.ofSeconds(2)); call.firstToken().thenRun(call::cancel);
            var result = h.result(call); assertEquals(NativeClient.Outcome.cancelled, result.outcome());
            assertEquals(1, result.outputTokens()); assertNotNull(result.ttftNanos());
        }
    }
    @Test void inconsistentTokenShapeAndWrongLifecycleCannotBecomeSuccessfulSamples() throws Exception {
        try (var h = new Harness()) {
            h.output = (id, response) -> {
                var frame = token(id, 1, true).toBuilder(); frame.getFlattenOutputBuilder().getOutputIdsBuilder().setShape(2, 2);
                response.onNext(frame.build()); response.onCompleted();
            };
            var result = h.result(h.client.send(request(41), Duration.ofSeconds(2)));
            assertEquals("DATA_LOSS", result.grpcCode()); assertNull(result.ttftNanos());
            h.route = r -> r.toBuilder().setLifecycle(r.getLifecycle().toBuilder().setRequestId(9999)).build();
            result = h.result(h.client.send(request(42), Duration.ofSeconds(2)));
            assertEquals("DATA_LOSS", result.grpcCode()); assertEquals(1, h.generates.get());
        }
    }
    @Test void rejectsInvalidInputAndUnsupportedRoutesBeforeClaimingSuccess() throws Exception {
        try (var h = new Harness()) {
            assertThrows(IllegalArgumentException.class, () -> h.client.send(request(36), Duration.ZERO));
            assertThrows(IllegalArgumentException.class, () -> h.client.send(request(36).toBuilder().setSeqLen(99).build(), Duration.ofSeconds(1)));
            h.route = r -> r.toBuilder().clearServerStatus().build();
            assertEquals("DATA_LOSS", h.result(h.client.send(request(36), Duration.ofSeconds(1))).grpcCode());
        }
    }
    @Test void transportFailureRetriesOnlyBackupAndCancelKeepsTheAttemptedMaster() throws Exception {
        try (var first = new Harness(); var backup = new Harness()) {
            first.scheduleFailure = Status.UNAVAILABLE;
            backup.output = (id, response) -> response.onNext(token(id, 1, false));
            var routes = new AtomicReference<>(List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")));
            try (var client = new NativeClient(routes::get)) {
                var call = client.send(request(100), Duration.ofSeconds(3));
                call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
                routes.set(List.of(first.endpoint().get("rpc")));
                call.cancel();
                assertEquals(outcome(call), "cancelled");
                assertEquals(1, first.schedules.get()); assertEquals(1, backup.schedules.get());
                assertEquals(0, first.generates.get()); assertEquals(1, backup.generates.get());
                assertEquals(100, Objects.requireNonNull(backup.cancellations.poll(2, TimeUnit.SECONDS)).getRequestId());
                assertNull(first.cancellations.poll(100, TimeUnit.MILLISECONDS));
            }
        }
    }
    private static String outcome(NativeClient.Call call) throws Exception {
        return call.completion().toCompletableFuture().get(3, TimeUnit.SECONDS).outcome().name();
    }
    @Test void businessRejectionDeadlineAndStreamErrorNeverRetryBackup() throws Exception {
        try (var first = new Harness(); var backup = new Harness();
             var client = new NativeClient(() -> List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")))) {
            first.route = r -> r.toBuilder().setCode(429).setSuccess(false).clearLifecycle().build();
            assertEquals("error", outcome(client.send(request(101), Duration.ofSeconds(2))));
            first.route = Function.identity(); first.scheduleDelay = 2000;
            assertEquals("timeout", outcome(client.send(request(102), Duration.ofMillis(100))));
            first.scheduleDelay = 0;
            first.output = (id, response) -> response.onError(Status.UNAVAILABLE.asRuntimeException());
            assertEquals("error", outcome(client.send(request(103), Duration.ofSeconds(2))));
            assertEquals(0, backup.schedules.get()); assertEquals(0, backup.generates.get());
        }
    }
    @Test void noHealthyMasterAndBothTransportFailuresNeverBypassScheduling() throws Exception {
        try (var first = new Harness(); var backup = new Harness()) {
            var routes = new AtomicReference<List<String>>(List.of());
            try (var client = new NativeClient(routes::get)) {
                assertEquals("error", outcome(client.send(request(104), Duration.ofSeconds(2))));
                first.scheduleFailure = Status.UNAVAILABLE; backup.scheduleFailure = Status.UNAVAILABLE;
                routes.set(List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")));
                assertEquals("error", outcome(client.send(request(105), Duration.ofSeconds(2))));
                assertEquals(1, first.schedules.get()); assertEquals(1, backup.schedules.get());
                assertEquals(0, first.generates.get()); assertEquals(0, backup.generates.get());
            }
        }
    }

    @Test void remoteCancelledTransportCanUseBackupWhileLocalCancellationCannot() throws Exception {
        try (var first = new Harness(); var backup = new Harness();
             var client = new NativeClient(() -> List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")))) {
            first.scheduleFailure = Status.CANCELLED;
            assertEquals("success", outcome(client.send(request(106), Duration.ofSeconds(2))));
            assertEquals(1, first.schedules.get()); assertEquals(1, backup.schedules.get());
            first.scheduleFailure = null; first.scheduleDelay = 2000;
            var call = client.send(request(107), Duration.ofSeconds(3));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (first.schedules.get() < 2 && System.nanoTime() < deadline) Thread.sleep(5);
            assertEquals(2, first.schedules.get()); call.cancel();
            assertEquals("cancelled", outcome(call)); assertEquals(1, backup.schedules.get());
        }
    }

}
