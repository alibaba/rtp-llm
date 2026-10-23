package org.flexlb.mockengine;

import io.grpc.Status;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Semaphore;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Fixed-target fallback and request ownership through the shared Schedule transport. */
class JavaLoadClientLegacyFallbackTest {

    @TempDir
    Path tempDir;

    private static final String UNREACHABLE_ENGINE = "127.0.0.1:2";

    private JavaLoadClient client(Server master, boolean enableFallback, boolean fetch) {
        return client(master, enableFallback, fetch, 1);
    }

    private JavaLoadClient client(Server master, boolean enableFallback, boolean fetch, int channels) {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:" + master.getPort(),
                0, 4, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                2_000L, 500.0, fetch, false, channels, 1, 0L, 120, true,
                "engine_service", "", 0, 0, "",
                enableFallback, "", false);
        return new JavaLoadClient(config);
    }

    private static Server master(Status failure, FlexlbScheduleProtocol.FlexlbScheduleResponsePB response)
            throws Exception {
        return NettyServerBuilder.forPort(0).addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
            @Override public void schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                    StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
                if (failure != null) observer.onError(failure.asRuntimeException());
                else { observer.onNext(response); observer.onCompleted(); }
            }
        }).build().start();
    }

    private static JavaLoadClient.RequestResult handle(
            JavaLoadClient client, JavaLoadClient.TraceRecord record) throws Exception {
        Method handle = JavaLoadClient.class.getDeclaredMethod(
                "handleRequest", JavaLoadClient.TraceRecord.class, Semaphore.class, double.class);
        handle.setAccessible(true);
        var result = (JavaLoadClient.RequestResult)
                handle.invoke(client, record, new Semaphore(4), 0.0);
        assertSettled(client, result);
        return result;
    }

    private static int count(JavaLoadClient client, String name) throws Exception {
        var field = JavaLoadClient.class.getDeclaredField(name);
        field.setAccessible(true);
        return ((java.util.concurrent.atomic.AtomicInteger) field.get(client)).get();
    }

    private static void assertSettled(JavaLoadClient client, JavaLoadClient.RequestResult result) throws Exception {
        assertEquals(1, count(client, "responseCount"));
        assertEquals(1, count(client, "monitoredCompleted"));
        assertEquals(List.of(result), client.completedResults);
        assertEquals(1, client.successCount.get() + client.errorCount.get());
        assertEquals(0, client.inflightCount.get());
        assertEquals(0, client.scheduleInflightCount.get());
    }

    private static void closeClient(JavaLoadClient client) throws Exception {
        Method close = JavaLoadClient.class.getDeclaredMethod("close");
        close.setAccessible(true);
        close.invoke(client);
    }

    private static JavaLoadClient.TraceRecord rec(int idx) {
        List<Integer> tokens = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            tokens.add(i);
        }
        return new JavaLoadClient.TraceRecord(idx, "rid-" + idx, "trace-" + idx, 0L,
                64, 8, List.of(), tokens);
    }

    @ParameterizedTest
    @ValueSource(strings = {"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "REJECTED"})
    @Timeout(30)
    void singleTargetRetainsAnyScheduleFailureFallback(String failure) throws Exception {
        verifyScheduleFailure(failure, true);
    }

    @ParameterizedTest
    @ValueSource(strings = {"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "REJECTED"})
    @Timeout(30)
    void singleTargetKeepsErrorOnMasterPathWithoutOptIn(String failure) throws Exception {
        verifyScheduleFailure(failure, false);
    }

    private void verifyScheduleFailure(String failure, boolean fallback) throws Exception {
        boolean rejected = failure.equals("REJECTED");
        var response = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(8431).setSuccess(false).setErrorMessage("admission rejected").build();
        Server master = master(rejected ? null : Status.fromCode(Status.Code.valueOf(failure)), response);
        try (JavaLoadClient client = client(master, fallback, false)) {
            client.fallbackPrefillAddrs.add(UNREACHABLE_ENGINE);
            JavaLoadClient.RequestResult result = handle(client, rec(0));
            assertEquals(fallback ? "fallback" : "master", result.routePath);
            assertEquals(!fallback && rejected ? "schedule_error" : "exception", result.status);
            assertEquals(fallback, result.error.contains("; fallback="));
            assertEquals(failure.equals("UNAVAILABLE") ? "transport"
                    : failure.equals("DEADLINE_EXCEEDED") ? "deadline" : "business", result.errorKind);
            assertFalse(result.failover);
            assertEquals(1, result.scheduleAttempts.size());
            var attempt = result.scheduleAttempts.get(0);
            assertEquals("127.0.0.1:" + master.getPort(), attempt.target);
            assertEquals(rejected ? "OK" : failure, attempt.status);
            assertEquals(rejected ? Integer.valueOf(8431) : null, attempt.responseCode);
            assertTrue(attempt.endedEpochMs >= attempt.startedEpochMs);
            if (!rejected) {
                assertEquals(0.0, result.scheduleMs);
                assertEquals(0.0, result.schedDoneEpochMs);
            }
        } finally {
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    @Timeout(30)
    void singleTargetFetchFailureRetainsFallbackPolicy(boolean fallback) throws Exception {
        var response = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(200).setSuccess(true).setEnqueuedByMaster(true)
                .addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole("PREFILL").setServerIp("127.0.0.1").setGrpcPort(2)).build();
        Server master = master(null, response);
        try (JavaLoadClient client = client(master, fallback, true)) {
            client.fallbackPrefillAddrs.add(UNREACHABLE_ENGINE);
            var result = handle(client, rec(0));
            assertEquals(fallback ? "fallback" : "master", result.routePath);
            assertEquals("exception", result.status);
            assertEquals(fallback, result.error.contains("; fallback="));
            assertEquals("none", result.errorKind);
            assertEquals(1, result.scheduleAttempts.size());
        } finally {
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "REJECTED", "FETCH"})
    @Timeout(30)
    void discoveryRetainsTransportOnlyFallbackEvenWithOneCandidate(String failure) throws Exception {
        boolean fetch = failure.equals("FETCH");
        boolean rejected = failure.equals("REJECTED");
        var response = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(rejected ? 8431 : 200).setSuccess(!rejected).setEnqueuedByMaster(true)
                .addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole("PREFILL").setServerIp("127.0.0.1").setGrpcPort(2)).build();
        Server master = master(fetch || rejected ? null : Status.fromCode(Status.Code.valueOf(failure)), response);
        var http = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        String httpTarget = "127.0.0.1:" + http.getAddress().getPort();
        http.createContext("/rtp_llm/master/info", exchange -> {
            byte[] body = ("{\"consistency_enabled\":true,\"real_master_host\":\"" + httpTarget + "\"}")
                    .getBytes(java.nio.charset.StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            try (var output = exchange.getResponseBody()) { output.write(body); }
        });
        http.start();
        try {
            Path discovery = tempDir.resolve("masters.json");
            java.nio.file.Files.writeString(discovery, "{\"hosts\":[{\"http\":\"" + httpTarget
                    + "\",\"grpc\":\"127.0.0.1:" + master.getPort() + "\"}]}");
            var config = JavaLoadClient.Config.fromEnvironment(java.util.Map.of(
                    "MASTER_DISCOVERY_FILE", discovery.toString(), "TIMEOUT_MS", "2000",
                    "FETCH_OUTPUT_STREAM", Boolean.toString(fetch), "ENABLE_FALLBACK", "true",
                    "N_CHANNELS", "1", "EVENT_LOOP_THREADS", "1"));
            try (JavaLoadClient client = new JavaLoadClient(config, false)) {
                client.fallbackPrefillAddrs.add(UNREACHABLE_ENGINE);
                var result = handle(client, rec(0));
                assertEquals(failure.equals("UNAVAILABLE") ? "fallback"
                        : rejected || fetch ? "master" : "failed", result.routePath);
                assertEquals(rejected ? "schedule_error" : "exception", result.status);
                assertEquals(failure.equals("UNAVAILABLE"), result.error.contains("; fallback="));
                assertEquals(failure.equals("UNAVAILABLE") ? "transport"
                        : failure.equals("DEADLINE_EXCEEDED") ? "deadline"
                        : fetch ? "none" : "business", result.errorKind);
                assertEquals(1, result.scheduleAttempts.size());
                assertFalse(result.failover);
                if (failure.equals("UNAVAILABLE")) assertEquals(0.0, result.scheduleMs);
                else assertTrue(result.scheduleMs > 0);
            }
        } finally {
            http.stop(0);
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @Timeout(30)
    void permitAndInflightRemainHeldUntilFetchCompletes() throws Exception {
        var opened = new java.util.concurrent.CountDownLatch(1);
        var observer = new java.util.concurrent.atomic.AtomicReference<
                io.grpc.stub.StreamObserver<org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB>>();
        io.grpc.Server engine = io.grpc.netty.NettyServerBuilder.forPort(0).addService(
                new org.flexlb.engine.grpc.RpcServiceGrpc.RpcServiceImplBase() {
                    @Override public void fetchResponse(
                            org.flexlb.engine.grpc.EngineRpcService.FetchRequestPB request,
                            io.grpc.stub.StreamObserver<org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB> output) {
                        observer.set(output);
                        opened.countDown();
                    }
                }).build().start();
        Server master = master(null, FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(200).setSuccess(true).setEnqueuedByMaster(true)
                .addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole("PREFILL").setServerIp("127.0.0.1").setGrpcPort(engine.getPort()))
                .build());
        JavaLoadClient client = client(master, false, true);
        var executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        Semaphore permit = new Semaphore(1);
        try {
            Method method = JavaLoadClient.class.getDeclaredMethod("handleRequest",
                    JavaLoadClient.TraceRecord.class, Semaphore.class, double.class);
            method.setAccessible(true);
            var future = executor.submit(() -> (JavaLoadClient.RequestResult)
                    method.invoke(client, rec(1), permit, 0.0));
            assertTrue(opened.await(5, java.util.concurrent.TimeUnit.SECONDS));
            assertEquals(1, client.inflightCount.get());
            assertEquals(0, client.scheduleInflightCount.get());
            assertFalse(permit.tryAcquire(), "a held Fetch must retain the request permit");
            observer.get().onNext(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB.newBuilder()
                    .setFlattenOutput(org.flexlb.engine.grpc.EngineRpcService.FlattenOutputPB.newBuilder()
                            .addFinished(true)).build());
            observer.get().onCompleted();
            var result = future.get(5, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals("ok", result.status);
            assertSettled(client, result);
            assertEquals(0, client.inflightCount.get());
            assertEquals(1, permit.availablePermits());
        } finally {
            executor.shutdownNow();
            closeClient(client);
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            engine.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void scheduleOnlySettlesOnceWithoutOpeningEngineStream() throws Exception {
        Server master = master(null, FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(200).setSuccess(true).setEnqueuedByMaster(true).build());
        try (JavaLoadClient client = client(master, false, false)) {
            assertEquals("scheduled", handle(client, rec(1)).status);
            assertEquals(1, client.successCount.get());
        } finally {
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void successfulFallbackPreservesInputAndSettlesOnce(boolean fetchFailure) throws Exception {
        var received = new java.util.concurrent.atomic.AtomicReference<org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB>();
        Server engine = NettyServerBuilder.forPort(0).addService(
                new org.flexlb.engine.grpc.RpcServiceGrpc.RpcServiceImplBase() {
                    @Override public void generateStreamCall(org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB request,
                            StreamObserver<org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB> observer) {
                        received.set(request);
                        observer.onNext(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB.newBuilder()
                                .setFlattenOutput(org.flexlb.engine.grpc.EngineRpcService.FlattenOutputPB.newBuilder()
                                        .addFinished(true)).build());
                        observer.onCompleted();
                    }
                }).build().start();
        Server master = master(fetchFailure ? null : Status.UNAVAILABLE,
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setCode(200).setSuccess(true).setEnqueuedByMaster(true)
                        .addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                                .setRole("PREFILL").setServerIp("127.0.0.1").setGrpcPort(2)).build());
        try (JavaLoadClient client = client(master, true, true)) {
            String target = "127.0.0.1:" + engine.getPort();
            client.fallbackPrefillAddrs.add(target);
            var record = rec(17);
            var result = handle(client, record);
            assertEquals("ok", result.status);
            assertEquals("fallback", result.routePath);
            assertEquals(target, result.prefill);
            assertEquals(0.0, result.scheduleMs);
            var input = received.get();
            assertEquals(record.requestId, input.getRequestId());
            assertEquals(record.tokenIds, input.getTokenIdsList());
            assertEquals(record.sourceRid, input.getRequestInfo().getRequestId());
            assertEquals(record.traceId, input.getRequestInfo().getTraceId());
            assertEquals(record.outputLen, input.getGenerateConfig().getMaxNewTokens());
            assertEquals(1, input.getGenerateConfig().getRoleAddrsCount());
            assertEquals(engine.getPort(), input.getGenerateConfig().getRoleAddrs(0).getGrpcPort());
            assertEquals(2_000, input.getGenerateConfig().getTimeoutMs());
            assertTrue(input.getStartTime() > 0);
        } finally {
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            engine.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void interruptionBeforeAdmissionDoesNotPublishOrReleaseAnUnownedPermit() throws Exception {
        Server master = master(null, FlexlbScheduleProtocol.FlexlbScheduleResponsePB.getDefaultInstance());
        try (JavaLoadClient client = client(master, false, true)) {
            var method = JavaLoadClient.class.getDeclaredMethod("handleRequest",
                    JavaLoadClient.TraceRecord.class, Semaphore.class, double.class);
            method.setAccessible(true);
            var permit = new Semaphore(0);
            Thread.currentThread().interrupt();
            var result = (JavaLoadClient.RequestResult) method.invoke(client, rec(1), permit, 0.0);
            assertTrue(Thread.interrupted(), "retain caller interruption");
            assertEquals("interrupted", result.error);
            assertEquals(0, permit.availablePermits());
            assertEquals(1, count(client, "responseCount"));
            assertEquals(1, client.errorCount.get());
            assertEquals(0, count(client, "monitoredCompleted"));
            assertEquals(0, client.actualSentCount.get());
            assertTrue(client.completedResults.isEmpty());
            assertEquals(0, client.inflightCount.get());
        } finally {
            Thread.interrupted();
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void enginePoolsKeepPerTargetRoundRobinAndCloseEveryChannel() throws Exception {
        Server master = master(null, FlexlbScheduleProtocol.FlexlbScheduleResponsePB.getDefaultInstance());
        var channels = new java.util.HashSet<io.grpc.ManagedChannel>();
        var executor = java.util.concurrent.Executors.newFixedThreadPool(4);
        try {
            try (JavaLoadClient client = client(master, false, true, 2)) {
                var method = JavaLoadClient.class.getDeclaredMethod("getEngineChannel", String.class);
                method.setAccessible(true);
                var first = (io.grpc.ManagedChannel) method.invoke(client, "127.0.0.1:2");
                var second = (io.grpc.ManagedChannel) method.invoke(client, "127.0.0.1:2");
                org.junit.jupiter.api.Assertions.assertNotSame(first, second);
                org.junit.jupiter.api.Assertions.assertSame(first, method.invoke(client, "127.0.0.1:2"));
                var calls = new ArrayList<java.util.concurrent.Future<io.grpc.ManagedChannel>>();
                for (int i = 0; i < 64; i++) {
                    calls.add(executor.submit(() -> (io.grpc.ManagedChannel) method.invoke(client, "127.0.0.1:2")));
                }
                for (var call : calls) channels.add(call.get(3, TimeUnit.SECONDS));
                assertEquals(java.util.Set.of(first, second), channels);
                var other = (io.grpc.ManagedChannel) method.invoke(client, "127.0.0.1:3");
                assertFalse(channels.contains(other));
                channels.add(other);
                channels.add((io.grpc.ManagedChannel) method.invoke(client, "127.0.0.1:3"));
                assertEquals(4, channels.size());
            }
            assertTrue(channels.stream().allMatch(io.grpc.ManagedChannel::isShutdown));
        } finally {
            executor.shutdownNow();
            master.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
