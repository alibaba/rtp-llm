package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Multi-shard routing tests for the Java mock engine.
 *
 * <p>Verifies that multi-shard configurations (multiple prefill engine groups
 * serving different shards) route requests correctly, complete all requests
 * without leaks, and distribute load evenly across engines within each shard.
 *
 * <p>Since the mock engine cluster has no native shard concept, shards are
 * simulated by grouping engines on different port ranges and routing requests
 * to decode engines within the same shard via {@code RoleAddrPB} port hints.
 */
class MultiShardRoutingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 63000;

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<List<JavaMockEngineCluster.FastRpcService>> prefillByShard;
    private List<List<JavaMockEngineCluster.FastRpcService>> decodeByShard;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: Balanced 2x2 shard configuration ────────────

    @Test
    void balancedTwoShardRoutingCompletesAllRequestsWithNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        // 2 shards, each with 2 prefill + 2 decode engines (4P + 4D total)
        startMultiShardCluster(model, List.of(
                new ShardConfig(2, 2),
                new ShardConfig(2, 2)), BASE_PORT);

        int requestsPerShard = 20;
        int totalRequests = requestsPerShard * prefillByShard.size(); // 40

        // Enqueue requests round-robin across prefill engines within each shard
        long batchId = 10_000;
        int requestId = 1;
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            for (int i = 0; i < requestsPerShard; i++) {
                JavaMockEngineCluster.FastRpcService prefill = prefillEngines.get(i % prefillEngines.size());
                int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
                EngineRpcService.GenerateInputPB input = inputWithDecode(requestId++, 10, decodePort);
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(batchId++, slot(0, input)));
                assertEquals(0, response.getErrorsCount(),
                        "no errors expected for request " + (requestId - 1));
                assertEquals(1, response.getSuccessesCount(),
                        "request " + (requestId - 1) + " should be accepted");
            }
        }

        // Wait for all requests to complete
        awaitTotalCompleted(totalRequests, 10_000);

        // Assert no inflight leak on any engine
        assertAllInflightZero();

        // Assert no leak_detected flag on any engine
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on port " + service.getGrpcPort());
        }

        // Verify prefill load distribution: no single engine handles >50% (tolerance 55%)
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            long shardPrefillTotal = prefillEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getAcceptedCount)
                    .sum();
            assertEquals(requestsPerShard, shardPrefillTotal,
                    "shard " + shard + " prefill engines should accept " + requestsPerShard + " total");
            for (JavaMockEngineCluster.FastRpcService engine : prefillEngines) {
                double ratio = (double) engine.getAcceptedCount() / shardPrefillTotal;
                assertTrue(ratio <= 0.55,
                        "prefill engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " load (>55%)");
            }
        }

        // Verify decode engines received all requests and distributed load
        for (int shard = 0; shard < decodeByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            long shardDecodeTotal = decodeEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(requestsPerShard, shardDecodeTotal,
                    "shard " + shard + " decode engines should complete " + requestsPerShard + " total");
            for (JavaMockEngineCluster.FastRpcService engine : decodeEngines) {
                double ratio = (double) engine.getCompletedCount() / shardDecodeTotal;
                assertTrue(ratio <= 0.55,
                        "decode engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " decode load (>55%)");
            }
        }

        // Verify HTTP snapshot shows no leaks
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port " + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port " + engine.get("port").asInt());
        }
    }

    // ──────────── Test 2: Asymmetric P/D ratios per shard ────────────

    @Test
    void asymmetricPdRatioPerShardCompletesAllRequestsWithNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        // Shard 0: 2P + 4D, Shard 1: 2P + 2D
        startMultiShardCluster(model, List.of(
                new ShardConfig(2, 4),
                new ShardConfig(2, 2)), BASE_PORT);

        int requestsPerShard = 20;
        int totalRequests = requestsPerShard * prefillByShard.size(); // 40

        long batchId = 20_000;
        int requestId = 1;
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            for (int i = 0; i < requestsPerShard; i++) {
                JavaMockEngineCluster.FastRpcService prefill = prefillEngines.get(i % prefillEngines.size());
                int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
                EngineRpcService.GenerateInputPB input = inputWithDecode(requestId++, 10, decodePort);
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(batchId++, slot(0, input)));
                assertEquals(0, response.getErrorsCount(),
                        "no errors expected for shard " + shard + " request " + i);
                assertEquals(1, response.getSuccessesCount(),
                        "request " + requestId + " should be accepted");
            }
        }

        awaitTotalCompleted(totalRequests, 10_000);
        assertAllInflightZero();

        // Verify prefill load distribution within each shard
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            long shardPrefillTotal = prefillEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getAcceptedCount)
                    .sum();
            assertEquals(requestsPerShard, shardPrefillTotal,
                    "shard " + shard + " prefill should accept " + requestsPerShard);
            for (JavaMockEngineCluster.FastRpcService engine : prefillEngines) {
                double ratio = (double) engine.getAcceptedCount() / shardPrefillTotal;
                assertTrue(ratio <= 0.6,
                        "prefill engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " load (>60%)");
            }
        }

        // Verify decode load distribution within each shard
        for (int shard = 0; shard < decodeByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            long shardDecodeTotal = decodeEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(requestsPerShard, shardDecodeTotal,
                    "shard " + shard + " decode should complete " + requestsPerShard);
            for (JavaMockEngineCluster.FastRpcService engine : decodeEngines) {
                double ratio = (double) engine.getCompletedCount() / shardDecodeTotal;
                assertTrue(ratio <= 0.55,
                        "decode engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " decode load (>55%)");
            }
        }

        // Verify HTTP snapshot
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port " + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port " + engine.get("port").asInt());
        }
    }

    // ──────────── Cluster setup ────────────

    private record ShardConfig(int nPrefill, int nDecode) {}

    /**
     * Start a multi-shard cluster. All prefill engines are created first
     * (contiguous ports), then all decode engines, so that shard prefill
     * ports don't collide with another shard's decode ports.
     */
    private void startMultiShardCluster(MockPerformanceModel model,
                                         List<ShardConfig> shards, int basePort) throws IOException {
        services = new ConcurrentHashMap<>();
        prefillByShard = new ArrayList<>();
        decodeByShard = new ArrayList<>();

        // First pass: create all prefill engines
        int portOffset = 0;
        for (ShardConfig shard : shards) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = new ArrayList<>();
            for (int i = 0; i < shard.nPrefill; i++) {
                int port = basePort + portOffset++;
                JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                        "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        port, services, scheduler, model, 100,
                        new JavaMockEngineCluster.ClusterStats());
                services.put(port, service);
                prefillEngines.add(service);
            }
            prefillByShard.add(prefillEngines);
        }

        // Second pass: create all decode engines
        for (ShardConfig shard : shards) {
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = new ArrayList<>();
            for (int i = 0; i < shard.nDecode; i++) {
                int port = basePort + portOffset++;
                JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                        "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port, services, scheduler, model, 100,
                        new JavaMockEngineCluster.ClusterStats());
                services.put(port, service);
                decodeEngines.add(service);
            }
            decodeByShard.add(decodeEngines);
        }

        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
        controlServer.start();
    }

    // ──────────── Polling helpers ────────────

    private long totalCompleted() {
        return services.values().stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                .sum();
    }

    private void awaitTotalCompleted(int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (totalCompleted() >= expected) {
                return;
            }
            Thread.sleep(10);
        }
        fail("expected " + expected + " completions, got " + totalCompleted());
    }

    private void assertAllInflightZero() {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getInflightCount(),
                    "inflight should be 0 for engine on port " + service.getGrpcPort());
        }
    }

    // ──────────── HTTP helpers ────────────

    private JsonNode snapshot() throws Exception {
        String body = httpGet(controlServer.getPort(), "/snapshot");
        return MAPPER.readTree(body).path("engines");
    }

    private static String httpGet(int port, String path) throws Exception {
        HttpResponse<String> response = HTTP_CLIENT.send(
                HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + port + path))
                        .GET()
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), "GET " + path + " failed");
        return response.body();
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                // Routing tests enqueue more batches per engine than a bounded
                // waiting queue allows; use -1 (unbounded) to keep the original
                // unbounded-queue routing semantics under test.
                "prefill", Map.of("scale", 1.0, "max_waiting_batches", -1),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", formula))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    // ──────────── Protobuf builders ────────────

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                                .setGrpcPort(decodePort)
                                .build())
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.EnqueueBatchDpSlotPB slot(
            int dpRank, EngineRpcService.GenerateInputPB... inputs) {
        EngineRpcService.EnqueueBatchDpSlotPB.Builder slot =
                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(dpRank);
        for (EngineRpcService.GenerateInputPB input : inputs) {
            slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                    .setInput(input)
                    .build());
        }
        return slot.build();
    }

    private static EngineRpcService.EnqueueBatchRequestPB batch(
            long batchId, EngineRpcService.EnqueueBatchDpSlotPB... slots) {
        return EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId)
                .addAllDpSlots(List.of(slots))
                .build();
    }

    // ──────────── RPC helpers ────────────

    private static EngineRpcService.EnqueueBatchResponsePB enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        return unary(observer -> service.enqueueBatch(request, observer));
    }

    private static <T> T unary(Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                fail("unary response timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for unary response");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "unary response");
        return response.get();
    }
}
