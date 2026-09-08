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
 * Inflight TTL expiry and recovery test for the Java mock engine.
 *
 * <p>Starts a cluster with 1 prefill + 2 decode engines, enqueues 10 requests
 * with a 5000ms prefill formula so they stay inflight. Verifies the TTL/leak
 * detection mechanism: within the 60s grace window no leak is detected, but
 * when grace is simulated to expire (0-window) the leak is flagged. Then
 * uses the HTTP /inject endpoint to inject noRespond on a decode engine,
 * verifies that scheduled completions still drain inflight to zero (TTL
 * cleanup is not a leak), clears the injection via /clear_inject, and
 * confirms the engine recovers by processing new requests normally.
 */
class InflightTtlExpiryTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 62500;

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    @Test
    void inflightTtlExpiryAndRecovery() throws Exception {
        MockPerformanceModel model = model("1000"); // 1s prefill to keep requests inflight briefly
        startCluster(model, 1, 2);

        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode0 = decodeServices.get(0);
        JavaMockEngineCluster.FastRpcService decode1 = decodeServices.get(1);

        int n = 10;

        // ── Phase 1: Enqueue 10 requests with decode routing → stay inflight ──
        enqueueBatch(prefill, 10000, 1, n, decodeServices);
        awaitInflight(prefill, 1, 1_000);
        assertTrue(prefill.getInflightCount() > 0,
                "prefill should have inflight > 0 after enqueue");

        // ── Phase 2: TTL grace window (60s) — should NOT detect leak ──
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.checkLeakDrain(60_000_000_000L);
            assertFalse(service.isLeakDetected(),
                    "leak should NOT be detected during grace window on port "
                            + service.getGrpcPort());
        }

        // ── Phase 3: Simulate TTL expiry (0 grace) — leak detected on prefill ──
        prefill.checkLeakDrain(0);
        assertTrue(prefill.isLeakDetected(),
                "leak should be detected when TTL expires and inflight > 0 on prefill");

        // Decode engines should NOT have leak (they may not have started decode yet)
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak detected");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak detected");

        // ── Phase 4: Wait for TTL cleanup (scheduled completions fire, inflight → 0) ──
        awaitAllInflightZero(5_000);
        assertAllInflightZero();

        // ── Phase 5: Decode engines remain clean — TTL cleanup is not a leak ──
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak after TTL cleanup");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak after TTL cleanup");

        // ── Phase 6: Inject noRespond on decode engine 0 via HTTP /inject ──
        httpPost(controlServer.getPort(), "/inject",
                "{\"port\":" + decode0.getGrpcPort()
                        + ",\"type\":\"no_respond\",\"enabled\":true}");
        assertTrue(decode0.getFaultConfig().isNoRespond(),
                "noRespond should be injected on decode engine 0");

        // ── Phase 7: Enqueue more requests — noRespond prevents response delivery
        //    but scheduled completions still drain inflight ──
        enqueueBatch(prefill, 10001, n + 1, n, decodeServices);
        awaitAllInflightZero(5_000);
        assertAllInflightZero();

        // Decode engine 0 with noRespond should still NOT have a leak
        // (scheduled completions fire regardless of noRespond)
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak despite noRespond injection");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak");

        // ── Phase 8: Clear injection via HTTP /clear_inject ──
        httpPost(controlServer.getPort(), "/clear_inject",
                "{\"port\":" + decode0.getGrpcPort() + "}");
        assertFalse(decode0.getFaultConfig().isNoRespond(),
                "noRespond should be cleared on decode engine 0");

        // ── Phase 9: Verify recovery — enqueue 10 more requests ──
        enqueueBatch(prefill, 10002, 2 * n + 1, n, decodeServices);
        awaitAllInflightZero(5_000);
        assertAllInflightZero();

        // Decode engines should still have no leak after recovery
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak after recovery");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak after recovery");

        // ── Phase 10: HTTP snapshot verification ──
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port "
                            + engine.get("port").asInt());
            // Decode engines must show leak_detected == false
            if (engine.get("port").asInt() != prefill.getGrpcPort()) {
                assertFalse(engine.get("leak_detected").asBoolean(),
                        "snapshot leak_detected should be false for decode engine on port "
                                + engine.get("port").asInt());
            }
        }
    }

    // ──────────── Cluster setup ────────────

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            prefillServices.add(service);
        }

        for (int i = 0; i < nDecode; i++) {
            int port = BASE_PORT + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            decodeServices.add(service);
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

    private void awaitInflight(JavaMockEngineCluster.FastRpcService service,
                               int min, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() >= min) {
                return;
            }
            Thread.sleep(5);
        }
        fail("inflight never reached " + min + " on port " + service.getGrpcPort()
                + ", got " + service.getInflightCount());
    }

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (services.values().stream()
                    .allMatch(s -> s.getInflightCount() == 0)) {
                return;
            }
            Thread.sleep(10);
        }
        StringBuilder sb = new StringBuilder("inflight not zero: ");
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            sb.append("port=").append(service.getGrpcPort())
                    .append(" inflight=").append(service.getInflightCount()).append(" ");
        }
        fail(sb.toString());
    }

    private void assertAllInflightZero() {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getInflightCount(),
                    "inflight should be 0 for engine on port " + service.getGrpcPort());
        }
    }

    // ──────────── Batch enqueue helper ────────────

    private void enqueueBatch(JavaMockEngineCluster.FastRpcService prefill,
                              long batchId, int startRequestId, int count,
                              List<JavaMockEngineCluster.FastRpcService> decodeEngines) {
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[count];
        for (int i = 0; i < count; i++) {
            int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
            inputs[i] = inputWithDecode(startRequestId + i, 10, decodePort);
        }
        enqueue(prefill, batch(batchId, slot(0, inputs)));
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

    private static String httpPost(int port, String path, String body) throws Exception {
        HttpResponse<String> response = HTTP_CLIENT.send(
                HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + port + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), "POST " + path + " failed");
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
                "prefill", Map.of("scale", 1.0),
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
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setRoleStr("DECODE")
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
