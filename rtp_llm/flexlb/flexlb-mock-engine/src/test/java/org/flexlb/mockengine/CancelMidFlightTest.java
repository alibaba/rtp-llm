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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Cancel mid-flight test for the Java mock engine.
 *
 * <p>Starts a cluster with 1 prefill + 2 decode engines, enqueues 10 requests
 * with a 200ms prefill formula, then immediately cancels 5 of them while they
 * are still in-flight. Verifies that cancelled requests are removed from
 * inflight, non-cancelled requests complete normally, no leak is detected,
 * and the enqueue response has zero errors (cancellation is not an error).
 */
class CancelMidFlightTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 62300;

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
    void cancelMidFlightRequestsNoLeakNoErrors() throws Exception {
        MockPerformanceModel model = model("200");
        startCluster(model, 1, 2);

        int n = 10;
        // Enqueue 10 requests on the single prefill engine, with decode routing
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[n];
        for (int i = 0; i < n; i++) {
            int decodePort = decodeServices.get(i % 2).getGrpcPort();
            inputs[i] = inputWithDecode(i + 1, 10, decodePort);
        }
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefillServices.get(0), batch(7000, slot(0, inputs)));

        // Error count should be 0 (no fault injection)
        assertEquals(0, response.getErrorsCount(),
                "enqueue should have 0 errors");
        assertEquals(n, response.getSuccessesCount(),
                "enqueue should have " + n + " successes");

        // Wait for at least 1 request to be in-flight before cancelling
        awaitInflight(prefillServices.get(0), 1, 1_000);

        // Cancel 5 of the 10 requests immediately
        List<Long> cancelledIds = List.of(1L, 2L, 3L, 4L, 5L);
        for (long requestId : cancelledIds) {
            prefillServices.get(0).cancel(requestId);
        }

        // Wait for all inflight to drain (scheduled completions still fire for
        // cancelled requests, decrementing pendingRequests)
        awaitAllInflightZero(10_000);

        // Assert all engines have inflight == 0
        assertAllInflightZero();

        // Assert no leak detected on any engine
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on port " + service.getGrpcPort());
        }

        // Assert cancelled requests are not in "running" state on any engine
        for (long requestId : cancelledIds) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                String state = service.getRequestStates().get(requestId);
                if (state != null) {
                    assertNotEquals("running", state,
                            "cancelled request " + requestId
                                    + " should not be 'running' on port "
                                    + service.getGrpcPort());
                }
            }
        }

        // Assert non-cancelled requests (6-10) completed normally
        List<Long> nonCancelledIds = List.of(6L, 7L, 8L, 9L, 10L);
        for (long requestId : nonCancelledIds) {
            boolean found = false;
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                if ("completed".equals(service.getRequestStates().get(requestId))) {
                    found = true;
                    break;
                }
            }
            assertTrue(found,
                    "non-cancelled request " + requestId + " should be 'completed'");
        }

        // Assert cancelled count on prefill engine is 5
        assertEquals(5, prefillServices.get(0).getCancelledCount(),
                "prefill engine should have 5 cancelled requests");

        // Assert HTTP snapshot shows inflight == 0 and leak_detected == false
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port "
                            + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port "
                            + engine.get("port").asInt());
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
        MockMasterConfig.writeWithPrefillExpression(master, formula);
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
