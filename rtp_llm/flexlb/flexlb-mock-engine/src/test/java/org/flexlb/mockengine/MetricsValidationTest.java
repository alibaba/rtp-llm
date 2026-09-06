package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Validates the {@code /metrics} and {@code /snapshot} HTTP endpoints of the
 * mock control server after processing requests.
 *
 * <p>Starts a 2P/4D cluster, enqueues 20 requests, waits for completion, then
 * verifies that:
 * <ul>
 *   <li>{@code /metrics} (Prometheus format) reports correct accepted/completed/inflight per engine</li>
 *   <li>{@code /snapshot} (JSON) reports {@code leak_detected=false} for all engines</li>
 *   <li>Aggregate accepted across prefill engines = 20</li>
 *   <li>Aggregate completed across decode engines = 20</li>
 *   <li>All engines have inflight = 0 and no errors</li>
 * </ul>
 */
class MetricsValidationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 62800;

    /** Pattern: {@code metric_name{port="12345",role="PREFILL"} value} */
    private static final Pattern METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{port=\"(\\d+)\",role=\"(\\w+)\"\\}\\s+(\\d+)");

    @TempDir
    Path tempDir;

    @Test
    void metricsEndpointReportsCorrectData() throws Exception {
        MockPerformanceModel model = model("10", 0.1);
        int nPrefill = 2;
        int nDecode = 4;
        int nRequests = 20;

        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        MockControlServer controlServer = null;

        try {
            // ── Create engines ──
            List<JavaMockEngineCluster.FastRpcService> prefillServices = new ArrayList<>();
            for (int i = 0; i < nPrefill; i++) {
                int port = BASE_PORT + i;
                JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                        "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        port, services, scheduler, model, 100,
                        new JavaMockEngineCluster.ClusterStats());
                services.put(port, service);
                prefillServices.add(service);
            }

            List<JavaMockEngineCluster.FastRpcService> decodeServices = new ArrayList<>();
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

            // ── Enqueue 20 requests distributed across 2 prefill engines ──
            int requestsPerPrefill = nRequests / nPrefill;
            int totalEnqueueErrors = 0;
            int requestIdCounter = 0;

            for (int i = 0; i < nPrefill; i++) {
                int count = requestsPerPrefill;
                int startRequestId = requestIdCounter + 1;
                requestIdCounter += count;

                EngineRpcService.GenerateInputPB[] inputs =
                        new EngineRpcService.GenerateInputPB[count];
                for (int j = 0; j < count; j++) {
                    int decodePort = decodeServices.get(
                            (i * count + j) % nDecode).getGrpcPort();
                    inputs[j] = inputWithDecode(String.valueOf(startRequestId + j), 10, decodePort);
                }
                EngineRpcService.EnqueueBatchResponsePB response = enqueue(
                        prefillServices.get(i), batch(1000 + i, slot(0, inputs)));
                totalEnqueueErrors += response.getErrorsCount();
            }

            assertEquals(0, totalEnqueueErrors, "no enqueue errors expected");

            // ── Wait for all requests to complete ──
            awaitTotalCompleted(services, nRequests, 10_000);

            // ── Wait for inflight to drain to 0 ──
            awaitInflightDrain(services, 2_000);

            // ── Fetch /metrics (Prometheus format) ──
            String metricsBody = httpGet(controlServer.getPort(), "/metrics");
            Map<String, Map<Integer, Long>> prometheusMetrics = parsePrometheusMetrics(metricsBody);

            // ── Fetch /snapshot (JSON) ──
            String snapshotBody = httpGet(controlServer.getPort(), "/snapshot");
            JsonNode snapshotArray = MAPPER.readTree(snapshotBody).path("engines");

            assertTrue(snapshotArray.isArray(), "/snapshot should return a JSON array");
            assertEquals(nPrefill + nDecode, snapshotArray.size(),
                    "/snapshot should contain all " + (nPrefill + nDecode) + " engines");

            // ── Verify per-engine metrics ──
            long totalAcceptedPrefill = 0;
            long totalCompletedDecode = 0;

            System.out.println();
            System.out.println("=== Metrics Validation Test ===");
            System.out.println("Cluster: " + nPrefill + "P / " + nDecode + "D, " + nRequests + " requests");
            System.out.println();
            System.out.printf("%-8s %-8s %-10s %-10s %-10s %-12s %-14s%n",
                    "PORT", "ROLE", "ACCEPTED", "COMPLETED", "INFLIGHT", "LEAK_DETECTED", "METRICS_MATCH");
            System.out.println("─".repeat(80));

            for (JsonNode engineNode : snapshotArray) {
                int port = engineNode.get("port").asInt();
                String role = engineNode.get("role").asText();
                long accepted = engineNode.get("accepted").asLong();
                long completed = engineNode.get("completed").asLong();
                long inflight = engineNode.get("inflight").asLong();
                boolean leakDetected = engineNode.get("leak_detected").asBoolean();

                // ── Verify inflight is 0 (all completed) ──
                assertEquals(0, inflight,
                        "engine port " + port + " (" + role
                                + ") inflight should be 0, got " + inflight);

                // ── Verify no leak ──
                assertFalse(leakDetected,
                        "engine port " + port + " (" + role
                                + ") leak_detected should be false");

                // ── Cross-verify /metrics values match /snapshot ──
                Map<Integer, Long> acceptedMetrics =
                        prometheusMetrics.get("mock_engine_accepted_total");
                Map<Integer, Long> completedMetrics =
                        prometheusMetrics.get("mock_engine_completed_total");
                Map<Integer, Long> inflightMetrics =
                        prometheusMetrics.get("mock_engine_inflight_count");

                assertNotNull(acceptedMetrics,
                        "mock_engine_accepted_total should exist in /metrics");
                assertNotNull(completedMetrics,
                        "mock_engine_completed_total should exist in /metrics");
                assertNotNull(inflightMetrics,
                        "mock_engine_inflight_count should exist in /metrics");

                assertEquals(accepted, acceptedMetrics.getOrDefault(port, -1L),
                        "/metrics accepted mismatch for port " + port);
                assertEquals(completed, completedMetrics.getOrDefault(port, -1L),
                        "/metrics completed mismatch for port " + port);
                assertEquals(0, inflightMetrics.getOrDefault(port, -1L),
                        "/metrics inflight should be 0 for port " + port);

                // ── Aggregate ──
                if ("prefill".equals(role)) {
                    assertTrue(accepted > 0,
                            "prefill engine port " + port + " should have accepted > 0");
                    totalAcceptedPrefill += accepted;
                }
                if ("decode".equals(role)) {
                    assertTrue(completed > 0,
                            "decode engine port " + port + " should have completed > 0");
                    totalCompletedDecode += completed;
                }

                System.out.printf("%-8d %-8s %-10d %-10d %-10d %-12s %-14s%n",
                        port, role, accepted, completed, inflight,
                        leakDetected ? "YES" : "NO", "OK");
            }

            // ── Verify aggregate metrics ──
            System.out.println("─".repeat(80));
            System.out.printf("TOTAL    %-8s %-10d %-10d%n", "",
                    totalAcceptedPrefill, totalCompletedDecode);

            assertEquals(nRequests, totalAcceptedPrefill,
                    "total accepted across prefill engines should be " + nRequests);
            assertEquals(nRequests, totalCompletedDecode,
                    "total completed across decode engines should be " + nRequests);
            assertEquals(totalAcceptedPrefill, totalCompletedDecode,
                    "total accepted should equal total completed (no errors in pipeline)");

            // ── Verify Prometheus metrics contain expected metric names ──
            assertTrue(metricsBody.contains("mock_engine_running_tasks"),
                    "/metrics should contain mock_engine_running_tasks");
            assertTrue(metricsBody.contains("mock_engine_accepted_total"),
                    "/metrics should contain mock_engine_accepted_total");
            assertTrue(metricsBody.contains("mock_engine_completed_total"),
                    "/metrics should contain mock_engine_completed_total");
            assertTrue(metricsBody.contains("mock_engine_inflight_count"),
                    "/metrics should contain mock_engine_inflight_count");

            System.out.println("\nMetrics validation test PASSED.");
            System.out.println("  Total accepted (prefill): " + totalAcceptedPrefill);
            System.out.println("  Total completed (decode): " + totalCompletedDecode);
            System.out.println("  All engines: inflight=0, leak_detected=false, no errors");
        } finally {
            if (controlServer != null) {
                controlServer.stop();
            }
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    // ──────────── Prometheus parser ────────────

    /**
     * Parses Prometheus text-format metrics into a nested map:
     * {@code metricName -> (port -> value)}.
     */
    private Map<String, Map<Integer, Long>> parsePrometheusMetrics(String body) {
        Map<String, Map<Integer, Long>> result = new LinkedHashMap<>();
        for (String line : body.split("\n")) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Matcher matcher = METRIC_PATTERN.matcher(line);
            if (matcher.matches()) {
                String metricName = matcher.group(1);
                int port = Integer.parseInt(matcher.group(2));
                long value = Long.parseLong(matcher.group(4));
                result.computeIfAbsent(metricName, k -> new LinkedHashMap<>())
                        .put(port, value);
            }
        }
        return result;
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula, double sleepScale) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", sleepScale,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, formula);
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    // ──────────── Polling helpers ────────────

    private void awaitTotalCompleted(
            Map<Integer, JavaMockEngineCluster.FastRpcService> services,
            int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            long completed = services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            if (completed >= expected) {
                return;
            }
            Thread.sleep(10);
        }
    }

    private void awaitInflightDrain(
            Map<Integer, JavaMockEngineCluster.FastRpcService> services,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            boolean allDrained = true;
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                if (service.getInflightCount() != 0) {
                    allDrained = false;
                    break;
                }
            }
            if (allDrained) {
                return;
            }
            Thread.sleep(5);
        }
    }

    // ──────────── HTTP helpers ────────────

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

    // ──────────── Protobuf builders ────────────

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            String requestId, int inputTokens, int decodePort) {
        EngineRpcService.GenerateInputPB.Builder input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
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
