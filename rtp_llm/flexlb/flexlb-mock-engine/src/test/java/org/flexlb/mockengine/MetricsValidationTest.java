package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.http.HttpClient;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates the {@code /metrics} and {@code /snapshot} HTTP endpoints of the
 * mock control server after processing requests.
 *
 * <p>Starts a 2P/4D cluster, enqueues 20 requests, waits for completion, then
 * verifies that:
 * <ul>
 *   <li>{@code /metrics?per_engine=true} (Python per-engine series) reports
 *       accepted/completed/running/waiting per engine, cross-checked against
 *       the per-engine values in {@code /snapshot}</li>
 *   <li>Default-mode {@code /metrics} role-aggregated accepted/completed match
 *       the snapshot-derived totals</li>
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

    /** Pattern: {@code metric_name{engine_name="...",role="...",grpc_port="12345",engine_ip="..."} value} */
    private static final Pattern PER_ENGINE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{engine_name=\"[^\"]+\",role=\"[^\"]+\",grpc_port=\"(\\d+)\",engine_ip=\"[^\"]+\"\\}\\s+(\\d+)");

    /** Pattern: {@code metric_name{role="prefill"} value} */
    private static final Pattern ROLE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{role=\"(\\w+)\"\\}\\s+(\\d+)");

    @TempDir
    Path tempDir;

    @Test
    void metricsEndpointReportsCorrectData() throws Exception {
        MockPerformanceModel model = model("10", 0.1);
        int nPrefill = 2;
        int nDecode = 4;
        int nRequests = 20;

        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.start(model, BASE_PORT, nPrefill, nDecode)) {
            Map<Integer, JavaMockEngineCluster.FastRpcService> services = cluster.services();
            List<JavaMockEngineCluster.FastRpcService> prefillServices = cluster.prefills();
            List<JavaMockEngineCluster.FastRpcService> decodeServices = cluster.decodes();

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
                    inputs[j] = inputWithDecode(startRequestId + j, 10, decodePort);
                }
                EngineRpcService.EnqueueBatchResponsePB response = enqueue(
                        prefillServices.get(i), batch(1000 + i, slot(0, inputs)));
                totalEnqueueErrors += response.getErrorsCount();
            }

            assertEquals(0, totalEnqueueErrors, "no enqueue errors expected");

            // ── Wait for all requests to complete ──
            cluster.awaitCompleted(nRequests, 10_000);

            // ── Wait for inflight to drain to 0 ──
            cluster.awaitAllInflightZero(2_000);

            // ── Fetch /metrics (Prometheus format, both emission modes) ──
            String metricsBody = MockEngineTestSupport.httpGet(cluster.controlPort(), "/metrics");
            String perEngineBody = MockEngineTestSupport.httpGet(
                    cluster.controlPort(), "/metrics?per_engine=true");
            Map<String, Map<Integer, Long>> perEngineMetrics = parsePerEngineMetrics(perEngineBody);
            Map<String, Map<String, Long>> roleMetrics = parseRoleMetrics(metricsBody);

            // ── Fetch /snapshot (JSON) ──
            JsonNode snapshotArray = cluster.snapshot();

            assertTrue(snapshotArray.isArray(), "/snapshot should return a JSON array");
            assertEquals(nPrefill + nDecode, snapshotArray.size(),
                    "/snapshot should contain all " + (nPrefill + nDecode) + " engines");

            // ── Verify per-engine metrics ──
            long totalAcceptedPrefill = 0;
            long totalCompletedDecode = 0;

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

                // ── Cross-verify /metrics per-engine Python series match /snapshot ──
                long running = engineNode.get("running").asLong();
                Map<Integer, Long> acceptedMetrics =
                        perEngineMetrics.get("mock_engine_accepted_total");
                Map<Integer, Long> completedMetrics =
                        perEngineMetrics.get("mock_engine_completed_total");
                Map<Integer, Long> runningMetrics =
                        perEngineMetrics.get("mock_engine_running");
                Map<Integer, Long> waitingMetrics =
                        perEngineMetrics.get("mock_engine_waiting");

                assertNotNull(acceptedMetrics,
                        "mock_engine_accepted_total should exist in per-engine /metrics");
                assertNotNull(completedMetrics,
                        "mock_engine_completed_total should exist in per-engine /metrics");
                assertNotNull(runningMetrics,
                        "mock_engine_running should exist in per-engine /metrics");
                assertNotNull(waitingMetrics,
                        "mock_engine_waiting should exist in per-engine /metrics");

                assertEquals(accepted, acceptedMetrics.getOrDefault(port, -1L),
                        "/metrics accepted mismatch for port " + port);
                assertEquals(completed, completedMetrics.getOrDefault(port, -1L),
                        "/metrics completed mismatch for port " + port);
                assertEquals(running, runningMetrics.getOrDefault(port, -1L),
                        "/metrics running mismatch for port " + port);
                assertEquals(0, waitingMetrics.getOrDefault(port, -1L),
                        "/metrics waiting should be 0 for port " + port);

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

            }

            // ── Verify aggregate metrics ──
            assertEquals(nRequests, totalAcceptedPrefill,
                    "total accepted across prefill engines should be " + nRequests);
            assertEquals(nRequests, totalCompletedDecode,
                    "total completed across decode engines should be " + nRequests);
            assertEquals(totalAcceptedPrefill, totalCompletedDecode,
                    "total accepted should equal total completed (no errors in pipeline)");

            // ── Verify aggregate role series (default /metrics mode) ──
            Map<String, Long> acceptedByRole = roleMetrics.get("mock_engine_accepted_total");
            Map<String, Long> completedByRole = roleMetrics.get("mock_engine_completed_total");
            assertNotNull(acceptedByRole,
                    "mock_engine_accepted_total{role=...} should exist in /metrics");
            assertNotNull(completedByRole,
                    "mock_engine_completed_total{role=...} should exist in /metrics");
            assertEquals(nRequests, acceptedByRole.getOrDefault("prefill", -1L),
                    "aggregated prefill accepted should be " + nRequests);
            assertEquals(nRequests, completedByRole.getOrDefault("decode", -1L),
                    "aggregated decode completed should be " + nRequests);

            // ── Verify Prometheus metrics contain expected Python metric names ──
            for (String metric : new String[]{
                    "mock_engine_running", "mock_engine_waiting",
                    "mock_engine_accepted_total", "mock_engine_completed_total",
                    "mock_engine_active_kv_tokens", "mock_engine_rpc_total"}) {
                assertTrue(metricsBody.contains(metric),
                        "/metrics should contain " + metric);
            }

        }
    }

    // ──────────── Prometheus parser ────────────

    /**
     * Parses per-engine Prometheus text-format metrics into a nested map:
     * {@code metricName -> (grpcPort -> value)}.
     */
    private Map<String, Map<Integer, Long>> parsePerEngineMetrics(String body) {
        Map<String, Map<Integer, Long>> result = new LinkedHashMap<>();
        for (String line : body.split("\n")) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Matcher matcher = PER_ENGINE_METRIC_PATTERN.matcher(line);
            if (matcher.matches()) {
                String metricName = matcher.group(1);
                int port = Integer.parseInt(matcher.group(2));
                long value = Long.parseLong(matcher.group(3));
                result.computeIfAbsent(metricName, k -> new LinkedHashMap<>())
                        .put(port, value);
            }
        }
        return result;
    }

    /**
     * Parses role-aggregated Prometheus text-format metrics into a nested map:
     * {@code metricName -> (role -> value)}.
     */
    private Map<String, Map<String, Long>> parseRoleMetrics(String body) {
        Map<String, Map<String, Long>> result = new LinkedHashMap<>();
        for (String line : body.split("\n")) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Matcher matcher = ROLE_METRIC_PATTERN.matcher(line);
            if (matcher.matches()) {
                result.computeIfAbsent(matcher.group(1), k -> new LinkedHashMap<>())
                        .put(matcher.group(2), Long.parseLong(matcher.group(3)));
            }
        }
        return result;
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula, double sleepScale) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula, sleepScale);
    }

}
