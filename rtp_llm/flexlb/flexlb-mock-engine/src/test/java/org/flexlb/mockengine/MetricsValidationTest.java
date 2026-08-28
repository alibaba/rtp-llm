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

            // ── Fetch /metrics (Prometheus format) ──
            String metricsBody = MockEngineTestSupport.httpGet(cluster.controlPort(), "/metrics");
            Map<String, Map<Integer, Long>> prometheusMetrics = parsePrometheusMetrics(metricsBody);

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

            }

            // ── Verify aggregate metrics ──
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
        return MockEngineTestSupport.performanceModel(tempDir, formula, sleepScale);
    }

}
