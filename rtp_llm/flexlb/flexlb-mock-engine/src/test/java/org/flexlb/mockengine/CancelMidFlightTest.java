package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    private static final int BASE_PORT = 62300;

    @TempDir
    Path tempDir;

    private MockEngineTestCluster cluster;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() {
        if (cluster != null) {
            cluster.close();
        }
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
        cluster = MockEngineTestCluster.start(model, BASE_PORT, nPrefill, nDecode);
        services = cluster.services();
        prefillServices = cluster.prefills();
        decodeServices = cluster.decodes();
    }

    // ──────────── Polling helpers ────────────

    private void awaitInflight(JavaMockEngineCluster.FastRpcService service,
                               int min, long timeoutMs) throws InterruptedException {
        cluster.awaitInflight(service, min, timeoutMs);
    }

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        cluster.awaitAllInflightZero(timeoutMs);
    }

    private void assertAllInflightZero() {
        cluster.assertAllInflightZero();
    }

    // ──────────── HTTP helpers ────────────

    private JsonNode snapshot() throws Exception {
        return cluster.snapshot();
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

}
