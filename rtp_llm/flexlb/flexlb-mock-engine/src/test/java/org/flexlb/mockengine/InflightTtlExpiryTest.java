package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    private static final int BASE_PORT = 62500;

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
    void inflightTtlExpiryAndRecovery() throws Exception {
        MockPerformanceModel model = model("1000"); // 1s prefill to keep requests inflight briefly
        startCluster(model, 1, 2);

        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode0 = decodeServices.get(0);
        JavaMockEngineCluster.FastRpcService decode1 = decodeServices.get(1);

        int n = 10;

        // ── Phase 1: Enqueue 10 requests with decode routing → stay inflight ──
        cluster.enqueueBatch(prefill, 10000, 1, n, decodeServices);
        cluster.awaitInflight(prefill, 1, 1_000);
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
        cluster.awaitAllInflightZero(5_000);
        cluster.assertAllInflightZero();

        // ── Phase 5: Decode engines remain clean — TTL cleanup is not a leak ──
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak after TTL cleanup");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak after TTL cleanup");

        // ── Phase 6: Inject noRespond on decode engine 0 via HTTP /inject ──
        MockEngineTestSupport.httpPost(cluster.controlPort(), "/inject",
                "{\"port\":" + decode0.getGrpcPort()
                        + ",\"type\":\"no_respond\",\"enabled\":true}");
        assertTrue(decode0.getFaultConfig().isNoRespond(),
                "noRespond should be injected on decode engine 0");

        // ── Phase 7: Enqueue more requests — noRespond prevents response delivery
        //    but scheduled completions still drain inflight ──
        cluster.enqueueBatch(prefill, 10001, n + 1, n, decodeServices);
        cluster.awaitAllInflightZero(5_000);
        cluster.assertAllInflightZero();

        // Decode engine 0 with noRespond should still NOT have a leak
        // (scheduled completions fire regardless of noRespond)
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak despite noRespond injection");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak");

        // ── Phase 8: Clear injection via HTTP /clear_inject ──
        MockEngineTestSupport.httpPost(cluster.controlPort(), "/clear_inject",
                "{\"port\":" + decode0.getGrpcPort() + "}");
        assertFalse(decode0.getFaultConfig().isNoRespond(),
                "noRespond should be cleared on decode engine 0");

        // ── Phase 9: Verify recovery — enqueue 10 more requests ──
        cluster.enqueueBatch(prefill, 10002, 2 * n + 1, n, decodeServices);
        cluster.awaitAllInflightZero(5_000);
        cluster.assertAllInflightZero();

        // Decode engines should still have no leak after recovery
        assertFalse(decode0.isLeakDetected(),
                "decode engine 0 should not have leak after recovery");
        assertFalse(decode1.isLeakDetected(),
                "decode engine 1 should not have leak after recovery");

        // ── Phase 10: HTTP snapshot verification ──
        JsonNode snapshot = cluster.snapshot();
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
        cluster = MockEngineTestCluster.start(model, BASE_PORT, nPrefill, nDecode);
        services = cluster.services();
        prefillServices = cluster.prefills();
        decodeServices = cluster.decodes();
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

}
