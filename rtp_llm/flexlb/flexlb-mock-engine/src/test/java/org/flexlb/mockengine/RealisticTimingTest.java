package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that the mock engine actually waits for the formula-computed duration
 * when {@code sleep_scale = 1.0} (realistic timing).
 *
 * <p>Configures a 1P/2D cluster with:
 * <ul>
 *   <li>Prefill {@code fixed_ms = 100} (100 ms per prefill batch)</li>
 *   <li>Decode {@code step_ms = 5} with {@code outputLen = 10} → 50 ms per decode request</li>
 * </ul>
 *
 * <p>Enqueues 5 requests and measures the wall-clock completion time of each.
 * Each request should take at least ~150 ms (100 ms prefill + 50 ms decode)
 * and no more than 300 ms, proving the engine honours realistic delays.
 */
class RealisticTimingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62700;

    @TempDir
    Path tempDir;

    @Test
    void realisticTimingVerifiesActualWait() throws Exception {
        MockPerformanceModel model = model();
        int nPrefill = 1;
        int nDecode = 2;
        int nRequests = 5;

        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.create(model, BASE_PORT, nPrefill, nDecode)) {
            Map<Integer, JavaMockEngineCluster.FastRpcService> services = cluster.services();
            List<JavaMockEngineCluster.FastRpcService> prefillServices = cluster.prefills();
            List<JavaMockEngineCluster.FastRpcService> decodeServices = cluster.decodes();

            // ── Enqueue 5 requests with outputLen = 10 ──
            long enqueueStartMs = System.currentTimeMillis();
            EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[nRequests];
            for (int i = 0; i < nRequests; i++) {
                int decodePort = decodeServices.get(i % nDecode).getGrpcPort();
                inputs[i] = inputWithDecode(i + 1, 10, decodePort, 10);
            }
            EngineRpcService.EnqueueBatchResponsePB response = enqueue(
                    prefillServices.get(0), batch(1000, slot(0, inputs)));

            assertEquals(nRequests, response.getSuccessesCount(),
                    "all requests should be accepted");
            assertEquals(0, response.getErrorsCount(),
                    "no enqueue errors expected");

            // ── Wait for all requests to complete ──
            cluster.awaitCompleted(nRequests, 10_000);

            // ── Collect per-request wall-clock times from decode engines ──
            List<Long> completionTimes = new ArrayList<>();
            for (JavaMockEngineCluster.FastRpcService decode : decodeServices) {
                EngineRpcService.WorkerStatusPB status = workerStatus(decode, 0);
                for (EngineRpcService.TaskInfoPB task : status.getFinishedTaskListList()) {
                    completionTimes.add(task.getEndTimeMs() - enqueueStartMs);
                }
            }

            // ── Verify timing ──
            assertEquals(nRequests, completionTimes.size(),
                    "all " + nRequests + " requests should have decode completions");

            for (int i = 0; i < completionTimes.size(); i++) {
                long time = completionTimes.get(i);
                assertTrue(time >= 100,
                        "request " + (i + 1) + " took " + time + "ms, expected >= 100ms"
                                + " (prefill should actually wait 100ms)");
                assertTrue(time <= 300,
                        "request " + (i + 1) + " took " + time + "ms, expected <= 300ms");
            }

            // ── Verify all completed ──
            long totalCompleted = services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(nRequests, totalCompleted,
                    "all requests should be completed");

            // ── Verify no inflight leak ──
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertEquals(0, service.getInflightCount(),
                        "engine port " + service.getGrpcPort()
                                + " has inflight=" + service.getInflightCount() + " (expected 0)");
                assertFalse(service.isLeakDetected(),
                        "engine port " + service.getGrpcPort() + " has leak detected");
            }

        }
    }

    // ──────────── Model helper ────────────

    /**
     * Creates a performance model with realistic timing:
     * {@code sleep_scale=1.0}, prefill {@code fixed_ms=100}, decode {@code step_ms=5}.
     *
     * <p>No FORMULA estimator is supplied through FLEXLB_CONFIG, so the model
     * falls through to {@code fixed_ms} for prefill duration.
     */
    private MockPerformanceModel model() throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0, "fixed_ms", 100),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 5.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of()))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

}
