package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Matrix sweep test for the Java mock engine.
 *
 * <p>Tests 5 cluster configurations (1P/1D, 1P/2D, 2P/2D, 2P/4D, 1P/4D)
 * with 3 concurrency levels (10, 50, 100) = 15 scenarios total.
 * Each scenario verifies: zero errors, all requests completed, no inflight leak.
 * Records TTFT p50/p99 and schedule latency, outputs a summary table.
 */
class MatrixSweepTest {

    private static final int BASE_PORT = 62100;

    @TempDir
    Path tempDir;

    @ParameterizedTest(name = "{0}P/{1}D concurrency={2}")
    @CsvSource({
            "1, 1, 10", "1, 1, 50", "1, 1, 100",
            "1, 2, 10", "1, 2, 50", "1, 2, 100",
            "2, 2, 10", "2, 2, 50", "2, 2, 100",
            "2, 4, 10", "2, 4, 50", "2, 4, 100",
            "1, 4, 10", "1, 4, 50", "1, 4, 100"
    })
    void matrixSweep(int nPrefill, int nDecode, int concurrency) throws Exception {
        MockPerformanceModel model = model("10");
        runScenario(model, nPrefill, nDecode, concurrency, BASE_PORT);
    }

    // ──────────── Scenario runner ────────────

    private void runScenario(MockPerformanceModel model, int nPrefill, int nDecode,
                             int concurrency, int basePort) throws Exception {
        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.create(model, basePort, nPrefill, nDecode)) {
            Map<Integer, JavaMockEngineCluster.FastRpcService> services = cluster.services();
            List<JavaMockEngineCluster.FastRpcService> prefillServices = cluster.prefills();
            List<JavaMockEngineCluster.FastRpcService> decodeServices = cluster.decodes();

            // ── Send requests ──
            int totalErrors = 0;

            int requestsPerPrefill = concurrency / nPrefill;
            int remainder = concurrency % nPrefill;
            int requestIdCounter = 0;

            for (int i = 0; i < nPrefill; i++) {
                int count = requestsPerPrefill + (i < remainder ? 1 : 0);
                int startRequestId = requestIdCounter + 1;
                requestIdCounter += count;

                EngineRpcService.GenerateInputPB[] inputs =
                        new EngineRpcService.GenerateInputPB[count];
                for (int j = 0; j < count; j++) {
                    int decodePort = decodeServices.get(
                            (i * count + j) % nDecode).getGrpcPort();
                    inputs[j] = inputWithDecode(startRequestId + j, 10, decodePort);
                }
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefillServices.get(i), batch(1000 + i, slot(0, inputs)));
                totalErrors += response.getErrorsCount();
            }

            // ── Wait for all completions ──
            cluster.awaitCompleted(concurrency, 10_000);

            assertEquals(0, totalErrors, "enqueue errors");
            long completedCount = services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(concurrency, completedCount, "completed requests");
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertEquals(0, service.getInflightCount(), "inflight requests");
                assertFalse(service.isLeakDetected(), "leak detected");
            }
        }
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

}
