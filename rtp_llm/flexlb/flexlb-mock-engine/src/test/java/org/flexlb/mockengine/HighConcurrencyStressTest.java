package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.http.HttpClient;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * High-concurrency stress test for the Java mock engine.
 *
 * <p>Starts a cluster with 2 prefill + 4 decode engines, uses a 10ms prefill
 * formula, and sends 500 requests with concurrency 100 (via a fixed thread
 * pool). Verifies that all requests complete with zero errors, no inflight
 * leak is detected, TTFT p99 stays under 1000ms, and load is reasonably
 * balanced across decode engines (no single engine handles more than 60%
 * of decode requests).
 */
class HighConcurrencyStressTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 62600;

    @TempDir
    Path tempDir;

    private MockEngineTestCluster cluster;
    private ExecutorService workerPool;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (cluster != null) {
            cluster.close();
        }
        if (workerPool != null) {
            workerPool.shutdownNow();
            workerPool.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    @Test
    void highConcurrencyStressNoLeakNoErrors() throws Exception {
        MockPerformanceModel model = model("10"); // 10ms prefill for fast completion
        startCluster(model, 2, 4);

        int totalRequests = 500;
        int concurrency = 100;
        int batchSize = 5; // each task enqueues a batch of 5 requests
        int numBatches = totalRequests / batchSize; // 100 batches

        AtomicInteger totalErrors = new AtomicInteger(0);
        AtomicInteger totalSuccesses = new AtomicInteger(0);

        // ── Phase 1: Send 500 requests with concurrency 100 ──
        long startNs = System.nanoTime();
        CountDownLatch allEnqueued = new CountDownLatch(numBatches);

        for (int b = 0; b < numBatches; b++) {
            final int batchIndex = b;
            final long batchId = 20000 + b;
            final int startRequestId = b * batchSize + 1;
            final JavaMockEngineCluster.FastRpcService prefill =
                    prefillServices.get(batchIndex % prefillServices.size());

            workerPool.submit(() -> {
                try {
                    EngineRpcService.GenerateInputPB[] inputs =
                            new EngineRpcService.GenerateInputPB[batchSize];
                    int decodeStart = batchIndex % decodeServices.size();
                    for (int i = 0; i < batchSize; i++) {
                        int decodePort = decodeServices
                                .get((decodeStart + i) % decodeServices.size())
                                .getGrpcPort();
                        inputs[i] = inputWithDecode(startRequestId + i, 10, decodePort);
                    }
                    EngineRpcService.EnqueueBatchResponsePB response =
                            enqueue(prefill, batch(batchId, slot(0, inputs)));
                    totalErrors.addAndGet(response.getErrorsCount());
                    totalSuccesses.addAndGet(response.getSuccessesCount());
                } catch (Throwable t) {
                    totalErrors.incrementAndGet();
                } finally {
                    allEnqueued.countDown();
                }
            });
        }

        // Wait for all enqueue calls to return
        assertTrue(allEnqueued.await(15, TimeUnit.SECONDS),
                "all enqueue calls should complete within 15s");

        // ── Phase 2: Wait for all 500 completions ──
        awaitTotalCompleted(totalRequests, 15_000);
        long endNs = System.nanoTime();

        // ── Phase 3: Verify no inflight leak ──
        assertAllInflightZero();
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on port " + service.getGrpcPort());
        }

        // ── Phase 4: Verify error_count == 0 and all 500 completed ──
        assertEquals(0, totalErrors.get(),
                "total errors should be 0, got " + totalErrors.get());
        assertEquals(totalRequests, totalSuccesses.get(),
                "total successes should be " + totalRequests
                        + ", got " + totalSuccesses.get());
        long completed = totalCompleted();
        assertEquals(totalRequests, completed,
                "total completed should be " + totalRequests + ", got " + completed);

        // ── Phase 5: Verify TTFT p99 < 1000ms ──
        long totalMs = TimeUnit.NANOSECONDS.toMillis(endNs - startNs);
        assertTrue(totalMs < 1000,
                "TTFT p99 (approximated by total completion time) should be < 1000ms, got "
                        + totalMs + "ms");

        // ── Phase 6: Verify load balancing across decode engines ──
        // No single engine should handle > 60% of decode requests
        long totalDecodeCompleted = decodeServices.stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                .sum();
        assertEquals(totalRequests, totalDecodeCompleted,
                "total decode completions should equal " + totalRequests
                        + ", got " + totalDecodeCompleted);

        double threshold = totalRequests * 0.60;
        for (int i = 0; i < decodeServices.size(); i++) {
            long engineCompleted = decodeServices.get(i).getCompletedCount();
            assertTrue(engineCompleted <= threshold,
                    "decode engine " + i + " (port " + decodeServices.get(i).getGrpcPort()
                            + ") handled " + engineCompleted + " requests ("
                            + (engineCompleted * 100.0 / totalRequests) + "%), "
                            + "which exceeds 60% threshold");
        }

        // ── Phase 7: HTTP snapshot verification ──
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
        cluster = MockEngineTestCluster.start(model, BASE_PORT, nPrefill, nDecode, 16);
        workerPool = Executors.newFixedThreadPool(100, runnable -> {
            Thread thread = new Thread(runnable, "stress-test-worker");
            thread.setDaemon(true);
            return thread;
        });
        services = cluster.services();
        prefillServices = cluster.prefills();
        decodeServices = cluster.decodes();
    }

    // ──────────── Polling helpers ────────────

    private long totalCompleted() {
        return cluster.totalCompleted();
    }

    private void awaitTotalCompleted(int expected, long timeoutMs) throws InterruptedException {
        cluster.awaitCompleted(expected, timeoutMs);
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
