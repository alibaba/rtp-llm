package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tests for direct-path (generate_stream / NON_BATCH) prefill coalescing
 * ({@code prefill.direct_batch_size_max}).
 *
 * <p>With max_prefill_concurrency 1 and a 100 ms constant prefill formula, the
 * first generate_stream occupies the only slot; later arrivals park in the
 * direct queue. When the running batch completes, the drain coalesces up to
 * {@code direct_batch_size_max} queued requests into ONE batch — production
 * engines run prefill continuous batching, so engine-side drain scales with
 * batch size instead of one request per prefillMs.
 */
class DirectPrefillCoalescingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63200;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private JavaMockEngineCluster.ClusterStats stats;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "direct-coalesce-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        stats = new JavaMockEngineCluster.ClusterStats();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: queued direct requests coalesce into one batch ────────────

    @Test
    void queuedDirectRequestsCoalesceIntoOneBatch() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill =
                startPrefill(model(0, 4));

        // Fire all 6 generate_stream calls up front (admission is synchronous,
        // responses arrive on the response executor): #1 takes the only slot,
        // #2-#6 park in the direct queue within #1's 100 ms prefill window.
        // A serial await-per-request loop would never leave anything queued.
        List<CompletableFuture<Throwable>> futures = new ArrayList<>();
        for (int i = 1; i <= 6; i++) {
            futures.add(generateAsync(prefill, input(1000 + i, 10)));
        }
        for (int i = 0; i < futures.size(); i++) {
            assertNull(futures.get(i).get(10, TimeUnit.SECONDS),
                    "request " + (i + 1) + " should be admitted");
        }

        awaitCompleted(prefill, 6, 5_000);
        assertEquals(6, prefill.getCompletedCount());
        assertEquals(0, prefill.getWaitingCount());
        assertEquals(0, prefill.getInflightCount());

        // Batch structure: [1] + [2,3,4,5] + [6] — coalescing is observable in
        // the per-engine prefill batch stats (java_mock_stats batch counters).
        assertEquals(3, stats.prefillBatches.sum(),
                "expected 1 + 4 + 1 batches after coalescing");
        assertEquals(6, stats.prefillBatchRequests.sum());
        assertEquals(4, stats.maxPrefillBatchSize.get(),
                "the coalesced batch should hold 4 requests (direct_batch_size_max)");
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 2: direct_batch_size_max=1 restores one-request batches ────────────

    @Test
    void maxOfOneRestoresSingleRequestBatches() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill =
                startPrefill(model(0, 1));

        List<CompletableFuture<Throwable>> futures = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            futures.add(generateAsync(prefill, input(2000 + i, 10)));
        }
        for (CompletableFuture<Throwable> future : futures) {
            assertNull(future.get(10, TimeUnit.SECONDS));
        }

        awaitCompleted(prefill, 5, 5_000);
        assertEquals(5, prefill.getCompletedCount());
        assertEquals(5, stats.prefillBatches.sum(),
                "legacy behaviour: every direct request runs as its own batch");
        assertEquals(5, stats.prefillBatchRequests.sum());
        assertEquals(1, stats.maxPrefillBatchSize.get());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 3: request-level backpressure cap ────────────

    @Test
    void directWaitingCapRejectsInRequests() throws Exception {
        // max_waiting_batches=1 x direct_batch_size_max=2 -> cap of 2 queued
        // requests: #1 runs, #2-#3 queue, #4 must be rejected.
        JavaMockEngineCluster.FastRpcService prefill =
                startPrefill(model(1, 2));

        // #1 runs, #2-#3 fill the 2-request cap — all three admissions land
        // inside #1's 100 ms prefill window, so the 4th must be rejected.
        List<CompletableFuture<Throwable>> admitted = new ArrayList<>();
        for (long requestId = 3001; requestId <= 3003; requestId++) {
            admitted.add(generateAsync(prefill, input(requestId, 10)));
        }

        Throwable rejected = generateAsync(prefill, input(3004, 10))
                .get(5, TimeUnit.SECONDS);
        assertNotNull(rejected, "4th request should hit the request-level cap");
        assertTrue(rejected.getMessage().contains("prefill waiting queue full"),
                "rejection must carry the backpressure message, got: " + rejected);
        assertTrue(rejected.getMessage().contains("cap=2"),
                "rejection should report the request-level cap, got: " + rejected);

        for (CompletableFuture<Throwable> future : admitted) {
            assertNull(future.get(10, TimeUnit.SECONDS));
        }

        // No residue from the rejection: 3 admitted requests still drain.
        awaitCompleted(prefill, 3, 5_000);
        assertEquals(3, prefill.getCompletedCount());
        assertEquals(0, prefill.getWaitingCount());
        assertEquals(0, prefill.getInflightCount());
        assertEquals("rejected", prefill.getRequestStates().get(3004L));
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Harness ────────────

    private JavaMockEngineCluster.FastRpcService startPrefill(MockPerformanceModel model) {
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 100, stats);
        services.put(BASE_PORT, service);
        return service;
    }

    private MockPerformanceModel model(int maxWaitingBatches, int directBatchSizeMax)
            throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of(
                        "scale", 1.0,
                        "max_waiting_batches", maxWaitingBatches,
                        "direct_batch_size_max", directBatchSizeMax),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, "100");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static void awaitCompleted(JavaMockEngineCluster.FastRpcService service,
                                       int expected, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getCompletedCount() >= expected && service.getInflightCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("not drained: completed=" + service.getCompletedCount()
                + " inflight=" + service.getInflightCount()
                + " waiting=" + service.getWaitingCount());
    }

    private static EngineRpcService.GenerateInputPB input(long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    /**
     * Fires generate_stream and returns a future completed with the stream
     * error (or null on success). Admission runs synchronously on the caller
     * thread; the single-frame response / error arrives on the response
     * executor, so callers can fire many requests before awaiting any.
     */
    private static CompletableFuture<Throwable> generateAsync(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.GenerateInputPB input) {
        CompletableFuture<Throwable> future = new CompletableFuture<>();
        service.generateStreamCall(input, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
            }

            @Override
            public void onError(Throwable throwable) {
                future.complete(throwable);
            }

            @Override
            public void onCompleted() {
                future.complete(null);
            }
        });
        return future;
    }
}
