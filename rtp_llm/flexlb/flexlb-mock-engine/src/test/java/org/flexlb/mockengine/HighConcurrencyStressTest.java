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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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

    private ScheduledExecutorService scheduler;
    private ExecutorService workerPool;
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
        if (workerPool != null) {
            workerPool.shutdownNow();
            workerPool.awaitTermination(3, TimeUnit.SECONDS);
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
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
        scheduler = Executors.newScheduledThreadPool(16, runnable -> {
            Thread thread = new Thread(runnable, "mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newFixedThreadPool(100, runnable -> {
            Thread thread = new Thread(runnable, "stress-test-worker");
            thread.setDaemon(true);
            return thread;
        });
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

    private long totalCompleted() {
        return services.values().stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                .sum();
    }

    private void awaitTotalCompleted(int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (totalCompleted() >= expected) {
                return;
            }
            Thread.sleep(10);
        }
        fail("expected " + expected + " completions, got " + totalCompleted());
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
