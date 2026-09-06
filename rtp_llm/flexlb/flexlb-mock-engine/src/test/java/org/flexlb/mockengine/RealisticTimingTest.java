package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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

            // ── Enqueue 5 requests with outputLen = 10 ──
            long enqueueStartMs = System.currentTimeMillis();
            EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[nRequests];
            for (int i = 0; i < nRequests; i++) {
                int decodePort = decodeServices.get(i % nDecode).getGrpcPort();
                inputs[i] = inputWithDecode(String.valueOf(i + 1), 10, decodePort, 10);
            }
            EngineRpcService.EnqueueBatchResponsePB response = enqueue(
                    prefillServices.get(0), batch(1000, slot(0, inputs)));

            assertEquals(nRequests, response.getSuccessesCount(),
                    "all requests should be accepted");
            assertEquals(0, response.getErrorsCount(),
                    "no enqueue errors expected");

            // ── Wait for all requests to complete ──
            awaitTotalCompleted(services, nRequests, 10_000);

            // ── Collect per-request wall-clock times from decode engines ──
            List<Long> completionTimes = new ArrayList<>();
            for (JavaMockEngineCluster.FastRpcService decode : decodeServices) {
                EngineRpcService.WorkerStatusPB workerStatus = status(decode);
                for (EngineRpcService.TaskInfoPB task : workerStatus.getFinishedTaskListList()) {
                    completionTimes.add(task.getEndTimeMs() - enqueueStartMs);
                }
            }

            // ── Verify timing ──
            assertEquals(nRequests, completionTimes.size(),
                    "all " + nRequests + " requests should have decode completions");

            System.out.println();
            System.out.println("=== Realistic Timing Test (sleep_scale=1.0) ===");
            System.out.println("Expected: prefill=100ms + decode=50ms = ~150ms per request");
            for (int i = 0; i < completionTimes.size(); i++) {
                long time = completionTimes.get(i);
                System.out.printf("  request %d: %dms%n", i + 1, time);
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

            System.out.println("\nRealistic timing test PASSED — all " + nRequests
                    + " requests completed within 100-300ms bounds.");
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

    // ──────────── Protobuf builders ────────────

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            String requestId, int inputTokens, int decodePort, int outputLen) {
        EngineRpcService.GenerateInputPB.Builder input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(outputLen)
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

    private static EngineRpcService.WorkerStatusPB status(
            JavaMockEngineCluster.FastRpcService service) {
        return unary(observer -> service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(0)
                        .build(),
                observer));
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
