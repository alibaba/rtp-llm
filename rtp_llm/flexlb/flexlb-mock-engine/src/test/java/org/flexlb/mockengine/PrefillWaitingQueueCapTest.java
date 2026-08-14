package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tests for the prefill waiting-queue cap ({@code prefill.max_waiting_batches}).
 *
 * <p>With the default {@code max_prefill_concurrency} of 1, an engine holds one
 * RUNNING batch plus at most {@code max_waiting_batches} QUEUED batches. The cap
 * counts queued batches only — the running batch never counts toward it — so
 * with cap 4 the 5th enqueued batch is the last accepted (1 running + 4
 * waiting) and the 6th is rejected with an explicit backpressure error.
 *
 * <p>Also verifies that a rejection leaves no counter residue
 * (pendingRequests / waitingPrefillRequests / runningTasks) and that the queue
 * keeps draining normally after a rejection.
 */
class PrefillWaitingQueueCapTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63100;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "prefill-cap-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: cap-full rejection + semantics ────────────

    @Test
    void sixthBatchRejectedWhenFourWaiting() throws Exception {
        // 200 ms per batch so all 6 enqueues land before the first completion:
        // batch 1 runs, batches 2-5 fill the waiting queue (cap 4), batch 6 must
        // be rejected. Running batch is NOT counted toward the cap.
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("200", 4));

        for (int i = 1; i <= 5; i++) {
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(1000 + i, slot(0, input(i, 10))));
            assertEquals(1, response.getSuccessesCount(), "batch " + i + " should be accepted");
            assertEquals(0, response.getErrorsCount(), "batch " + i + " should have no errors");
        }
        assertEquals(4, prefill.getWaitingCount(), "4 requests should be waiting (queued)");
        assertEquals(4, prefill.getSnapshot().get("prefill_waiting_batches"),
                "snapshot should report 4 queued batches");

        EngineRpcService.EnqueueBatchResponsePB rejected =
                enqueue(prefill, batch(1006, slot(0, input(6, 10))));
        assertEquals(0, rejected.getSuccessesCount(), "6th batch should not be accepted");
        assertEquals(1, rejected.getErrorsCount(), "6th batch should be rejected");
        String message = rejected.getErrors(0).getErrorInfo().getErrorMessage();
        assertTrue(message.contains("prefill waiting queue full (backpressure)"),
                "rejection must carry an explicit backpressure error, got: " + message);
        assertTrue(message.contains("waiting=4 cap=4"),
                "rejection should report waiting/cap, got: " + message);

        // No residue from the rejection: still exactly 5 admitted requests inflight.
        assertEquals(5, prefill.getInflightCount(), "rejected request must not leak pendingRequests");
        assertEquals(4, prefill.getWaitingCount(), "rejected request must not leak waitingPrefillRequests");
        assertEquals(5, prefill.getRunningCount(), "rejected request must not leak runningTasks");
        assertEquals("rejected", prefill.getRequestStates().get(6L));

        // The 5 admitted batches drain to completion despite the rejection.
        awaitInflightZero(prefill, 5_000);
        assertEquals(0, prefill.getWaitingCount());
        assertEquals(0, prefill.getRunningCount());
        assertEquals(5, prefill.getCompletedCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 2: queue drains and accepts again after rejection ────────────

    @Test
    void acceptsNewBatchesAfterRejectionAndDrain() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("100", 2));

        // Fill: 1 running + 2 waiting (cap 2), 4th rejected.
        for (int i = 1; i <= 3; i++) {
            assertEquals(1, enqueue(prefill, batch(2000 + i, slot(0, input(i, 10))))
                    .getSuccessesCount());
        }
        EngineRpcService.EnqueueBatchResponsePB rejected =
                enqueue(prefill, batch(2004, slot(0, input(4, 10))));
        assertEquals(1, rejected.getErrorsCount(), "4th batch should hit the cap");

        awaitInflightZero(prefill, 5_000);

        // After the drain the queue is empty again: new enqueues are accepted
        // and complete normally (a rejection must not poison the queue).
        EngineRpcService.EnqueueBatchResponsePB retry =
                enqueue(prefill, batch(2005, slot(0, input(5, 10))));
        assertEquals(1, retry.getSuccessesCount(), "post-drain enqueue should be accepted");
        assertEquals(0, retry.getErrorsCount());

        awaitInflightZero(prefill, 5_000);
        assertEquals(4, prefill.getCompletedCount(), "3 initial + 1 retry should complete");
        assertEquals(0, prefill.getWaitingCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 3: cap <= 0 disables the gate (unbounded, legacy) ────────────

    @Test
    void nonPositiveCapDisablesBackpressure() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("50", 0));

        for (int i = 1; i <= 10; i++) {
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(3000 + i, slot(0, input(i, 10))));
            assertEquals(1, response.getSuccessesCount(), "batch " + i + " should be accepted");
            assertEquals(0, response.getErrorsCount());
        }

        awaitInflightZero(prefill, 5_000);
        assertEquals(10, prefill.getCompletedCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Harness ────────────

    private JavaMockEngineCluster.FastRpcService startPrefill(MockPerformanceModel model) {
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(BASE_PORT, service);
        return service;
    }

    private MockPerformanceModel model(String prefillFormula, int maxWaitingBatches)
            throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0, "max_waiting_batches", maxWaitingBatches),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", prefillFormula))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static void awaitInflightZero(JavaMockEngineCluster.FastRpcService service,
                                          long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("inflight not zero: inflight=" + service.getInflightCount()
                + " waiting=" + service.getWaitingCount()
                + " running=" + service.getRunningCount());
    }

    // ──────────── Protobuf builders ────────────

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
        AtomicReference<EngineRpcService.EnqueueBatchResponsePB> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        service.enqueueBatch(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.EnqueueBatchResponsePB value) {
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
                fail("enqueue response timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for enqueue response");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "enqueue response");
        return response.get();
    }
}
