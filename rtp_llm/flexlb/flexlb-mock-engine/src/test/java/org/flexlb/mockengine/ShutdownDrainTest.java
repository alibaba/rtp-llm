package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
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
import java.util.function.BooleanSupplier;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Shutdown drain tests (false LEAK alarm fix).
 *
 * <p>Stopping a load test leaves long decode simulations in flight (up to
 * 20000 tokens x 4.5ms ~ 90s); the old shutdown hook killed the process
 * without draining, and the last checkLeakDrain cycles misreported the
 * still-pending requests as LEAK DETECTED. These tests verify that
 * {@code drainAndShutdown()} cancels all in-flight requests through the
 * existing cancel() bookkeeping (all counters net to zero, both pending
 * queues empty) and that checkLeakDrain never sets leak_detected while
 * shutting down, without weakening runtime leak detection.
 */
class ShutdownDrainTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63200;

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ────────── Test 1: long decode in-flight → drain zeroes everything ──────────

    @Test
    void drainZeroesCountersWithLongDecodeInflight() throws Exception {
        MockPerformanceModel model = model("10"); // fast prefill hand-off
        // decodeMaxConcurrency=4 so 10 requests per decode engine leave 4
        // running + 6 parked in the decode pending queue — the drain must
        // clear both populations.
        startCluster(model, 1, 2, 4);
        for (JavaMockEngineCluster.FastRpcService decode : decodeServices) {
            // ~60s per decode step: completions cannot fire during the test,
            // reproducing the stop-time in-flight backlog.
            decode.getPerformance().setOverrideDecodeStepMs(60_000.0);
        }

        int n = 20;
        enqueueBatch(prefillServices.get(0), 9000, 1, n, decodeServices);

        // Wait until both decode engines have running AND queued requests.
        for (JavaMockEngineCluster.FastRpcService decode : decodeServices) {
            await(() -> decode.getActiveDecodeCount() >= 4
                            && decode.getDecodePendingQueueDepth() >= 1, 5_000,
                    "decode engine " + decode.getGrpcPort() + " never built a backlog: active="
                            + decode.getActiveDecodeCount()
                            + " queued=" + decode.getDecodePendingQueueDepth());
        }
        assertTrue(totalInflight() > 0, "requests should be in flight before drain");

        // Drain every engine, as the JVM shutdown hook does.
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.drainAndShutdown();
        }

        // All counters net to zero immediately (no 90s natural drain).
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            String port = "port " + service.getGrpcPort();
            assertTrue(service.isShuttingDown(), port + " should be shutting down");
            assertTrue(service.isStopped(), port + " should be stopped");
            assertEquals(0, service.getInflightCount(), port + " pendingRequests");
            assertEquals(0, service.getRunningCount(), port + " runningTasks");
            assertEquals(0, service.getActiveDecodeCount(), port + " activeDecodeRequests");
            assertEquals(0, service.getActiveKvTokens(), port + " activeKvTokens");
            assertEquals(0, service.getWaitingCount(), port + " waitingPrefillRequests");
            assertEquals(0, service.getActivePrefillRequestCount(), port + " activePrefillRequests");
            assertEquals(0, service.getActivePrefillBatchCount(), port + " activePrefillBatches");
            assertEquals(0, service.getDecodePendingQueueDepth(), port + " decodePendingQueue");
            assertEquals(0, service.getPrefillPendingQueueDepth(), port + " prefillPendingQueue");
        }

        // The in-flight requests were cancelled, not leaked.
        long cancelled = services.values().stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getCancelledCount).sum();
        assertTrue(cancelled >= n, "expected >= " + n + " cancellations, got " + cancelled);

        // checkLeakDrain must not report while shutting down, even with zero grace.
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.checkLeakDrain(0);
            assertFalse(service.isLeakDetected(),
                    "leak must NOT be detected during shutdown on port " + service.getGrpcPort());
        }
    }

    // ────────── Test 2: queued prefill batches are dropped by the drain ──────────

    @Test
    void drainDropsQueuedPrefillBatches() throws Exception {
        MockPerformanceModel model = model("5000"); // 5s prefill keeps batches queued
        startCluster(model, 1, 1, 4);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        // First batch occupies the single prefill lane; the next two queue up.
        enqueueBatch(prefill, 9100, 1, 4, decodeServices);
        enqueueBatch(prefill, 9101, 5, 4, decodeServices);
        enqueueBatch(prefill, 9102, 9, 4, decodeServices);
        await(() -> prefill.getPrefillPendingQueueDepth() >= 2, 2_000,
                "prefill pending queue never filled: depth=" + prefill.getPrefillPendingQueueDepth());
        assertTrue(prefill.getWaitingCount() >= 8, "queued prefill requests expected");

        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.drainAndShutdown();
        }

        // Queue-owned state is zeroed synchronously by the drain.
        assertEquals(0, prefill.getInflightCount(), "pendingRequests");
        assertEquals(0, prefill.getRunningCount(), "runningTasks");
        assertEquals(0, prefill.getWaitingCount(), "waitingPrefillRequests");
        assertEquals(0, prefill.getPrefillPendingQueueDepth(), "prefillPendingQueue");

        // The RUNNING batch's slot counters are freed by its already-scheduled
        // completion callback (all members cancelled → no further decrements).
        await(() -> prefill.getActivePrefillBatchCount() == 0
                        && prefill.getActivePrefillRequestCount() == 0, 7_000,
                "running prefill batch slot never freed: batches="
                        + prefill.getActivePrefillBatchCount()
                        + " reqs=" + prefill.getActivePrefillRequestCount());
        // No negative drift from the completion racing the drain.
        assertEquals(0, prefill.getInflightCount(), "pendingRequests after completion");
        assertEquals(0, prefill.getWaitingCount(), "waitingPrefillRequests after completion");

        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.checkLeakDrain(0);
            assertFalse(service.isLeakDetected(),
                    "leak must NOT be detected during shutdown on port " + service.getGrpcPort());
        }
    }

    // ────────── Test 3: new work is rejected after the drain ──────────

    @Test
    void drainRejectsNewAdmissions() throws Exception {
        MockPerformanceModel model = model("10");
        startCluster(model, 1, 1, 4);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.drainAndShutdown();
        }

        // enqueueBatch on a stopped engine returns an empty response (no
        // successes, nothing admitted) — existing /stop_engine semantics.
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefill, batch(9200, slot(0, inputWithDecode(100, 10, decode.getGrpcPort()))));
        assertEquals(0, response.getSuccessesCount(), "stopped engine must not admit requests");
        assertEquals(0, prefill.getInflightCount(), "no residue after rejected enqueue");

        // generateStreamCall on a stopped engine errors out.
        AtomicReference<Throwable> streamError = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        decode.generateStreamCall(input(101, 10), new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
            }

            @Override
            public void onError(Throwable throwable) {
                streamError.set(throwable);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        assertTrue(latch.await(5, TimeUnit.SECONDS), "generateStreamCall should answer");
        assertNotNull(streamError.get(), "stopped decode engine should reject generate_stream");
        assertEquals(0, decode.getInflightCount(), "no residue after rejected stream");
    }

    // ────────── Test 4: runtime leak detection is NOT weakened ──────────

    @Test
    void runtimeLeakDetectionStillFires() throws Exception {
        MockPerformanceModel model = model("3000"); // 3s prefill keeps requests in-flight
        startCluster(model, 1, 1, 4);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        // Prefill-only requests (no decode routing) stay in-flight.
        enqueue(prefill, batch(9300, slot(0, input(1, 10), input(2, 10))));
        await(() -> prefill.getInflightCount() > 0, 1_000, "requests never got in-flight");

        // NOT shutting down + grace expired → the real leak check still trips.
        prefill.checkLeakDrain(0);
        assertTrue(prefill.isLeakDetected(),
                "runtime leak detection must still fire when not shutting down");
    }

    // ──────────── Cluster setup ────────────

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode,
                              int decodeMaxConcurrency) throws IOException {
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();
        JavaMockEngineCluster.ClusterStats stats = new JavaMockEngineCluster.ClusterStats();

        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill-" + i, "127.0.0.1", "prefill",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100, stats,
                    JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, decodeMaxConcurrency);
            services.put(port, service);
            prefillServices.add(service);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = BASE_PORT + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode-" + i, "127.0.0.1", "decode",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100, stats,
                    JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, decodeMaxConcurrency);
            services.put(port, service);
            decodeServices.add(service);
        }
    }

    // ──────────── Helpers ────────────

    private long totalInflight() {
        return services.values().stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getInflightCount).sum();
    }

    private static void await(BooleanSupplier condition, long timeoutMs, String message)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(10);
        }
        fail(message);
    }

    private void enqueueBatch(JavaMockEngineCluster.FastRpcService prefill,
                              long batchId, int startRequestId, int count,
                              List<JavaMockEngineCluster.FastRpcService> decodeEngines) {
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[count];
        for (int i = 0; i < count; i++) {
            int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
            inputs[i] = inputWithDecode(startRequestId + i, 10, decodePort);
        }
        enqueue(prefill, batch(batchId, slot(0, inputs)));
    }

    private MockPerformanceModel model(String formula) throws Exception {
        // The decode hard concurrency gate is unconditional, so these
        // drain tests always exercise the queued-backlog + drain path.
        return MockEngineTestSupport.performanceModel(
                tempDir, formula, 1.0, 1.0, Map.of(), Map.of());
    }

}
