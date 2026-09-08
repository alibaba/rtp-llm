package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Two-quadrant regression tests for the opt-in decode hard admission gate
 * (performance JSON {@code decode.max_pending_requests}):
 *
 * <ol>
 *   <li>{@link #absentCapKeepsLegacySoftAccounting} — when the key is ABSENT,
 *       behavior must be exactly the legacy one: decodeMaxConcurrency stays a
 *       soft accounting/reporting value, every request is admitted immediately
 *       (activeDecodeRequests may exceed the cap), nothing is queued and
 *       nothing is rejected (no decode-side rejection surface).</li>
 *   <li>{@link #optInCapGatesQueuesAndRejectsOverflow} — when the key is
 *       configured, decodeMaxConcurrency becomes a hard gate: excess requests
 *       are parked in the pending queue up to the configured cap, overflow is
 *       rejected (backpressure), and queued requests drain as slots free.</li>
 * </ol>
 */
class DecodePendingCapOptInTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63300;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private ExecutorService workerPool;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "decode-cap-optin-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r, "decode-cap-optin-worker");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        workerPool.shutdownNow();
        workerPool.awaitTermination(3, TimeUnit.SECONDS);
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Quadrant 1: key absent → legacy soft accounting ────────────

    @Test
    void absentCapKeepsLegacySoftAccounting() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms — all ten requests are
        // still in flight when the assertions run.
        MockPerformanceModel model = model(10_000.0, null);
        // Tiny cap 2 to prove it is NOT enforced in legacy mode.
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        int nRequests = 10;
        for (int i = 0; i < nRequests; i++) {
            assertTrue(invokeScheduleDecodeCompletion(
                            decode, shapeOf(model, 100L + i, 8), -1, null),
                    "legacy mode must never reject a decode request");
        }

        assertEquals(nRequests, getActiveDecodeRequests(decode),
                "legacy mode: activeDecodeRequests is soft accounting and may "
                        + "exceed decodeMaxConcurrency (no hard gate)");
        assertEquals(nRequests, decode.getRunningCount(),
                "legacy mode: all requests run immediately");
        assertEquals(nRequests, decode.getInflightCount(),
                "legacy mode: all requests are in flight");
        assertEquals(0, decodePendingQueueSize(decode),
                "legacy mode: nothing must be parked in the decode pending queue");

        // All requests must complete normally and drain every counter.
        awaitQuiescence(decode, 15_000);
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decode.getActiveKvTokens());
        assertEquals(nRequests, decode.getCompletedCount(),
                "legacy mode: every request completes (none rejected/queued)");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Quadrant 2: key configured → gate + queue + backpressure ────────────

    @Test
    void optInCapGatesQueuesAndRejectsOverflow() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms per request; cap 2
        // concurrency + queue capped at 3 → 6th request is overflow.
        MockPerformanceModel model = model(10_000.0, 3);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        // First 2 fill the concurrency slots.
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1L, 8), -1, null));
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 2L, 8), -1, null));
        assertEquals(2, getActiveDecodeRequests(decode),
                "gated mode: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(0, decodePendingQueueSize(decode));

        // Next 3 are parked in the pending queue (accepted, not running).
        for (long rid = 3; rid <= 5; rid++) {
            assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, rid, 8), -1, null),
                    "request " + rid + " must be accepted into the pending queue");
        }
        assertEquals(2, getActiveDecodeRequests(decode),
                "gated mode: queued requests must not consume concurrency slots");
        assertEquals(3, decodePendingQueueSize(decode),
                "requests 3..5 must be parked in the pending queue");
        assertEquals(5, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(5, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // 6th overflows the queue cap → rejected, nothing claimed.
        assertFalse(invokeScheduleDecodeCompletion(decode, shapeOf(model, 6L, 8), -1, null),
                "overflow beyond decode.max_pending_requests must be rejected");
        assertEquals(2, getActiveDecodeRequests(decode));
        assertEquals(3, decodePendingQueueSize(decode));
        assertEquals(5, decode.getInflightCount(),
                "a rejected request must not leave any counter claimed");
        assertEquals(5, decode.getRunningCount(),
                "a rejected request must not leave a runningTasks entry behind");

        // Completions hand freed slots to queued requests until fully drained.
        awaitQuiescence(decode, 30_000);
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decode.getActiveKvTokens());
        assertEquals(0, decodePendingQueueSize(decode));
        assertEquals(5, decode.getCompletedCount(),
                "all five accepted requests must complete via the drain chain");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Service / model helpers ────────────

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "decode-" + port, "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, decodeMaxConcurrency);
        services.put(port, service);
        return service;
    }

    /**
     * Builds a performance model with a single-point decode curve and an
     * optional {@code decode.max_pending_requests} opt-in (null = absent =
     * legacy soft accounting).
     */
    private MockPerformanceModel model(double decodeStepMs, Integer maxPendingRequests)
            throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        decodeConfig.put("scale", 1.0);
        decodeConfig.put("step_ms_by_batch", List.of(List.of(1, decodeStepMs)));
        if (maxPendingRequests != null) {
            decodeConfig.put("max_pending_requests", maxPendingRequests);
        }
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decodeConfig));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", "10"))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static MockPerformanceModel.RequestShape shapeOf(
            MockPerformanceModel model, long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return model.shape(input.build(), new MockLruBlockCache(100));
    }

    // ──────────── Quiescence helper ────────────

    private void awaitQuiescence(JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0 && service.getRunningCount() == 0
                    && getActiveDecodeRequests(service) == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("engine did not quiesce: inflight=" + service.getInflightCount()
                + " running=" + service.getRunningCount()
                + " activeDecode=" + getActiveDecodeRequests(service)
                + " kv=" + service.getActiveKvTokens());
    }

    // ──────────── Reflection helpers ────────────

    private static int getActiveDecodeRequests(JavaMockEngineCluster.FastRpcService service)
            throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("activeDecodeRequests");
        field.setAccessible(true);
        return ((AtomicInteger) field.get(service)).get();
    }

    private static int decodePendingQueueSize(JavaMockEngineCluster.FastRpcService service)
            throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("decodePendingQueue");
        field.setAccessible(true);
        return ((ArrayDeque<?>) field.get(service)).size();
    }

    private static boolean invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        Method method = JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod(
                "scheduleDecodeCompletion",
                MockPerformanceModel.RequestShape.class,
                long.class,
                LinkedBlockingQueue.class);
        method.setAccessible(true);
        return (Boolean) method.invoke(service, shape, batchId, responseQueue);
    }
}
