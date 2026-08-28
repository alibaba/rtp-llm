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
 * Regression tests for the unconditional decode hard admission gate
 * (production waiting_streams_ semantics, no configuration key):
 *
 * <ol>
 *   <li>{@link #hardGateParksExcessInUnboundedQueue} — decodeMaxConcurrency
 *       is a hard cap: excess requests park in the (unbounded) pending queue,
 *       activeDecodeRequests never exceeds the cap, nothing is ever rejected
 *       on the decode side, and queued requests keep their runningTasks
 *       claim (dedup guard) while counting as in flight.</li>
 *   <li>{@link #queuedRequestsDrainThroughCompletionChain} — completions
 *       hand freed slots to queued requests one-for-one until the queue is
 *       fully drained; every accepted request completes, all counters settle
 *       at zero and checkLeakDrain flags nothing.</li>
 * </ol>
 */
class DecodePendingQueueHardGateTest {

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
            Thread thread = new Thread(runnable, "decode-hard-gate-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r, "decode-hard-gate-worker");
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

    // ──────────── Test 1: hard cap parks excess in the unbounded queue ────────────

    @Test
    void hardGateParksExcessInUnboundedQueue() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms — all ten requests are
        // still in flight when the assertions run.
        MockPerformanceModel model = model(10_000.0);
        // Tiny cap 2: the unconditional gate must pin running at 2 and park
        // the remaining 8 in the pending queue.
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        int nRequests = 10;
        for (int i = 0; i < nRequests; i++) {
            assertTrue(invokeScheduleDecodeCompletion(
                            decode, shapeOf(model, 100L + i, 8), -1, null),
                    "the hard gate must never reject a decode request — "
                            + "excess parks in the unbounded waiting queue");
        }

        assertEquals(2, getActiveDecodeRequests(decode),
                "hard gate: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(nRequests - 2, decodePendingQueueSize(decode),
                "hard gate: excess requests must park in the decode pending queue");
        assertEquals(nRequests, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(nRequests, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // All requests must complete normally and drain every counter.
        awaitQuiescence(decode, 30_000);
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decode.getActiveKvTokens());
        assertEquals(0, decodePendingQueueSize(decode));
        assertEquals(nRequests, decode.getCompletedCount(),
                "every accepted request must complete (none rejected)");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Test 2: queued requests drain as completions free slots ────────────

    @Test
    void queuedRequestsDrainThroughCompletionChain() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms per request; cap 2
        // concurrency → requests 1-2 run, 3-6 park and drain in order.
        MockPerformanceModel model = model(10_000.0);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        // First 2 fill the concurrency slots.
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1L, 8), -1, null));
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 2L, 8), -1, null));
        assertEquals(2, getActiveDecodeRequests(decode),
                "hard gate: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(0, decodePendingQueueSize(decode));

        // Next 4 are parked in the pending queue (accepted, not running) —
        // the queue is unbounded, so there is no overflow / rejection point.
        for (long rid = 3; rid <= 6; rid++) {
            assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, rid, 8), -1, null),
                    "request " + rid + " must be accepted into the pending queue");
        }
        assertEquals(2, getActiveDecodeRequests(decode),
                "queued requests must not consume concurrency slots");
        assertEquals(4, decodePendingQueueSize(decode),
                "requests 3..6 must be parked in the pending queue");
        assertEquals(6, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(6, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // Completions hand freed slots to queued requests until fully drained.
        awaitQuiescence(decode, 30_000);
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decode.getActiveKvTokens());
        assertEquals(0, decodePendingQueueSize(decode));
        assertEquals(6, decode.getCompletedCount(),
                "all six accepted requests must complete via the drain chain");
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
     * Builds a performance model with a single-point decode curve. The decode
     * hard concurrency gate is unconditional (no opt-in key exists anymore).
     */
    private MockPerformanceModel model(double decodeStepMs)
            throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        decodeConfig.put("scale", 1.0);
        decodeConfig.put("step_ms_by_batch", List.of(List.of(1, decodeStepMs)));
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decodeConfig));
        MockMasterConfig.writeWithPrefillExpression(master, "10");
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
