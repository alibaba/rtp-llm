package org.flexlb.mockengine;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.activeDecodeRequests;
import static org.flexlb.mockengine.MockEngineTestSupport.awaitDecodeQuiescence;
import static org.flexlb.mockengine.MockEngineTestSupport.decodeModel;
import static org.flexlb.mockengine.MockEngineTestSupport.decodePendingQueueSize;
import static org.flexlb.mockengine.MockEngineTestSupport.requestShape;
import static org.flexlb.mockengine.MockEngineTestSupport.scheduleDecodeCompletion;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        // Tiny cap 2: the unconditional gate must pin running at 2 and park
        // the remaining 8 in the pending queue.
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        int nRequests = 10;
        for (int i = 0; i < nRequests; i++) {
            assertTrue(scheduleDecodeCompletion(
                            decode, requestShape(model, 100L + i, 8), -1, null),
                    "the hard gate must never reject a decode request — "
                            + "excess parks in the unbounded waiting queue");
        }

        assertEquals(2, activeDecodeRequests(decode),
                "hard gate: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(nRequests - 2, decodePendingQueueSize(decode),
                "hard gate: excess requests must park in the decode pending queue");
        assertEquals(nRequests, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(nRequests, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // All requests must complete normally and drain every counter.
        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(0, activeDecodeRequests(decode));
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
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        // First 2 fill the concurrency slots.
        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 1L, 8), -1, null));
        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 2L, 8), -1, null));
        assertEquals(2, activeDecodeRequests(decode),
                "hard gate: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(0, decodePendingQueueSize(decode));

        // Next 4 are parked in the pending queue (accepted, not running) —
        // the queue is unbounded, so there is no overflow / rejection point.
        for (long rid = 3; rid <= 6; rid++) {
            assertTrue(scheduleDecodeCompletion(decode, requestShape(model, rid, 8), -1, null),
                    "request " + rid + " must be accepted into the pending queue");
        }
        assertEquals(2, activeDecodeRequests(decode),
                "queued requests must not consume concurrency slots");
        assertEquals(4, decodePendingQueueSize(decode),
                "requests 3..6 must be parked in the pending queue");
        assertEquals(6, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(6, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // Completions hand freed slots to queued requests until fully drained.
        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(0, activeDecodeRequests(decode));
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
        return MockEngineTestSupport.decodeService(
                model, port, services, scheduler, decodeMaxConcurrency);
    }

}
