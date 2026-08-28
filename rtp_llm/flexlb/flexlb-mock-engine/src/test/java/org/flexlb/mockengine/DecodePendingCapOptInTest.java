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
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        // Tiny cap 2 to prove it is NOT enforced in legacy mode.
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        int nRequests = 10;
        for (int i = 0; i < nRequests; i++) {
            assertTrue(scheduleDecodeCompletion(
                            decode, requestShape(model, 100L + i, 8), -1, null),
                    "legacy mode must never reject a decode request");
        }

        assertEquals(nRequests, activeDecodeRequests(decode),
                "legacy mode: activeDecodeRequests is soft accounting and may "
                        + "exceed decodeMaxConcurrency (no hard gate)");
        assertEquals(nRequests, decode.getRunningCount(),
                "legacy mode: all requests run immediately");
        assertEquals(nRequests, decode.getInflightCount(),
                "legacy mode: all requests are in flight");
        assertEquals(0, decodePendingQueueSize(decode),
                "legacy mode: nothing must be parked in the decode pending queue");

        // All requests must complete normally and drain every counter.
        awaitDecodeQuiescence(decode, 15_000);
        assertEquals(0, activeDecodeRequests(decode));
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
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, 3);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        // First 2 fill the concurrency slots.
        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 1L, 8), -1, null));
        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 2L, 8), -1, null));
        assertEquals(2, activeDecodeRequests(decode),
                "gated mode: activeDecodeRequests capped at decodeMaxConcurrency");
        assertEquals(0, decodePendingQueueSize(decode));

        // Next 3 are parked in the pending queue (accepted, not running).
        for (long rid = 3; rid <= 5; rid++) {
            assertTrue(scheduleDecodeCompletion(decode, requestShape(model, rid, 8), -1, null),
                    "request " + rid + " must be accepted into the pending queue");
        }
        assertEquals(2, activeDecodeRequests(decode),
                "gated mode: queued requests must not consume concurrency slots");
        assertEquals(3, decodePendingQueueSize(decode),
                "requests 3..5 must be parked in the pending queue");
        assertEquals(5, decode.getInflightCount(),
                "running + queued requests all count as in flight");
        assertEquals(5, decode.getRunningCount(),
                "queued requests keep their runningTasks claim (dedup guard)");

        // 6th overflows the queue cap → rejected, nothing claimed.
        assertFalse(scheduleDecodeCompletion(decode, requestShape(model, 6L, 8), -1, null),
                "overflow beyond decode.max_pending_requests must be rejected");
        assertEquals(2, activeDecodeRequests(decode));
        assertEquals(3, decodePendingQueueSize(decode));
        assertEquals(5, decode.getInflightCount(),
                "a rejected request must not leave any counter claimed");
        assertEquals(5, decode.getRunningCount(),
                "a rejected request must not leave a runningTasks entry behind");

        // Completions hand freed slots to queued requests until fully drained.
        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(0, activeDecodeRequests(decode));
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
        return MockEngineTestSupport.decodeService(
                model, port, services, scheduler, decodeMaxConcurrency);
    }

}
