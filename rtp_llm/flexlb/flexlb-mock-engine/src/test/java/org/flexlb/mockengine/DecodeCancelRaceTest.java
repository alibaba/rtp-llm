package org.flexlb.mockengine;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.activeDecodeRequests;
import static org.flexlb.mockengine.MockEngineTestSupport.activeDecodeRequestsRef;
import static org.flexlb.mockengine.MockEngineTestSupport.awaitDecodeQuiescence;
import static org.flexlb.mockengine.MockEngineTestSupport.decodeModel;
import static org.flexlb.mockengine.MockEngineTestSupport.requestShape;
import static org.flexlb.mockengine.MockEngineTestSupport.scheduleDecodeCompletion;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrency regression tests for the two decode cancel races fixed in
 * {@code JavaMockEngineCluster.FastRpcService}:
 *
 * <ol>
 *   <li><b>P1-1 (permanent leak)</b> — {@code scheduleDecodeCompletion} used to
 *       call {@code runningTasks.putIfAbsent} OUTSIDE {@code decodeQueueLock}
 *       before claiming any counter. A cancel() landing in that window removed
 *       the entry and over-decremented pendingRequests / activeDecodeRequests /
 *       activeKvTokens that were never incremented; the admission then claimed
 *       them back and the completion (wasRunning=false) never released them —
 *       a permanent +1 on all three counters and a KV leak.
 *       {@link #cancelInAdmissionWindowNeverLeaksCounters} hammers that exact
 *       window and asserts all counters and KV settle at zero with no LEAK.</li>
 *   <li><b>P1-2 (transient over-admission)</b> — cancel() used to release the
 *       running slot (activeDecodeRequests--, KV release) outside the lock and
 *       only then take {@code decodeQueueLock} to drain, so a concurrent
 *       admission could grab the freed slot before the drain re-consumed it,
 *       transiently pushing activeDecodeRequests to cap+1.
 *       {@link #activeDecodeRequestsNeverExceedsCapUnderCancelStorm} samples
 *       the counter continuously under a schedule/cancel/completion storm and
 *       asserts it never exceeds decodeMaxConcurrency in gated mode.</li>
 * </ol>
 */
class DecodeCancelRaceTest {

    private static final int BASE_PORT = 63200;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private ExecutorService workerPool;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(8, runnable -> {
            Thread thread = new Thread(runnable, "decode-cancel-race-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r, "decode-cancel-race-worker");
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

    // ──────────── Test 1: P1-1 — cancel in the admission window must not leak ────────────

    /**
     * Barrier-aligns a scheduleDecodeCompletion call against a cancel() for the
     * same requestId over many iterations, maximising the chance that the cancel
     * lands exactly between the runningTasks claim and the counter claims (the
     * pre-fix leak window). After every request has either completed or been
     * cancelled, all three counters and activeKvTokens must be exactly zero and
     * checkLeakDrain must not flag a LEAK.
     */
    @Test
    void cancelInAdmissionWindowNeverLeaksCounters() throws Exception {
        // decode step 50 × sleep_scale 0.1 = 5 ms — fast completions so the
        // stress loop and the final quiescence stay quick.
        MockPerformanceModel model = decodeModel(tempDir, 50.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 132);

        int iterations = 300;
        for (int i = 0; i < iterations; i++) {
            long requestId = 1_000L + i;
            MockPerformanceModel.RequestShape shape = requestShape(model, requestId, 16);
            CyclicBarrier barrier = new CyclicBarrier(2);
            Future<?> scheduleFuture = workerPool.submit(() -> {
                barrier.await();
                scheduleDecodeCompletion(decode, shape, -1, null);
                return null;
            });
            Future<?> cancelFuture = workerPool.submit(() -> {
                barrier.await();
                decode.cancel(requestId);
                return null;
            });
            scheduleFuture.get(5, TimeUnit.SECONDS);
            cancelFuture.get(5, TimeUnit.SECONDS);
        }

        awaitDecodeQuiescence(decode, 10_000);

        assertEquals(0, activeDecodeRequests(decode),
                "activeDecodeRequests must settle at 0 (no permanent slot leak)");
        assertEquals(0, decode.getActiveKvTokens(),
                "activeKvTokens must settle at 0 (no permanent KV leak)");
        assertEquals(0, decode.getInflightCount(),
                "pendingRequests must settle at 0");
        assertEquals(0, decode.getRunningCount(),
                "runningTasks must be empty");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected(),
                "checkLeakDrain must not flag a LEAK after the cancel storm");
    }

    // ──────────── Test 2: P1-2 — cap never exceeded, even transiently ────────────

    /**
     * Gated mode (decode.max_pending_requests = 0 → hard gate + unbounded
     * queue) with a tiny cap. A dedicated sampler thread continuously reads
     * activeDecodeRequests while schedule / cancel / completion drains race;
     * the pre-fix code allowed a transient cap+1 here because cancel released
     * the slot outside the lock before draining.
     */
    @Test
    void activeDecodeRequestsNeverExceedsCapUnderCancelStorm() throws Exception {
        // decode step 100 × sleep_scale 0.1 = 10 ms — completions fire while
        // the storm is still running, exercising the completion drain too.
        MockPerformanceModel model = decodeModel(tempDir, 100.0, 0);
        int cap = 4;
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, cap);

        AtomicInteger counter = activeDecodeRequestsRef(decode);
        AtomicInteger maxObserved = new AtomicInteger();
        AtomicBoolean sampling = new AtomicBoolean(true);
        Thread sampler = new Thread(() -> {
            while (sampling.get()) {
                maxObserved.accumulateAndGet(counter.get(), Math::max);
                Thread.onSpinWait();
            }
        }, "decode-cap-sampler");
        sampler.setDaemon(true);
        sampler.start();

        int nRequests = 200;
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch allDone = new CountDownLatch(nRequests + nRequests / 2);
        for (int i = 0; i < nRequests; i++) {
            long requestId = 5_000L + i;
            MockPerformanceModel.RequestShape shape = requestShape(model, requestId, 8);
            workerPool.submit(() -> {
                try {
                    startGate.await();
                    scheduleDecodeCompletion(decode, shape, -1, null);
                } catch (Throwable ignored) {
                    // surfaced via the final counter asserts
                } finally {
                    allDone.countDown();
                }
            });
            if (i % 2 == 0) {
                workerPool.submit(() -> {
                    try {
                        startGate.await();
                        decode.cancel(requestId);
                    } catch (Throwable ignored) {
                        // surfaced via the final counter asserts
                    } finally {
                        allDone.countDown();
                    }
                });
            }
        }
        startGate.countDown();
        assertTrue(allDone.await(30, TimeUnit.SECONDS),
                "storm should finish within 30s, remaining: " + allDone.getCount());

        awaitDecodeQuiescence(decode, 20_000);
        sampling.set(false);
        sampler.join(1_000);

        assertTrue(maxObserved.get() <= cap,
                "activeDecodeRequests must never exceed cap " + cap
                        + " — observed max " + maxObserved.get());
        assertEquals(0, activeDecodeRequests(decode),
                "activeDecodeRequests must settle at 0");
        assertEquals(0, decode.getActiveKvTokens(),
                "activeKvTokens must settle at 0");
        assertEquals(0, decode.getInflightCount(),
                "pendingRequests must settle at 0");
        assertEquals(0, decode.getRunningCount(),
                "runningTasks must be empty");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected(),
                "checkLeakDrain must not flag a LEAK after the cancel storm");
    }

    // ──────────── Service / model helpers ────────────

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        return MockEngineTestSupport.decodeService(
                model, port, services, scheduler, decodeMaxConcurrency);
    }

}
